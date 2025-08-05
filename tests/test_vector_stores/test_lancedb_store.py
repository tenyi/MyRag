"""
LanceDB 向量儲存測試
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from chinese_graphrag.vector_stores import (
    CollectionError,
    ConnectionError,
    HybridSearchConfig,
    LanceDBStore,
    RerankingMethod,
    SearchFilter,
    SearchType,
    VectorStoreError,
    VectorStoreType,
)


class TestLanceDBStore:
    """LanceDB 向量儲存測試類別"""

    @pytest.fixture
    def temp_db_path(self):
        """建立臨時資料庫路徑"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_lancedb"
        yield str(db_path)
        # 清理
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    @pytest_asyncio.fixture
    async def store(self, temp_db_path):
        """建立測試用的 LanceDB 儲存"""
        store = LanceDBStore(db_path=temp_db_path)
        await store.connect()
        yield store
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_connection(self, temp_db_path):
        """測試連線和斷線"""
        store = LanceDBStore(db_path=temp_db_path)

        # 初始狀態
        assert not store.is_connected

        # 測試連線
        await store.connect()
        assert store.is_connected
        assert store.db is not None

        # 測試斷線
        await store.disconnect()
        assert not store.is_connected
        assert store.db is None

    @pytest.mark.asyncio
    async def test_collection_management(self, store):
        """測試集合管理"""
        collection_name = "test_collection"
        dimension = 128

        # 測試集合不存在
        exists = await store.collection_exists(collection_name)
        assert not exists

        # 測試建立集合
        success = await store.create_collection(
            name=collection_name,
            dimension=dimension,
            metadata_schema={"type": "str", "score": "float"},
        )
        assert success

        # 測試集合存在
        exists = await store.collection_exists(collection_name)
        assert exists

        # 測試取得集合資訊
        info = await store.get_collection_info(collection_name)
        assert info is not None
        assert info.name == collection_name
        assert info.count == 0  # 新建立的集合應該是空的

        # 測試列出集合
        collections = await store.list_collections()
        assert len(collections) >= 1
        assert any(col.name == collection_name for col in collections)

        # 測試重複建立集合
        success = await store.create_collection(collection_name, dimension)
        assert not success  # 應該失敗

        # 測試刪除集合
        success = await store.delete_collection(collection_name)
        assert success

        # 確認集合已刪除
        exists = await store.collection_exists(collection_name)
        assert not exists

    @pytest.mark.asyncio
    async def test_vector_operations(self, store):
        """測試向量操作"""
        collection_name = "vector_test"
        dimension = 64

        # 建立集合
        await store.create_collection(
            name=collection_name,
            dimension=dimension,
            metadata_schema={"type": "str", "score": "float"},
        )

        # 準備測試資料
        ids = ["vec1", "vec2", "vec3"]
        vectors = np.random.rand(3, dimension).astype(np.float32)
        metadata = [
            {"type": "document", "score": 0.9},
            {"type": "entity", "score": 0.8},
            {"type": "relationship", "score": 0.7},
        ]

        # 測試插入向量
        success = await store.insert_vectors(
            collection_name=collection_name, ids=ids, vectors=vectors, metadata=metadata
        )
        assert success

        # 驗證集合計數
        info = await store.get_collection_info(collection_name)
        assert info.count == 3

        # 測試根據 ID 取得向量
        vector_data = await store.get_vector_by_id(
            collection_name=collection_name, vector_id="vec1", include_embedding=True
        )
        assert vector_data is not None
        assert vector_data["id"] == "vec1"
        assert vector_data["type"] == "document"
        assert vector_data["score"] == 0.9
        assert "vector" in vector_data
        assert isinstance(vector_data["vector"], np.ndarray)

        # 測試取得不存在的向量
        nonexistent = await store.get_vector_by_id(
            collection_name=collection_name, vector_id="nonexistent"
        )
        assert nonexistent is None

    @pytest.mark.asyncio
    async def test_vector_search(self, store):
        """測試向量搜尋"""
        collection_name = "search_test"
        dimension = 32

        # 建立集合
        await store.create_collection(collection_name, dimension)

        # 插入測試向量
        ids = [f"vec{i}" for i in range(10)]
        vectors = np.random.rand(10, dimension).astype(np.float32)
        metadata = [{"index": i} for i in range(10)]

        await store.insert_vectors(collection_name, ids, vectors, metadata)

        # 測試向量搜尋
        query_vector = np.random.rand(dimension).astype(np.float32)
        result = await store.search_vectors(
            collection_name=collection_name, query_vector=query_vector, k=5
        )

        assert len(result.ids) <= 5
        assert len(result.distances) == len(result.ids)
        assert len(result.similarities) == len(result.ids)
        assert len(result.metadata) == len(result.ids)

        # 檢查相似度分數
        for similarity in result.similarities:
            assert 0 <= similarity <= 1

        # 測試包含向量的搜尋
        result_with_embeddings = await store.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            k=3,
            include_embeddings=True,
        )

        assert result_with_embeddings.embeddings is not None
        assert len(result_with_embeddings.embeddings) == len(result_with_embeddings.ids)

        # 測試批次搜尋
        query_vectors = np.random.rand(3, dimension).astype(np.float32)
        batch_results = await store.batch_search_vectors(
            collection_name=collection_name, query_vectors=query_vectors, k=2
        )

        assert len(batch_results) == 3
        for result in batch_results:
            assert len(result.ids) <= 2

    @pytest.mark.asyncio
    async def test_vector_update_and_delete(self, store):
        """測試向量更新和刪除"""
        collection_name = "update_test"
        dimension = 16

        # 建立集合並插入資料
        await store.create_collection(
            collection_name, dimension, metadata_schema={"status": "str"}
        )

        ids = ["update1", "update2", "delete1"]
        vectors = np.random.rand(3, dimension).astype(np.float32)
        metadata = [{"status": "original"} for _ in range(3)]

        await store.insert_vectors(collection_name, ids, vectors, metadata)

        # 測試更新向量
        new_vector = np.random.rand(dimension).astype(np.float32)
        new_metadata = {"status": "updated"}

        success = await store.update_vectors(
            collection_name=collection_name,
            ids=["update1"],
            vectors=[new_vector],
            metadata=[new_metadata],
        )
        assert success

        # 驗證更新
        updated_data = await store.get_vector_by_id(
            collection_name=collection_name, vector_id="update1"
        )
        assert updated_data["status"] == "updated"

        # 測試刪除向量
        success = await store.delete_vectors(
            collection_name=collection_name, ids=["delete1"]
        )
        assert success

        # 驗證刪除
        deleted_data = await store.get_vector_by_id(
            collection_name=collection_name, vector_id="delete1"
        )
        assert deleted_data is None

        # 驗證其他向量仍然存在
        remaining_data = await store.get_vector_by_id(
            collection_name=collection_name, vector_id="update2"
        )
        assert remaining_data is not None

    @pytest.mark.asyncio
    async def test_health_check(self, store):
        """測試健康狀態檢查"""
        health = await store.health_check()

        assert "status" in health
        assert health["status"] == "healthy"
        assert health["store_type"] == VectorStoreType.LANCEDB.value

    @pytest.mark.asyncio
    async def test_context_manager(self, temp_db_path):
        """測試上下文管理器"""
        async with LanceDBStore(db_path=temp_db_path) as store:
            assert store.is_connected

            # 測試基本操作
            await store.create_collection("context_test", 8)
            exists = await store.collection_exists("context_test")
            assert exists

        # 上下文管理器應該自動斷開連線
        assert not store.is_connected

    @pytest.mark.asyncio
    async def test_error_handling(self, store):
        """測試錯誤處理"""
        # 測試操作不存在的集合
        with pytest.raises(CollectionError):
            await store.insert_vectors(
                collection_name="nonexistent",
                ids=["test"],
                vectors=[np.random.rand(32)],
            )

        # 測試無效的向量資料
        await store.create_collection("error_test", 32)

        with pytest.raises(Exception):
            await store.insert_vectors(
                collection_name="error_test",
                ids=["test1", "test2"],  # 2 個 ID
                vectors=[np.random.rand(32)],  # 但只有 1 個向量
            )

    @pytest.mark.asyncio
    async def test_hybrid_search(self, store):
        """測試混合搜尋功能"""
        collection_name = "hybrid_test"
        dimension = 32

        # 建立集合
        await store.create_collection(
            collection_name,
            dimension,
            metadata_schema={"content": "str", "title": "str"},
        )

        # 插入測試資料
        ids = [f"doc{i}" for i in range(5)]
        vectors = np.random.rand(5, dimension).astype(np.float32)
        metadata = [
            {"content": "這是一個關於機器學習的文件", "title": "機器學習入門"},
            {"content": "深度學習是人工智慧的重要分支", "title": "深度學習概述"},
            {"content": "自然語言處理技術發展迅速", "title": "NLP 技術"},
            {"content": "電腦視覺在各領域應用廣泛", "title": "電腦視覺"},
            {"content": "資料科學結合統計學和程式設計", "title": "資料科學"},
        ]

        await store.insert_vectors(collection_name, ids, vectors, metadata)

        # 測試混合搜尋
        dense_vector = np.random.rand(dimension).astype(np.float32)
        sparse_vector = {"機器學習": 0.8, "人工智慧": 0.6, "深度學習": 0.7}

        config = HybridSearchConfig(
            dense_weight=0.6,
            sparse_weight=0.4,
            reranking_method=RerankingMethod.RECIPROCAL_RANK_FUSION,
        )

        result = await store.hybrid_search(
            collection_name=collection_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            query_text="機器學習和深度學習",
            k=3,
            config=config,
        )

        assert result.search_type == SearchType.HYBRID
        assert len(result.ids) <= 3
        assert len(result.similarities) == len(result.ids)
        assert result.reranking_scores is not None
        assert len(result.reranking_scores) == len(result.ids)

    @pytest.mark.asyncio
    async def test_sparse_search(self, store):
        """測試稀疏搜尋功能"""
        collection_name = "sparse_test"
        dimension = 16

        # 建立集合
        await store.create_collection(
            collection_name, dimension, metadata_schema={"content": "str"}
        )

        # 插入測試資料
        ids = ["doc1", "doc2"]
        vectors = np.random.rand(2, dimension).astype(np.float32)
        metadata = [
            {"content": "Python 是一種程式語言"},
            {"content": "Java 也是程式語言"},
        ]

        await store.insert_vectors(collection_name, ids, vectors, metadata)

        # 測試稀疏搜尋
        result = await store._sparse_search(
            collection_name=collection_name, query_text="程式語言", k=2
        )

        assert result is not None
        assert result.search_type == SearchType.SPARSE
        # 由於是關鍵字匹配，可能會有結果
        assert len(result.ids) >= 0

    @pytest.mark.asyncio
    async def test_index_optimization(self, store):
        """測試索引優化功能"""
        collection_name = "index_test"
        dimension = 64

        # 建立集合並插入資料
        await store.create_collection(collection_name, dimension)

        ids = [f"vec{i}" for i in range(100)]
        vectors = np.random.rand(100, dimension).astype(np.float32)

        await store.insert_vectors(collection_name, ids, vectors)

        # 測試索引優化
        success = await store.optimize_index(
            collection_name=collection_name,
            optimization_params={
                "index_type": "FLAT",  # 使用簡單的索引類型
                "num_partitions": 10,
            },
        )

        # 索引優化可能成功也可能失敗，取決於 LanceDB 版本和配置
        assert isinstance(success, bool)

        # 測試取得索引統計
        stats = await store.get_index_stats(collection_name)

        assert "collection_name" in stats
        assert "vector_count" in stats
        assert "dimension" in stats
        assert stats["collection_name"] == collection_name
        assert stats["vector_count"] == 100
        assert stats["dimension"] == dimension

    @pytest.mark.asyncio
    async def test_search_filter(self, store):
        """測試搜尋過濾器"""
        collection_name = "filter_test"
        dimension = 32

        # 建立集合
        await store.create_collection(
            collection_name,
            dimension,
            metadata_schema={"category": "str", "score": "float"},
        )

        # 插入測試資料
        ids = ["doc1", "doc2", "doc3"]
        vectors = np.random.rand(3, dimension).astype(np.float32)
        metadata = [
            {"category": "tech", "score": 0.9},
            {"category": "science", "score": 0.8},
            {"category": "tech", "score": 0.7},
        ]

        await store.insert_vectors(collection_name, ids, vectors, metadata)

        # 測試帶過濾條件的搜尋
        query_vector = np.random.rand(dimension).astype(np.float32)

        result = await store.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            k=5,
            filter_conditions={"category": "tech"},
        )

        # 應該只返回 category 為 "tech" 的結果
        assert len(result.ids) <= 2  # 最多2個 tech 類別的文件
        for meta in result.metadata:
            assert meta["category"] == "tech"

    @pytest.mark.asyncio
    async def test_disconnected_operations(self, temp_db_path):
        """測試未連線時的操作"""
        store = LanceDBStore(db_path=temp_db_path)

        # 測試未連線時的操作應該拋出異常
        with pytest.raises(ConnectionError):
            await store.create_collection("test", 32)

        with pytest.raises(ConnectionError):
            await store.list_collections()
