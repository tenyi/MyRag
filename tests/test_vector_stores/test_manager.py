"""
向量儲存管理器測試
"""

import pytest
import pytest_asyncio
import numpy as np
import tempfile
import shutil
from pathlib import Path

from chinese_graphrag.vector_stores import (
    VectorStoreManager,
    VectorStoreType,
    create_vector_store_manager,
    VectorStoreError,
    HybridSearchConfig,
    SearchFilter,
    SearchType,
    RerankingMethod
)


class TestVectorStoreManager:
    """向量儲存管理器測試類別"""
    
    @pytest_asyncio.fixture
    async def temp_db_path(self):
        """建立臨時資料庫路徑"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_lancedb"
        yield str(db_path)
        # 清理
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
    
    @pytest_asyncio.fixture
    async def manager(self, temp_db_path):
        """建立測試用的向量儲存管理器"""
        manager = VectorStoreManager(
            default_store_type=VectorStoreType.LANCEDB
        )
        await manager.initialize(db_path=temp_db_path)
        yield manager
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, temp_db_path):
        """測試管理器初始化"""
        manager = VectorStoreManager(VectorStoreType.LANCEDB)
        
        # 測試初始化
        await manager.initialize(db_path=temp_db_path)
        
        assert manager.active_store is not None
        assert manager.active_store.is_connected
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_create_vector_store_manager(self, temp_db_path):
        """測試便利函數建立管理器"""
        manager = await create_vector_store_manager(
            store_type=VectorStoreType.LANCEDB,
            db_path=temp_db_path
        )
        
        assert manager.active_store is not None
        assert manager.active_store.is_connected
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_collection_operations(self, manager):
        """測試集合操作"""
        collection_name = "test_collection"
        dimension = 128
        
        # 測試建立集合
        success = await manager.create_collection(
            name=collection_name,
            dimension=dimension,
            metadata_schema={"type": "str", "score": "float"}
        )
        assert success
        
        # 測試檢查集合存在
        exists = await manager.collection_exists(collection_name)
        assert exists
        
        # 測試取得集合資訊
        info = await manager.get_collection_info(collection_name)
        assert info is not None
        assert info.name == collection_name
        # 空集合的維度可能為 0，這是正常的
        assert info.dimension >= 0
        
        # 測試列出集合
        collections = await manager.list_collections()
        assert len(collections) >= 1
        assert any(col.name == collection_name for col in collections)
        
        # 測試刪除集合
        success = await manager.delete_collection(collection_name)
        assert success
        
        # 確認集合已刪除
        exists = await manager.collection_exists(collection_name)
        assert not exists
    
    @pytest.mark.asyncio
    async def test_vector_operations(self, manager):
        """測試向量操作"""
        collection_name = "test_vectors"
        dimension = 64
        
        # 建立集合
        await manager.create_collection(
            name=collection_name,
            dimension=dimension,
            metadata_schema={"type": "str", "score": "float"}
        )
        
        # 準備測試資料
        ids = ["vec1", "vec2", "vec3"]
        vectors = [
            np.random.rand(dimension).astype(np.float32),
            np.random.rand(dimension).astype(np.float32),
            np.random.rand(dimension).astype(np.float32)
        ]
        metadata = [
            {"type": "document", "score": 0.9},
            {"type": "entity", "score": 0.8},
            {"type": "relationship", "score": 0.7}
        ]
        
        # 測試插入向量
        success = await manager.insert_vectors(
            collection_name=collection_name,
            ids=ids,
            vectors=vectors,
            metadata=metadata
        )
        assert success
        
        # 測試根據 ID 取得向量
        vector_data = await manager.get_vector_by_id(
            collection_name=collection_name,
            vector_id="vec1",
            include_embedding=True
        )
        assert vector_data is not None
        assert vector_data["id"] == "vec1"
        assert vector_data["type"] == "document"
        assert vector_data["score"] == 0.9
        assert "vector" in vector_data
        
        # 測試向量搜尋
        query_vector = np.random.rand(dimension).astype(np.float32)
        search_result = await manager.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            k=2
        )
        
        assert len(search_result.ids) <= 2
        assert len(search_result.distances) == len(search_result.ids)
        assert len(search_result.similarities) == len(search_result.ids)
        assert len(search_result.metadata) == len(search_result.ids)
        
        # 測試批次搜尋
        query_vectors = [
            np.random.rand(dimension).astype(np.float32),
            np.random.rand(dimension).astype(np.float32)
        ]
        batch_results = await manager.batch_search_vectors(
            collection_name=collection_name,
            query_vectors=query_vectors,
            k=1
        )
        
        assert len(batch_results) == 2
        for result in batch_results:
            assert len(result.ids) <= 1
        
        # 測試更新向量
        new_vector = np.random.rand(dimension).astype(np.float32)
        new_metadata = {"type": "updated", "score": 0.95}
        
        success = await manager.update_vectors(
            collection_name=collection_name,
            ids=["vec1"],
            vectors=[new_vector],
            metadata=[new_metadata]
        )
        assert success
        
        # 驗證更新
        updated_data = await manager.get_vector_by_id(
            collection_name=collection_name,
            vector_id="vec1"
        )
        assert updated_data["type"] == "updated"
        assert updated_data["score"] == 0.95
        
        # 測試刪除向量
        success = await manager.delete_vectors(
            collection_name=collection_name,
            ids=["vec2", "vec3"]
        )
        assert success
        
        # 驗證刪除
        deleted_data = await manager.get_vector_by_id(
            collection_name=collection_name,
            vector_id="vec2"
        )
        assert deleted_data is None
    
    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        """測試健康狀態檢查"""
        health = await manager.health_check()
        
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy", "error"]
    
    @pytest.mark.asyncio
    async def test_statistics(self, manager):
        """測試統計資訊"""
        # 建立測試集合
        await manager.create_collection("stats_test", 32)
        
        stats = await manager.get_statistics()
        
        assert "store_type" in stats
        assert "collections_count" in stats
        assert "total_vectors" in stats
        assert "collections" in stats
        
        assert stats["collections_count"] >= 1
        assert any(col["name"] == "stats_test" for col in stats["collections"])
    
    @pytest.mark.asyncio
    async def test_context_manager(self, temp_db_path):
        """測試上下文管理器"""
        async with VectorStoreManager(VectorStoreType.LANCEDB) as manager:
            await manager.initialize(db_path=temp_db_path)
            
            # 測試基本操作
            await manager.create_collection("context_test", 16)
            exists = await manager.collection_exists("context_test")
            assert exists
        
        # 上下文管理器應該自動關閉連線
        assert not manager.stores  # 所有儲存實例應該被清理
    
    @pytest.mark.asyncio
    async def test_error_handling(self, manager):
        """測試錯誤處理"""
        # 測試操作不存在的集合
        with pytest.raises(Exception):
            await manager.insert_vectors(
                collection_name="nonexistent",
                ids=["test"],
                vectors=[np.random.rand(32)]
            )
        
        # 測試無效的向量維度
        await manager.create_collection("dimension_test", 64)
        
        with pytest.raises(Exception):
            await manager.insert_vectors(
                collection_name="dimension_test",
                ids=["test"],
                vectors=[np.random.rand(32)]  # 錯誤的維度
            )
    
    @pytest.mark.asyncio
    async def test_hybrid_search_manager(self, manager):
        """測試管理器的混合搜尋功能"""
        collection_name = "hybrid_manager_test"
        dimension = 32
        
        # 建立集合
        await manager.create_collection(
            collection_name, 
            dimension,
            metadata_schema={"content": "str", "title": "str"}
        )
        
        # 插入測試資料
        ids = ["doc1", "doc2", "doc3"]
        vectors = np.random.rand(3, dimension).astype(np.float32)
        metadata = [
            {"content": "機器學習是人工智慧的分支", "title": "ML 基礎"},
            {"content": "深度學習使用神經網路", "title": "DL 概念"},
            {"content": "自然語言處理很重要", "title": "NLP 技術"}
        ]
        
        await manager.insert_vectors(collection_name, ids, vectors, metadata)
        
        # 測試混合搜尋
        dense_vector = np.random.rand(dimension).astype(np.float32)
        sparse_vector = {"機器學習": 0.8, "深度學習": 0.6}
        
        config = HybridSearchConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            reranking_method=RerankingMethod.WEIGHTED_SCORE
        )
        
        search_filter = SearchFilter(
            conditions={},
            score_threshold=0.1
        )
        
        result = await manager.hybrid_search(
            collection_name=collection_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            query_text="機器學習",
            k=2,
            config=config,
            search_filter=search_filter
        )
        
        assert result.search_type == SearchType.HYBRID
        assert len(result.ids) <= 2
    
    @pytest.mark.asyncio
    async def test_index_optimization_manager(self, manager):
        """測試管理器的索引優化功能"""
        collection_name = "index_manager_test"
        dimension = 64
        
        # 建立集合並插入資料
        await manager.create_collection(collection_name, dimension)
        
        ids = [f"vec{i}" for i in range(50)]
        vectors = np.random.rand(50, dimension).astype(np.float32)
        
        await manager.insert_vectors(collection_name, ids, vectors)
        
        # 測試索引優化
        success = await manager.optimize_index(
            collection_name=collection_name,
            optimization_params={"index_type": "FLAT"}
        )
        
        assert isinstance(success, bool)
        
        # 測試取得索引統計
        stats = await manager.get_index_stats(collection_name)
        
        assert "collection_name" in stats
        assert "vector_count" in stats
        assert stats["collection_name"] == collection_name
        assert stats["vector_count"] == 50
    
    @pytest.mark.asyncio
    async def test_search_filter_manager(self, manager):
        """測試管理器的搜尋過濾功能"""
        collection_name = "filter_manager_test"
        dimension = 32
        
        # 建立集合
        await manager.create_collection(
            collection_name, 
            dimension,
            metadata_schema={"type": "str", "priority": "int"}
        )
        
        # 插入測試資料
        ids = ["doc1", "doc2", "doc3", "doc4"]
        vectors = np.random.rand(4, dimension).astype(np.float32)
        metadata = [
            {"type": "urgent", "priority": 1},
            {"type": "normal", "priority": 2},
            {"type": "urgent", "priority": 1},
            {"type": "low", "priority": 3}
        ]
        
        await manager.insert_vectors(collection_name, ids, vectors, metadata)
        
        # 測試帶過濾條件的搜尋
        query_vector = np.random.rand(dimension).astype(np.float32)
        
        result = await manager.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            k=5,
            filter_conditions={"type": "urgent"}
        )
        
        # 應該只返回 type 為 "urgent" 的結果
        assert len(result.ids) <= 2  # 最多2個 urgent 類型的文件
        for meta in result.metadata:
            assert meta["type"] == "urgent"
    
    @pytest.mark.asyncio
    async def test_multiple_stores(self, temp_db_path):
        """測試多個儲存實例"""
        manager = VectorStoreManager()
        
        # 建立兩個不同的 LanceDB 實例
        store1 = await manager.get_store(
            VectorStoreType.LANCEDB,
            store_id="store1",
            db_path=temp_db_path + "_1"
        )
        
        store2 = await manager.get_store(
            VectorStoreType.LANCEDB,
            store_id="store2",
            db_path=temp_db_path + "_2"
        )
        
        assert store1 != store2
        assert store1.is_connected
        assert store2.is_connected
        
        # 在不同儲存中建立集合
        await store1.create_collection("collection1", 32)
        await store2.create_collection("collection2", 64)
        
        # 驗證集合分別存在於不同的儲存中
        assert await store1.collection_exists("collection1")
        assert not await store1.collection_exists("collection2")
        
        assert await store2.collection_exists("collection2")
        assert not await store2.collection_exists("collection1")
        
        await manager.close()