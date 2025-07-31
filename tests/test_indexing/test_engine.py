"""
GraphRAG 索引引擎測試

測試索引引擎的核心功能
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from chinese_graphrag.config import GraphRAGConfig, VectorStoreConfig, VectorStoreType
from chinese_graphrag.indexing import GraphRAGIndexer
from chinese_graphrag.models import Document


class TestGraphRAGIndexer:
    """GraphRAG 索引引擎測試"""

    @pytest.fixture
    def config(self):
        """建立測試配置"""
        return GraphRAGConfig(
            models={},
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LANCEDB,
                uri="./test_data/lancedb"
            )
        )

    @pytest.fixture
    def indexer(self, config):
        """建立索引引擎實例"""
        return GraphRAGIndexer(config)

    @pytest.fixture
    def sample_documents(self):
        """建立範例文件"""
        return [
            Document(
                id="doc1",
                title="測試文件1",
                content="這是一個關於人工智慧的文件。張三博士在台灣大學進行研究。",
                file_path=Path("test1.txt"),
                metadata={"language": "zh"}
            ),
            Document(
                id="doc2",
                title="測試文件2",
                content="李四教授在清華大學開發了新的機器學習演算法。這個演算法在自然語言處理方面有重要應用。",
                file_path=Path("test2.txt"),
                metadata={"language": "zh"}
            )
        ]

    def test_indexer_initialization(self, config):
        """測試索引引擎初始化"""
        indexer = GraphRAGIndexer(config)
        
        assert indexer.config == config
        assert indexer.model_selector is not None
        assert indexer.document_processor is not None
        assert indexer.embedding_manager is not None
        assert indexer.vector_store_manager is not None
        
        # 檢查初始狀態
        assert len(indexer.indexed_documents) == 0
        assert len(indexer.text_units) == 0
        assert len(indexer.entities) == 0
        assert len(indexer.relationships) == 0
        assert len(indexer.communities) == 0

    @pytest.mark.asyncio
    async def test_create_text_units(self, indexer, sample_documents):
        """測試文本單元建立"""
        text_units = await indexer._create_text_units(sample_documents)
        
        assert len(text_units) > 0
        
        # 檢查文本單元屬性
        for unit in text_units:
            assert unit.id is not None
            assert unit.text is not None
            assert unit.document_id in [doc.id for doc in sample_documents]
            assert unit.chunk_index >= 0
            assert "language" in unit.metadata

    @pytest.mark.asyncio
    async def test_extract_entities_and_relationships(self, indexer, sample_documents):
        """測試實體和關係提取"""
        # 先建立文本單元
        text_units = await indexer._create_text_units(sample_documents)
        
        # 提取實體和關係
        entities, relationships = await indexer._extract_entities_and_relationships(text_units)
        
        # 檢查結果
        assert isinstance(entities, list)
        assert isinstance(relationships, list)
        
        # 如果有實體，檢查其屬性
        for entity in entities:
            assert entity.id is not None
            assert entity.name is not None
            assert entity.type is not None
            assert entity.description is not None
            assert len(entity.text_units) > 0

    @pytest.mark.asyncio
    async def test_detect_communities(self, indexer, sample_documents):
        """測試社群檢測"""
        # 先建立文本單元和實體
        text_units = await indexer._create_text_units(sample_documents)
        entities, relationships = await indexer._extract_entities_and_relationships(text_units)
        
        # 檢測社群
        communities = await indexer._detect_communities(entities, relationships)
        
        # 檢查結果
        assert isinstance(communities, list)
        
        # 如果有社群，檢查其屬性
        for community in communities:
            assert community.id is not None
            assert community.title is not None
            assert community.level >= 0
            assert len(community.entities) > 0
            assert community.summary is not None

    @pytest.mark.asyncio
    async def test_create_embeddings(self, indexer, sample_documents):
        """測試向量嵌入建立"""
        # 準備資料
        text_units = await indexer._create_text_units(sample_documents)
        entities, relationships = await indexer._extract_entities_and_relationships(text_units)
        communities = await indexer._detect_communities(entities, relationships)
        
        # 建立嵌入
        await indexer._create_embeddings(text_units, entities, communities)
        
        # 檢查嵌入是否建立
        for unit in text_units:
            if unit.embedding is not None:
                assert unit.embedding.shape[0] > 0  # 檢查向量維度

    def test_get_statistics(self, indexer):
        """測試統計資訊取得"""
        stats = indexer.get_statistics()
        
        expected_keys = ["documents", "text_units", "entities", "relationships", "communities"]
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], int)
            assert stats[key] >= 0

    def test_clear_index(self, indexer):
        """測試清除索引"""
        # 添加一些測試資料
        indexer.indexed_documents["test"] = "dummy"
        indexer.text_units["test"] = "dummy"
        indexer.entities["test"] = "dummy"
        indexer.relationships["test"] = "dummy"
        indexer.communities["test"] = "dummy"
        
        # 清除索引
        indexer.clear_index()
        
        # 檢查是否清除
        assert len(indexer.indexed_documents) == 0
        assert len(indexer.text_units) == 0
        assert len(indexer.entities) == 0
        assert len(indexer.relationships) == 0
        assert len(indexer.communities) == 0

    @pytest.mark.asyncio
    async def test_full_indexing_workflow(self, indexer):
        """測試完整的索引工作流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 建立測試文件
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            
            test_file = input_dir / "test.txt"
            test_file.write_text("這是一個測試文件。張三在台灣大學工作。", encoding="utf-8")
            
            output_dir = Path(temp_dir) / "output"
            
            # 執行索引
            stats = await indexer.index_documents(input_dir, output_dir)
            
            # 檢查統計結果
            assert isinstance(stats, dict)
            assert "documents" in stats
            assert "text_units" in stats
            assert "entities" in stats
            assert "relationships" in stats
            assert "communities" in stats
            
            # 檢查輸出檔案
            if output_dir.exists():
                expected_files = [
                    "documents.json",
                    "entities.json", 
                    "relationships.json",
                    "communities.json",
                    "index_stats.json"
                ]
                
                for filename in expected_files:
                    file_path = output_dir / filename
                    if file_path.exists():
                        assert file_path.stat().st_size > 0