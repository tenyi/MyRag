"""
GraphRAG 索引引擎測試

測試索引引擎的核心功能
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.chinese_graphrag.config import GraphRAGConfig, VectorStoreConfig, VectorStoreType
from src.chinese_graphrag.indexing import GraphRAGIndexer
from src.chinese_graphrag.models import Document


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
        # assert indexer.model_selector is not None  # 已移除
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
    async def test_entity_and_relationship_extraction(self, indexer, sample_documents, mocker):
        """測試實體和關係提取的整合流程"""
        import uuid
        import json
        from chinese_graphrag.models import Entity, Relationship

        # 模擬 LLM 的回應
        mock_llm_output = {
            "entities": [
                {"name": "張三博士", "type": "人物", "description": "在台灣大學進行研究的博士"},
                {"name": "台灣大學", "type": "組織", "description": "台灣的一所頂尖大學"}
            ],
            "relationships": [
                {"source": "張三博士", "target": "台灣大學", "description": "張三博士在台灣大學工作"}
            ]
        }
        
        # 模擬 LLM 實例
        mock_llm = mocker.AsyncMock()
        mock_llm.async_generate.return_value = json.dumps(mock_llm_output)
        
        # 模擬 create_llm 函數
        mocker.patch(
            "chinese_graphrag.llm.create_llm",
            return_value=mock_llm
        )
        
        # 建立文本單元
        text_units = await indexer._create_text_units(sample_documents)
        
        # 執行實體和關係提取
        entities, relationships = await indexer._extract_entities_and_relationships(text_units)
        
        # 驗證結果
        assert len(entities) > 0
        assert len(relationships) > 0
        
        # 檢查實體和關係
        entity_names = [e.name for e in entities]
        assert "張三博士" in entity_names
        assert "台灣大學" in entity_names

    @pytest.mark.asyncio
    async def test_detect_communities(self, indexer, mocker):
        """測試社群檢測"""
        import uuid
        from chinese_graphrag.models import Entity, Relationship, Community

        # 準備模擬的實體和關係資料
        entities = [
            Entity(id="e1", name="張三", type="人物"),
            Entity(id="e2", name="台灣大學", type="組織"),
            Entity(id="e3", name="李四", type="人物"),
            Entity(id="e4", name="清華大學", type="組織"),
            Entity(id="e5", name="人工智慧", type="技術"),
        ]
        relationships = [
            Relationship(id="r1", source_entity_id="e1", target_entity_id="e2", description="畢業於"),
            Relationship(id="r2", source_entity_id="e3", target_entity_id="e4", description="畢業於"),
            Relationship(id="r3", source_entity_id="e1", target_entity_id="e5", description="研究領域"),
            Relationship(id="r4", source_entity_id="e3", target_entity_id="e5", description="研究領域"),
        ]
        
        # 模擬社群檢測器的 detect_communities 方法
        mock_communities = [
            Community(
                id="c1",
                title="學術社群",
                level=1,
                entities=["e1", "e2", "e5"],
                relationships=["r1", "r3"],
                summary="關於學術研究的社群"
            )
        ]
        
        mocker.patch.object(
            indexer.community_detector,
            "detect_communities",
            return_value=mock_communities
        )
        
        # 檢測社群
        communities = await indexer._detect_communities(entities, relationships)
        
        # 檢查結果
        assert isinstance(communities, list)
        assert len(communities) > 0
        
        # 檢查社群屬性
        for community in communities:
            assert community.id is not None
            assert community.title is not None
            assert community.level >= 0
            assert len(community.entities) > 0
            assert community.summary is not None

    @pytest.mark.asyncio
    async def test_create_embeddings(self, indexer, sample_documents, mocker):
        """測試向量嵌入建立"""
        import numpy as np
        import json
        
        # 準備模擬資料
        from chinese_graphrag.models import TextUnit, Entity, Community
        
        text_units = [
            TextUnit(
                id=f"{doc.id}_chunk_0",
                text=doc.content[:50],
                document_id=doc.id,
                chunk_index=0,
                metadata={"document_title": doc.title}
            )
            for doc in sample_documents
        ]
        
        entities = [
            Entity(id="e1", name="張三博士", type="人物", text_units=[text_units[0].id]),
            Entity(id="e2", name="台灣大學", type="組織", text_units=[text_units[0].id])
        ]
        
        communities = [
            Community(
                id="c1",
                title="學術社群",
                level=1,
                entities=["e1", "e2"],
                relationships=["r1"],
                summary="關於學術研究的社群"
            )
        ]
        
        # 模擬 embed_texts 方法
        mock_text_embeddings = np.random.rand(len(text_units), 768)  # 文本單元嵌入
        mock_entity_embeddings = np.random.rand(len(entities), 768)  # 實體嵌入
        mock_community_embeddings = np.random.rand(len(communities), 768)  # 社群嵌入
        
        # 建立 AsyncMock 來模擬 embed_texts 方法
        mock_embed_texts = mocker.AsyncMock()
        mock_embed_texts.side_effect = [mock_text_embeddings, mock_entity_embeddings, mock_community_embeddings]
        
        # model_selector 已被移除，不需要模擬
        
        # 替換 EmbeddingManager 的 embed_texts 方法
        mocker.patch.object(
            indexer.embedding_manager,
            "embed_texts",
            mock_embed_texts
        )
        
        # 模擬 vector_store_manager 的方法
        mock_store_text_unit = mocker.AsyncMock()
        mock_store_entity = mocker.AsyncMock()
        mock_store_community = mocker.AsyncMock()
        mocker.patch.object(indexer.vector_store_manager, "store_text_unit", mock_store_text_unit)
        mocker.patch.object(indexer.vector_store_manager, "store_entity", mock_store_entity)
        mocker.patch.object(indexer.vector_store_manager, "store_community", mock_store_community)
        
        # 建立嵌入
        await indexer._create_embeddings(text_units, entities, communities)
        
        # 驗證 embed_texts 被正確呼叫
        assert mock_embed_texts.call_count == 3
        
        # 驗證文本單元和實體的嵌入已設置
        for i, unit in enumerate(text_units):
            assert unit.embedding is not None
            np.testing.assert_array_equal(unit.embedding, mock_text_embeddings[i])
            
        for i, entity in enumerate(entities):
            assert entity.embedding is not None
            np.testing.assert_array_equal(entity.embedding, mock_entity_embeddings[i])
            
        for i, community in enumerate(communities):
            assert community.embedding is not None
            np.testing.assert_array_equal(community.embedding, mock_community_embeddings[i])
            
        # 驗證向量存儲方法被呼叫
        assert mock_store_text_unit.call_count == len(text_units)
        assert mock_store_entity.call_count == len(entities)
        assert mock_store_community.call_count == len(communities)

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
    async def test_full_indexing_workflow(self, indexer, sample_documents, mocker):
        """測試完整的索引工作流程"""
        import numpy as np
        from chinese_graphrag.models import TextUnit, Entity, Relationship, Community
        
        # 模擬 GraphRAG 可用性檢查，確保走自定義流程
        mocker.patch.object(indexer, '_check_graphrag_availability', return_value=False)
        
        # 模擬文件處理
        mock_process_documents = mocker.patch.object(
            indexer, 
            "_process_documents", 
            return_value=sample_documents
        )
        
        # 模擬文本單元建立
        text_units = [
            TextUnit(
                id=f"{doc.id}_chunk_0",
                text=doc.content[:50],
                document_id=doc.id,
                chunk_index=0,
                metadata={"document_title": doc.title}
            )
            for doc in sample_documents
        ]
        mock_create_text_units = mocker.patch.object(
            indexer, 
            "_create_text_units", 
            return_value=text_units
        )
        
        # 模擬實體和關係提取
        entities = [
            Entity(id="e1", name="張三博士", type="人物", text_units=[text_units[0].id]),
            Entity(id="e2", name="台灣大學", type="組織", text_units=[text_units[0].id])
        ]
        relationships = [
            Relationship(
                id="r1", 
                source_entity_id="e1", 
                target_entity_id="e2", 
                relationship_type="工作於",
                description="張三博士在台灣大學工作",
                text_units=[text_units[0].id]
            )
        ]
        mock_extract = mocker.patch.object(
            indexer, 
            "_extract_entities_and_relationships", 
            return_value=(entities, relationships)
        )
        
        # 模擬社群檢測
        communities = [
            Community(
                id="c1",
                title="學術社群",
                level=1,
                entities=["e1", "e2"],
                relationships=["r1"],
                summary="關於學術研究的社群"
            )
        ]
        mock_detect = mocker.patch.object(
            indexer, 
            "_detect_communities", 
            return_value=communities
        )
        
        # 模擬嵌入建立
        mock_create_embeddings = mocker.patch.object(
            indexer, 
            "_create_embeddings", 
            return_value=None
        )
        
        # 模擬結果儲存
        mock_save_results = mocker.patch.object(
            indexer, 
            "_save_results", 
            return_value=None
        )
        
        # 執行索引流程
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input"
            output_path = Path(temp_dir) / "output"
            input_path.mkdir()
            
            # 執行索引
            stats = await indexer.index_documents(input_path, output_path)
            
            # 驗證各個方法被正確呼叫
            mock_process_documents.assert_called_once_with(input_path)
            mock_create_text_units.assert_called_once_with(sample_documents)
            mock_extract.assert_called_once_with(text_units)
            mock_detect.assert_called_once_with(entities, relationships)
            mock_create_embeddings.assert_called_once()
            mock_save_results.assert_called_once_with(output_path)
            
            # 驗證統計結果
            assert isinstance(stats, dict)
            assert stats["documents"] == len(sample_documents)
            assert stats["text_units"] == len(text_units)
            assert stats["entities"] == len(entities)
            assert stats["relationships"] == len(relationships)
            assert stats["communities"] == len(communities)
            
            # 驗證索引狀態已更新
            for doc in sample_documents:
                assert doc.id in indexer.indexed_documents
            for unit in text_units:
                assert unit.id in indexer.text_units
            for entity in entities:
                assert entity.id in indexer.entities
            for rel in relationships:
                assert rel.id in indexer.relationships
            for comm in communities:
                assert comm.id in indexer.communities
