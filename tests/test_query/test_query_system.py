"""
查詢系統測試

測試中文 GraphRAG 查詢系統的各個元件，包括：
- LLM 管理和適配器
- 中文查詢處理器
- 全域和本地搜尋引擎
- 統一查詢介面
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chinese_graphrag.config import GraphRAGConfig, VectorStoreConfig, VectorStoreType
from chinese_graphrag.models import Community, Document, Entity, Relationship, TextUnit
from chinese_graphrag.query import (
    ChineseQueryProcessor,
    GlobalSearchEngine,
    LLMConfig,
    LLMManager,
    LLMProvider,
    LocalSearchEngine,
    QueryEngine,
    QueryEngineConfig,
    QueryIntent,
    QueryType,
    TaskType,
)


@pytest.fixture
def mock_vector_store():
    """模擬向量存儲管理器"""
    mock_store = Mock()
    mock_store.health_check = AsyncMock(return_value=True)
    return mock_store


@pytest.fixture
def mock_indexer():
    """模擬 GraphRAG 索引器"""
    mock_indexer = Mock()

    # 建立測試實體
    entities = {
        "e1": Entity(id="e1", name="張三博士", type="人物", description="台灣大學教授"),
        "e2": Entity(id="e2", name="台灣大學", type="組織", description="台灣頂尖學府"),
        "e3": Entity(
            id="e3", name="人工智慧", type="技術", description="機器學習相關技術"
        ),
    }

    # 建立測試關係
    relationships = {
        "r1": Relationship(
            id="r1",
            source_entity_id="e1",
            target_entity_id="e2",
            description="張三博士在台灣大學任教",
        ),
        "r2": Relationship(
            id="r2",
            source_entity_id="e1",
            target_entity_id="e3",
            description="張三博士研究人工智慧",
        ),
    }

    # 建立測試社群
    communities = {
        "c1": Community(
            id="c1",
            title="學術研究社群",
            level=1,
            entities=["e1", "e2", "e3"],
            relationships=["r1", "r2"],
            summary="這是一個關於學術研究的社群，包含研究人員和機構",
        )
    }

    # 建立測試文本單元
    text_units = {
        "t1": TextUnit(
            id="t1",
            text="張三博士是台灣大學的資深教授，專精於人工智慧研究。",
            document_id="doc1",
            chunk_index=0,
        )
    }

    mock_indexer.entities = entities
    mock_indexer.relationships = relationships
    mock_indexer.communities = communities
    mock_indexer.text_units = text_units

    return mock_indexer


@pytest.fixture
def llm_configs():
    """LLM 配置列表"""
    return [
        LLMConfig(
            provider=LLMProvider.MOCK,
            model="test_model",
            config={"mock_response": '{"test": "success"}'},
            task_types=[
                TaskType.GLOBAL_SEARCH,
                TaskType.LOCAL_SEARCH,
                TaskType.GENERAL_QA,
            ],
        )
    ]


@pytest.fixture
def query_engine_config(llm_configs):
    """查詢引擎配置"""
    return QueryEngineConfig(
        llm_configs=llm_configs, enable_caching=False  # 測試時禁用快取
    )


@pytest.fixture
def graphrag_config():
    """GraphRAG 配置"""
    return GraphRAGConfig(
        models={},
        vector_store=VectorStoreConfig(type=VectorStoreType.LANCEDB, uri="./test"),
    )


class TestLLMManager:
    """測試 LLM 管理器"""

    @pytest.fixture
    def llm_manager(self, llm_configs):
        """LLM 管理器 fixture"""
        return LLMManager(llm_configs)

    @pytest.mark.asyncio
    async def test_generate_with_mock_llm(self, llm_manager):
        """測試使用 Mock LLM 生成回應"""
        result = await llm_manager.generate("測試提示詞", TaskType.GENERAL_QA)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_health_check(self, llm_manager):
        """測試健康檢查"""
        health_status = await llm_manager.health_check_all()

        assert isinstance(health_status, dict)
        assert len(health_status) > 0
        # Mock LLM 應該始終健康
        assert any(health_status.values())

    def test_get_metrics(self, llm_manager):
        """測試獲取指標"""
        metrics = llm_manager.get_metrics()

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        for name, metric in metrics.items():
            assert hasattr(metric, "total_requests")
            assert hasattr(metric, "success_rate")

    def test_get_adapter_info(self, llm_manager):
        """測試獲取適配器資訊"""
        info = llm_manager.get_adapter_info()

        assert isinstance(info, dict)
        assert len(info) > 0

        for name, adapter_info in info.items():
            assert "provider" in adapter_info
            assert "model" in adapter_info
            assert "is_healthy" in adapter_info


class TestChineseQueryProcessor:
    """測試中文查詢處理器"""

    @pytest.fixture
    def processor(self):
        """查詢處理器 fixture"""
        return ChineseQueryProcessor()

    def test_process_entity_query(self, processor):
        """測試實體查詢分析"""
        query = "張三博士是誰？"
        analysis = processor.process_query(query)

        assert analysis.original_query == query
        assert analysis.query_type in [QueryType.ENTITY_SEARCH, QueryType.LOCAL_SEARCH]
        assert analysis.intent == QueryIntent.INFORMATION_SEEKING
        assert "張三博士" in analysis.entities or "張三" in analysis.entities
        assert analysis.confidence > 0

    def test_process_relationship_query(self, processor):
        """測試關係查詢分析"""
        query = "張三博士和台灣大學的關係是什麼？"
        analysis = processor.process_query(query)

        assert analysis.query_type in [
            QueryType.RELATION_SEARCH,
            QueryType.LOCAL_SEARCH,
        ]
        assert len(analysis.entities) >= 1  # 至少識別一個實體
        assert analysis.confidence > 0

    def test_process_global_query(self, processor):
        """測試全域查詢分析"""
        query = "請總結一下學術研究的整體狀況"
        analysis = processor.process_query(query)

        assert analysis.query_type in [QueryType.GLOBAL_SEARCH, QueryType.SUMMARY]
        assert analysis.intent in [
            QueryIntent.INFORMATION_SEEKING,
            QueryIntent.EXPLANATION,
        ]
        assert analysis.confidence > 0

    def test_extract_keywords(self, processor):
        """測試關鍵詞提取"""
        query = "人工智慧在台灣大學的研究現況如何？"
        analysis = processor.process_query(query)

        assert len(analysis.keywords) > 0
        # 應該包含一些重要關鍵詞
        keywords_text = " ".join(analysis.keywords)
        assert any(word in keywords_text for word in ["人工智慧", "台灣大學", "研究"])

    def test_suggest_query_enhancement(self, processor):
        """測試查詢增強建議"""
        query = "這個"  # 模糊查詢
        analysis = processor.process_query(query)
        suggestions = processor.suggest_query_enhancement(analysis)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # 信心度低的查詢應該有建議
        if analysis.confidence < 0.6:
            assert any("明確" in suggestion for suggestion in suggestions)


class TestGlobalSearchEngine:
    """測試全域搜尋引擎"""

    @pytest.fixture
    def global_search_engine(self, llm_configs, mock_vector_store):
        """全域搜尋引擎 fixture"""
        llm_manager = LLMManager(llm_configs)
        return GlobalSearchEngine(llm_manager, mock_vector_store)

    @pytest.mark.asyncio
    async def test_global_search(self, global_search_engine, mock_indexer):
        """測試全域搜尋"""
        from chinese_graphrag.query.processor import (
            QueryAnalysis,
            QueryIntent,
            QueryType,
        )

        # 建立測試查詢分析
        analysis = QueryAnalysis(
            original_query="學術研究的整體狀況如何？",
            normalized_query="學術研究的整體狀況如何？",
            query_type=QueryType.GLOBAL_SEARCH,
            intent=QueryIntent.INFORMATION_SEEKING,
            entities=["學術研究"],
            keywords=["學術", "研究", "狀況"],
            confidence=0.8,
            suggested_llm_task=TaskType.GLOBAL_SEARCH,
            preprocessing_notes=[],
        )

        communities = list(mock_indexer.communities.values())
        entities = list(mock_indexer.entities.values())
        relationships = list(mock_indexer.relationships.values())

        result = await global_search_engine.search(
            query="學術研究的整體狀況如何？",
            analysis=analysis,
            communities=communities,
            entities=entities,
            relationships=relationships,
        )

        assert result.query == "學術研究的整體狀況如何？"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.confidence > 0
        assert len(result.sources) > 0
        assert len(result.reasoning_path) > 0

    @pytest.mark.asyncio
    async def test_health_check(self, global_search_engine):
        """測試健康檢查"""
        health_status = await global_search_engine.health_check()

        assert isinstance(health_status, dict)
        assert "llm_manager" in health_status
        assert "vector_store" in health_status
        assert "strategies" in health_status

    def test_get_available_strategies(self, global_search_engine):
        """測試獲取可用策略"""
        strategies = global_search_engine.get_available_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert "community_based" in strategies


class TestLocalSearchEngine:
    """測試本地搜尋引擎"""

    @pytest.fixture
    def local_search_engine(self, llm_configs, mock_vector_store):
        """本地搜尋引擎 fixture"""
        llm_manager = LLMManager(llm_configs)
        return LocalSearchEngine(llm_manager, mock_vector_store)

    @pytest.mark.asyncio
    async def test_local_search(self, local_search_engine, mock_indexer):
        """測試本地搜尋"""
        from chinese_graphrag.query.processor import (
            QueryAnalysis,
            QueryIntent,
            QueryType,
        )

        # 建立測試查詢分析
        analysis = QueryAnalysis(
            original_query="張三博士是誰？",
            normalized_query="張三博士是誰？",
            query_type=QueryType.ENTITY_SEARCH,
            intent=QueryIntent.INFORMATION_SEEKING,
            entities=["張三博士"],
            keywords=["張三博士"],
            confidence=0.9,
            suggested_llm_task=TaskType.LOCAL_SEARCH,
            preprocessing_notes=[],
        )

        entities = list(mock_indexer.entities.values())
        relationships = list(mock_indexer.relationships.values())
        text_units = list(mock_indexer.text_units.values())

        result = await local_search_engine.search(
            query="張三博士是誰？",
            analysis=analysis,
            entities=entities,
            relationships=relationships,
            text_units=text_units,
        )

        assert result.query == "張三博士是誰？"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.confidence > 0
        assert len(result.target_entities) > 0


class TestQueryEngine:
    """測試統一查詢引擎"""

    @pytest.fixture
    def query_engine(
        self, query_engine_config, graphrag_config, mock_indexer, mock_vector_store
    ):
        """查詢引擎 fixture"""
        return QueryEngine(
            config=query_engine_config,
            graphrag_config=graphrag_config,
            indexer=mock_indexer,
            vector_store=mock_vector_store,
        )

    @pytest.mark.asyncio
    async def test_simple_query(self, query_engine):
        """測試簡單查詢"""
        result = await query_engine.query("張三博士是誰？")

        assert result.query == "張三博士是誰？"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.confidence >= 0
        assert result.search_type in ["global", "local", "hybrid"]

    @pytest.mark.asyncio
    async def test_global_search_query(self, query_engine):
        """測試全域搜尋查詢"""
        result = await query_engine.query(
            "請總結學術研究的整體情況", search_type="global"
        )

        assert result.search_type == "global"
        assert result.global_result is not None
        assert isinstance(result.answer, str)

    @pytest.mark.asyncio
    async def test_local_search_query(self, query_engine):
        """測試本地搜尋查詢"""
        result = await query_engine.query("張三博士的詳細資訊", search_type="local")

        assert result.search_type == "local"
        assert result.local_result is not None
        assert isinstance(result.answer, str)

    @pytest.mark.asyncio
    async def test_hybrid_search_query(self, query_engine):
        """測試混合搜尋查詢"""
        result = await query_engine.query(
            "張三博士和台灣大學的研究合作情況", search_type="hybrid"
        )

        assert result.search_type == "hybrid"
        assert isinstance(result.answer, str)

    @pytest.mark.asyncio
    async def test_batch_query(self, query_engine):
        """測試批次查詢"""
        queries = ["張三博士是誰？", "台灣大學有什麼特色？", "人工智慧的發展趨勢？"]

        results = await query_engine.batch_query(queries, max_concurrent=2)

        assert len(results) == len(queries)
        for i, result in enumerate(results):
            assert result.query == queries[i]
            assert isinstance(result.answer, str)

    def test_get_engine_status(self, query_engine):
        """測試獲取引擎狀態"""
        status = query_engine.get_engine_status()

        assert isinstance(status, dict)
        assert "config" in status
        assert "cache" in status
        assert "data_status" in status
        assert "llm_status" in status

        # 檢查資料狀態
        data_status = status["data_status"]
        assert data_status["entities_count"] > 0
        assert data_status["relationships_count"] > 0
        assert data_status["communities_count"] > 0

    @pytest.mark.asyncio
    async def test_health_check(self, query_engine):
        """測試健康檢查"""
        health_status = await query_engine.health_check()

        assert isinstance(health_status, dict)
        assert "query_processor" in health_status
        assert "llm_manager" in health_status
        assert "global_search" in health_status
        assert "local_search" in health_status
        assert "indexer" in health_status

        # 查詢處理器應該始終健康
        assert health_status["query_processor"] is True

    def test_clear_cache(self, query_engine):
        """測試清空快取"""
        # 這個測試主要驗證方法不會拋出異常
        query_engine.clear_cache()

        cache_stats = query_engine.query_cache.get_cache_stats()
        assert cache_stats["total_entries"] == 0


class TestIntegration:
    """整合測試"""

    @pytest.mark.asyncio
    async def test_end_to_end_query_flow(
        self, query_engine_config, graphrag_config, mock_indexer, mock_vector_store
    ):
        """測試端到端查詢流程"""

        # 建立查詢引擎
        query_engine = QueryEngine(
            config=query_engine_config,
            graphrag_config=graphrag_config,
            indexer=mock_indexer,
            vector_store=mock_vector_store,
        )

        # 執行一系列查詢
        test_queries = [
            ("張三博士是誰？", "local"),
            ("學術研究的整體狀況？", "global"),
            ("張三博士與台灣大學的關係？", "hybrid"),
        ]

        for query, expected_type in test_queries:
            result = await query_engine.query(query)

            # 基本驗證
            assert result.query == query
            assert isinstance(result.answer, str)
            assert len(result.answer) > 0
            assert result.confidence >= 0
            assert result.search_time >= 0

            # 檢查分析結果
            assert result.analysis is not None
            assert result.analysis.original_query == query

            from loguru import logger

            logger.info(f"查詢: {query}")
            logger.info(f"搜尋類型: {result.search_type}")
            logger.info(f"信心度: {result.confidence:.2f}")
            logger.info(f"回答: {result.answer[:100]}...")

    def test_query_result_serialization(
        self, query_engine_config, graphrag_config, mock_indexer, mock_vector_store
    ):
        """測試查詢結果序列化"""
        from chinese_graphrag.query.engine import UnifiedQueryResult
        from chinese_graphrag.query.processor import (
            QueryAnalysis,
            QueryIntent,
            QueryType,
        )

        # 建立測試結果
        analysis = QueryAnalysis(
            original_query="測試查詢",
            normalized_query="測試查詢",
            query_type=QueryType.ENTITY_SEARCH,
            intent=QueryIntent.INFORMATION_SEEKING,
            entities=["測試實體"],
            keywords=["測試"],
            confidence=0.8,
            suggested_llm_task=TaskType.LOCAL_SEARCH,
            preprocessing_notes=[],
        )

        result = UnifiedQueryResult(
            query="測試查詢",
            answer="測試回答",
            confidence=0.8,
            search_type="local",
            analysis=analysis,
            sources=["測試來源"],
            reasoning_path=["測試推理"],
            search_time=1.0,
            llm_model_used="test_model",
        )

        # 測試轉換為字典
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["query"] == "測試查詢"
        assert result_dict["answer"] == "測試回答"
        assert result_dict["confidence"] == 0.8
        assert "analysis" in result_dict

        # 測試轉換為 QueryResult 模型
        query_result = result.to_query_result()
        assert query_result.query == "測試查詢"
        assert query_result.answer == "測試回答"
        assert isinstance(query_result.metadata, dict)
