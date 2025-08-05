"""
模型選擇策略測試

測試各種模型選擇和切換策略
"""

import pytest

from chinese_graphrag.config.models import (
    EmbeddingConfig,
    EmbeddingType,
    GraphRAGConfig,
    LLMConfig,
    LLMType,
    ModelSelectionConfig,
    VectorStoreConfig,
    VectorStoreType,
)
from chinese_graphrag.config.strategy import (
    AdaptiveSelectionStrategy,
    CostOptimizedSelectionStrategy,
    DefaultModelSelectionStrategy,
    ModelPerformanceMetrics,
    ModelSelector,
    TaskType,
)


class TestModelPerformanceMetrics:
    """模型效能指標測試"""

    def test_record_and_retrieve_metrics(self):
        """測試記錄和檢索指標"""
        metrics = ModelPerformanceMetrics()

        # 記錄指標
        metrics.record_response_time("model1", 1.5)
        metrics.record_response_time("model1", 2.0)
        metrics.record_success("model1", True)
        metrics.record_success("model1", False)
        metrics.record_quality_score("model1", 0.8)
        metrics.record_cost("model1", 0.01)

        # 檢索指標
        assert metrics.get_average_response_time("model1") == 1.75
        assert metrics.get_success_rate("model1") == 0.5
        assert metrics.get_average_quality_score("model1") == 0.8
        assert metrics.get_average_cost("model1") == 0.01

        # 測試不存在的模型
        assert metrics.get_average_response_time("nonexistent") is None


class TestDefaultModelSelectionStrategy:
    """預設模型選擇策略測試"""

    def test_select_default_models(self):
        """測試選擇預設模型"""
        config = self._create_test_config()
        strategy = DefaultModelSelectionStrategy()

        llm_model = strategy.select_llm_model(config, TaskType.ENTITY_EXTRACTION)
        embedding_model = strategy.select_embedding_model(
            config, TaskType.TEXT_EMBEDDING
        )

        assert llm_model == "default_chat_model"
        assert embedding_model == "ollama_embedding_model"

    def _create_test_config(self) -> GraphRAGConfig:
        """建立測試配置"""
        return GraphRAGConfig(
            models={
                "default_chat_model": LLMConfig(
                    type=LLMType.OPENAI_CHAT, model="gpt-4"
                ),
                "ollama_embedding_model": EmbeddingConfig(
                    type=EmbeddingType.BGE_M3, model="BAAI/bge-m3"
                ),
            },
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LANCEDB, uri="./data/lancedb"
            ),
            model_selection=ModelSelectionConfig(
                default_llm="default_chat_model",
                default_embedding="ollama_embedding_model",
            ),
        )


class TestCostOptimizedSelectionStrategy:
    """成本優化選擇策略測試"""

    def test_cost_optimization_disabled(self):
        """測試成本優化關閉時的行為"""
        config = self._create_test_config()
        config.model_selection.cost_optimization = False

        metrics = ModelPerformanceMetrics()
        strategy = CostOptimizedSelectionStrategy(metrics)

        llm_model = strategy.select_llm_model(config, TaskType.ENTITY_EXTRACTION)
        assert llm_model == "default_chat_model"

    def test_cost_optimization_enabled(self):
        """測試成本優化啟用時的行為"""
        config = self._create_test_config()
        config.model_selection.cost_optimization = True

        metrics = ModelPerformanceMetrics()
        # 記錄一些效能指標
        metrics.record_quality_score("default_chat_model", 0.9)
        metrics.record_cost("default_chat_model", 0.1)

        strategy = CostOptimizedSelectionStrategy(metrics)

        llm_model = strategy.select_llm_model(config, TaskType.ENTITY_EXTRACTION)
        assert llm_model == "default_chat_model"

    def test_chinese_embedding_preference(self):
        """測試中文 Embedding 模型偏好"""
        config = self._create_test_config()
        config.model_selection.cost_optimization = True

        metrics = ModelPerformanceMetrics()
        strategy = CostOptimizedSelectionStrategy(metrics)

        # 測試中文上下文
        context = {"language": "zh"}
        embedding_model = strategy.select_embedding_model(
            config, TaskType.TEXT_EMBEDDING, context
        )
        assert embedding_model == "ollama_embedding_model"

    def _create_test_config(self) -> GraphRAGConfig:
        """建立測試配置"""
        return GraphRAGConfig(
            models={
                "default_chat_model": LLMConfig(
                    type=LLMType.OPENAI_CHAT, model="gpt-4"
                ),
                "ollama_embedding_model": EmbeddingConfig(
                    type=EmbeddingType.BGE_M3, model="BAAI/bge-m3"
                ),
                "openai_embedding_model": EmbeddingConfig(
                    type=EmbeddingType.OPENAI_EMBEDDING, model="text-embedding-3-small"
                ),
            },
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LANCEDB, uri="./data/lancedb"
            ),
            model_selection=ModelSelectionConfig(
                default_llm="default_chat_model",
                default_embedding="ollama_embedding_model",
            ),
        )


class TestAdaptiveSelectionStrategy:
    """自適應選擇策略測試"""

    def test_task_based_model_selection(self):
        """測試基於任務的模型選擇"""
        config = self._create_test_config()
        metrics = ModelPerformanceMetrics()
        strategy = AdaptiveSelectionStrategy(metrics)

        # 測試不同任務類型
        llm_model = strategy.select_llm_model(config, TaskType.ENTITY_EXTRACTION)
        assert llm_model in ["gpt4_model", "default_chat_model"]

        embedding_model = strategy.select_embedding_model(
            config, TaskType.TEXT_EMBEDDING
        )
        assert embedding_model == "ollama_embedding_model"

    def test_chinese_context_preference(self):
        """測試中文上下文偏好"""
        config = self._create_test_config()
        metrics = ModelPerformanceMetrics()
        strategy = AdaptiveSelectionStrategy(metrics)

        context = {"language": "zh"}
        embedding_model = strategy.select_embedding_model(
            config, TaskType.TEXT_EMBEDDING, context
        )
        assert embedding_model == "ollama_embedding_model"

    def _create_test_config(self) -> GraphRAGConfig:
        """建立測試配置"""
        return GraphRAGConfig(
            models={
                "default_chat_model": LLMConfig(
                    type=LLMType.OPENAI_CHAT, model="gpt-3.5-turbo"
                ),
                "gpt4_model": LLMConfig(type=LLMType.OPENAI_CHAT, model="gpt-4"),
                "ollama_embedding_model": EmbeddingConfig(
                    type=EmbeddingType.BGE_M3, model="BAAI/bge-m3"
                ),
                "openai_embedding_model": EmbeddingConfig(
                    type=EmbeddingType.OPENAI_EMBEDDING, model="text-embedding-3-small"
                ),
            },
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LANCEDB, uri="./data/lancedb"
            ),
            model_selection=ModelSelectionConfig(
                default_llm="default_chat_model",
                default_embedding="ollama_embedding_model",
            ),
        )


class TestModelSelector:
    """模型選擇器測試"""

    def test_select_llm_model(self):
        """測試選擇 LLM 模型"""
        config = self._create_test_config()
        selector = ModelSelector(config)

        model_name, model_config = selector.select_llm_model(TaskType.ENTITY_EXTRACTION)

        assert model_name in config.models
        assert isinstance(model_config, LLMConfig)

    def test_select_embedding_model(self):
        """測試選擇 Embedding 模型"""
        config = self._create_test_config()
        selector = ModelSelector(config)

        model_name, model_config = selector.select_embedding_model(
            TaskType.TEXT_EMBEDDING
        )

        assert model_name in config.models
        assert isinstance(model_config, EmbeddingConfig)

    def test_fallback_model_selection(self):
        """測試備用模型選擇"""
        config = self._create_test_config()
        # 設定備用模型
        config.model_selection.fallback_models = {
            "nonexistent_model": "default_chat_model"
        }

        selector = ModelSelector(config)

        # 模擬選擇不存在的模型
        selector.strategy = MockStrategy("nonexistent_model", "ollama_embedding_model")

        model_name, model_config = selector.select_llm_model(TaskType.ENTITY_EXTRACTION)

        # 應該使用備用模型
        assert model_name == "default_chat_model"
        assert isinstance(model_config, LLMConfig)

    def test_record_model_performance(self):
        """測試記錄模型效能"""
        config = self._create_test_config()
        selector = ModelSelector(config)

        selector.record_model_performance(
            "test_model", response_time=1.5, success=True, quality_score=0.8, cost=0.01
        )

        stats = selector.get_model_statistics("test_model")
        assert stats["average_response_time"] == 1.5
        assert stats["success_rate"] == 1.0
        assert stats["average_quality_score"] == 0.8
        assert stats["average_cost"] == 0.01

    def _create_test_config(self) -> GraphRAGConfig:
        """建立測試配置"""
        return GraphRAGConfig(
            models={
                "default_chat_model": LLMConfig(
                    type=LLMType.OPENAI_CHAT, model="gpt-4"
                ),
                "ollama_embedding_model": EmbeddingConfig(
                    type=EmbeddingType.BGE_M3, model="BAAI/bge-m3"
                ),
            },
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LANCEDB, uri="./data/lancedb"
            ),
            model_selection=ModelSelectionConfig(
                default_llm="default_chat_model",
                default_embedding="ollama_embedding_model",
            ),
        )


class MockStrategy:
    """模擬策略類別，用於測試"""

    def __init__(self, llm_model: str, embedding_model: str):
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    def select_llm_model(self, config, task_type, context=None):
        return self.llm_model

    def select_embedding_model(self, config, task_type, context=None):
        return self.embedding_model
