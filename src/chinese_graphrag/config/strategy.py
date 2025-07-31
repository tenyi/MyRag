"""
模型選擇和切換策略

實作智慧模型選擇、成本優化和品質控制機制
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .models import EmbeddingConfig, GraphRAGConfig, LLMConfig


logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """任務類型枚舉"""
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    COMMUNITY_DETECTION = "community_detection"
    COMMUNITY_REPORT = "community_report"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    TEXT_EMBEDDING = "text_embedding"
    QUERY_EMBEDDING = "query_embedding"


class ModelPerformanceMetrics:
    """模型效能指標"""
    
    def __init__(self):
        self.response_times: Dict[str, List[float]] = {}
        self.success_rates: Dict[str, List[bool]] = {}
        self.quality_scores: Dict[str, List[float]] = {}
        self.cost_per_request: Dict[str, List[float]] = {}

    def record_response_time(self, model_name: str, response_time: float):
        """記錄回應時間"""
        if model_name not in self.response_times:
            self.response_times[model_name] = []
        self.response_times[model_name].append(response_time)

    def record_success(self, model_name: str, success: bool):
        """記錄成功率"""
        if model_name not in self.success_rates:
            self.success_rates[model_name] = []
        self.success_rates[model_name].append(success)

    def record_quality_score(self, model_name: str, score: float):
        """記錄品質分數"""
        if model_name not in self.quality_scores:
            self.quality_scores[model_name] = []
        self.quality_scores[model_name].append(score)

    def record_cost(self, model_name: str, cost: float):
        """記錄成本"""
        if model_name not in self.cost_per_request:
            self.cost_per_request[model_name] = []
        self.cost_per_request[model_name].append(cost)

    def get_average_response_time(self, model_name: str) -> Optional[float]:
        """取得平均回應時間"""
        times = self.response_times.get(model_name, [])
        return sum(times) / len(times) if times else None

    def get_success_rate(self, model_name: str) -> Optional[float]:
        """取得成功率"""
        successes = self.success_rates.get(model_name, [])
        return sum(successes) / len(successes) if successes else None

    def get_average_quality_score(self, model_name: str) -> Optional[float]:
        """取得平均品質分數"""
        scores = self.quality_scores.get(model_name, [])
        return sum(scores) / len(scores) if scores else None

    def get_average_cost(self, model_name: str) -> Optional[float]:
        """取得平均成本"""
        costs = self.cost_per_request.get(model_name, [])
        return sum(costs) / len(costs) if costs else None


class ModelSelectionStrategy(ABC):
    """模型選擇策略抽象基類"""

    @abstractmethod
    def select_llm_model(
        self, 
        config: GraphRAGConfig, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> str:
        """
        選擇 LLM 模型
        
        Args:
            config: GraphRAG 配置
            task_type: 任務類型
            context: 額外上下文資訊
            
        Returns:
            str: 選中的模型名稱
        """
        pass

    @abstractmethod
    def select_embedding_model(
        self, 
        config: GraphRAGConfig, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> str:
        """
        選擇 Embedding 模型
        
        Args:
            config: GraphRAG 配置
            task_type: 任務類型
            context: 額外上下文資訊
            
        Returns:
            str: 選中的模型名稱
        """
        pass


class DefaultModelSelectionStrategy(ModelSelectionStrategy):
    """預設模型選擇策略"""

    def select_llm_model(
        self, 
        config: GraphRAGConfig, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> str:
        """選擇預設 LLM 模型"""
        return config.model_selection.default_llm

    def select_embedding_model(
        self, 
        config: GraphRAGConfig, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> str:
        """選擇預設 Embedding 模型"""
        return config.model_selection.default_embedding


class CostOptimizedSelectionStrategy(ModelSelectionStrategy):
    """成本優化選擇策略"""

    def __init__(self, performance_metrics: ModelPerformanceMetrics):
        self.performance_metrics = performance_metrics
        
        # 預定義的模型成本係數（相對值）
        self.llm_cost_coefficients = {
            "gpt-4": 1.0,
            "gpt-4-turbo": 0.5,
            "gpt-3.5-turbo": 0.1,
            "claude-3": 0.8,
            "local_model": 0.01
        }
        
        self.embedding_cost_coefficients = {
            "text-embedding-3-large": 1.0,
            "text-embedding-3-small": 0.5,
            "text-embedding-ada-002": 0.3,
            "bge-m3": 0.01,
            "local_embedding": 0.01
        }

    def select_llm_model(
        self, 
        config: GraphRAGConfig, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> str:
        """基於成本和品質選擇 LLM 模型"""
        if not config.model_selection.cost_optimization:
            return config.model_selection.default_llm

        # 取得所有可用的 LLM 模型
        llm_models = [
            name for name, model_config in config.models.items()
            if isinstance(model_config, LLMConfig)
        ]

        if not llm_models:
            return config.model_selection.default_llm

        # 根據任務類型和成本選擇模型
        best_model = self._select_best_model_by_cost_quality(
            llm_models, task_type, config.model_selection.quality_threshold
        )

        return best_model or config.model_selection.default_llm

    def select_embedding_model(
        self, 
        config: GraphRAGConfig, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> str:
        """基於成本和品質選擇 Embedding 模型"""
        if not config.model_selection.cost_optimization:
            return config.model_selection.default_embedding

        # 取得所有可用的 Embedding 模型
        embedding_models = [
            name for name, model_config in config.models.items()
            if isinstance(model_config, EmbeddingConfig)
        ]

        if not embedding_models:
            return config.model_selection.default_embedding

        # 對於中文任務，優先選擇中文優化模型
        if context and context.get('language') == 'zh':
            chinese_models = [
                name for name in embedding_models
                if 'chinese' in name.lower() or 'bge' in name.lower()
            ]
            if chinese_models:
                return chinese_models[0]

        # 根據成本選擇模型
        best_model = self._select_best_model_by_cost_quality(
            embedding_models, task_type, config.model_selection.quality_threshold
        )

        return best_model or config.model_selection.default_embedding

    def _select_best_model_by_cost_quality(
        self, 
        models: List[str], 
        task_type: TaskType,
        quality_threshold: float
    ) -> Optional[str]:
        """根據成本和品質選擇最佳模型"""
        model_scores = []

        for model_name in models:
            # 取得品質分數
            quality_score = self.performance_metrics.get_average_quality_score(model_name)
            if quality_score is None or quality_score < quality_threshold:
                continue

            # 取得成本
            avg_cost = self.performance_metrics.get_average_cost(model_name)
            if avg_cost is None:
                # 使用預定義的成本係數
                avg_cost = self._get_estimated_cost(model_name)

            # 計算成本效益分數（品質/成本）
            cost_efficiency = quality_score / max(avg_cost, 0.001)
            model_scores.append((model_name, cost_efficiency))

        if not model_scores:
            return None

        # 選擇成本效益最高的模型
        model_scores.sort(key=lambda x: x[1], reverse=True)
        return model_scores[0][0]

    def _get_estimated_cost(self, model_name: str) -> float:
        """取得預估成本"""
        # 檢查 LLM 成本
        for model_key, cost in self.llm_cost_coefficients.items():
            if model_key in model_name.lower():
                return cost

        # 檢查 Embedding 成本
        for model_key, cost in self.embedding_cost_coefficients.items():
            if model_key in model_name.lower():
                return cost

        # 預設成本
        return 0.5


class AdaptiveSelectionStrategy(ModelSelectionStrategy):
    """自適應選擇策略"""

    def __init__(self, performance_metrics: ModelPerformanceMetrics):
        self.performance_metrics = performance_metrics
        self.task_model_preferences = {
            TaskType.ENTITY_EXTRACTION: ["gpt-4", "claude-3"],
            TaskType.RELATIONSHIP_EXTRACTION: ["gpt-4", "gpt-4-turbo"],
            TaskType.COMMUNITY_DETECTION: ["gpt-4-turbo", "gpt-3.5-turbo"],
            TaskType.COMMUNITY_REPORT: ["gpt-4", "claude-3"],
            TaskType.SUMMARIZATION: ["gpt-4", "claude-3"],
            TaskType.QUESTION_ANSWERING: ["gpt-4", "gpt-4-turbo"],
            TaskType.TEXT_EMBEDDING: ["bge-m3", "text-embedding-3-small"],
            TaskType.QUERY_EMBEDDING: ["bge-m3", "text-embedding-3-small"]
        }

    def select_llm_model(
        self, 
        config: GraphRAGConfig, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> str:
        """基於任務類型和歷史效能選擇 LLM 模型"""
        # 取得任務偏好的模型
        preferred_models = self.task_model_preferences.get(task_type, [])
        
        # 取得所有可用的 LLM 模型
        available_models = [
            name for name, model_config in config.models.items()
            if isinstance(model_config, LLMConfig)
        ]

        # 找到可用且偏好的模型
        for preferred in preferred_models:
            for available in available_models:
                if preferred in available.lower():
                    # 檢查歷史效能
                    success_rate = self.performance_metrics.get_success_rate(available)
                    if success_rate is None or success_rate > 0.8:
                        return available

        return config.model_selection.default_llm

    def select_embedding_model(
        self, 
        config: GraphRAGConfig, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> str:
        """基於任務類型和歷史效能選擇 Embedding 模型"""
        # 對於中文內容，優先選擇中文優化模型
        if context and context.get('language') == 'zh':
            chinese_models = [
                name for name, model_config in config.models.items()
                if isinstance(model_config, EmbeddingConfig) and 
                ('chinese' in name.lower() or 'bge' in name.lower())
            ]
            if chinese_models:
                return chinese_models[0]

        # 取得任務偏好的模型
        preferred_models = self.task_model_preferences.get(task_type, [])
        
        # 取得所有可用的 Embedding 模型
        available_models = [
            name for name, model_config in config.models.items()
            if isinstance(model_config, EmbeddingConfig)
        ]

        # 找到可用且偏好的模型
        for preferred in preferred_models:
            for available in available_models:
                if preferred in available.lower():
                    return available

        return config.model_selection.default_embedding


class ModelSelector:
    """模型選擇器"""

    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.performance_metrics = ModelPerformanceMetrics()
        
        # 根據配置選擇策略
        if config.model_selection.cost_optimization:
            self.strategy = CostOptimizedSelectionStrategy(self.performance_metrics)
        else:
            self.strategy = AdaptiveSelectionStrategy(self.performance_metrics)

    def select_llm_model(
        self, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> Tuple[str, LLMConfig]:
        """
        選擇 LLM 模型
        
        Args:
            task_type: 任務類型
            context: 額外上下文
            
        Returns:
            Tuple[str, LLMConfig]: 模型名稱和配置
        """
        model_name = self.strategy.select_llm_model(self.config, task_type, context)
        model_config = self.config.get_llm_config(model_name)
        
        if model_config is None:
            # 使用備用模型
            fallback_name = self.config.model_selection.fallback_models.get(model_name)
            if fallback_name:
                model_config = self.config.get_llm_config(fallback_name)
                model_name = fallback_name
            
            if model_config is None:
                # 使用預設模型
                model_name = self.config.model_selection.default_llm
                model_config = self.config.get_default_llm_config()
        
        logger.info(f"為任務 {task_type} 選擇 LLM 模型: {model_name}")
        return model_name, model_config

    def select_embedding_model(
        self, 
        task_type: TaskType,
        context: Optional[Dict] = None
    ) -> Tuple[str, EmbeddingConfig]:
        """
        選擇 Embedding 模型
        
        Args:
            task_type: 任務類型
            context: 額外上下文
            
        Returns:
            Tuple[str, EmbeddingConfig]: 模型名稱和配置
        """
        model_name = self.strategy.select_embedding_model(self.config, task_type, context)
        model_config = self.config.get_embedding_config(model_name)
        
        if model_config is None:
            # 使用備用模型
            fallback_name = self.config.model_selection.fallback_models.get(model_name)
            if fallback_name:
                model_config = self.config.get_embedding_config(fallback_name)
                model_name = fallback_name
            
            if model_config is None:
                # 使用預設模型
                model_name = self.config.model_selection.default_embedding
                model_config = self.config.get_default_embedding_config()
        
        logger.info(f"為任務 {task_type} 選擇 Embedding 模型: {model_name}")
        return model_name, model_config

    def record_model_performance(
        self, 
        model_name: str, 
        response_time: float,
        success: bool,
        quality_score: Optional[float] = None,
        cost: Optional[float] = None
    ):
        """記錄模型效能"""
        self.performance_metrics.record_response_time(model_name, response_time)
        self.performance_metrics.record_success(model_name, success)
        
        if quality_score is not None:
            self.performance_metrics.record_quality_score(model_name, quality_score)
        
        if cost is not None:
            self.performance_metrics.record_cost(model_name, cost)

    def get_model_statistics(self, model_name: str) -> Dict:
        """取得模型統計資訊"""
        return {
            "average_response_time": self.performance_metrics.get_average_response_time(model_name),
            "success_rate": self.performance_metrics.get_success_rate(model_name),
            "average_quality_score": self.performance_metrics.get_average_quality_score(model_name),
            "average_cost": self.performance_metrics.get_average_cost(model_name)
        }