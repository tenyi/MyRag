"""
配置管理模組

提供 GraphRAG 系統的完整配置管理功能，包括：
- 配置資料模型定義
- YAML 配置檔案載入和驗證
- 模型選擇和切換策略
- 環境變數處理
"""

from .loader import ConfigLoader, load_config, create_default_config
from .models import (
    GraphRAGConfig,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    ChineseProcessingConfig,
    InputConfig,
    ChunkConfig,
    IndexingConfig,
    QueryConfig,
    StorageConfig,
    ParallelizationConfig,
    ModelSelectionConfig,
    LLMType,
    EmbeddingType,
    VectorStoreType,
    DeviceType,
)
from .strategy import (
    ModelSelector,
    ModelSelectionStrategy,
    DefaultModelSelectionStrategy,
    CostOptimizedSelectionStrategy,
    AdaptiveSelectionStrategy,
    TaskType,
    ModelPerformanceMetrics,
)

__all__ = [
    # 載入器
    "ConfigLoader",
    "load_config",
    "create_default_config",
    
    # 配置模型
    "GraphRAGConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "ChineseProcessingConfig",
    "InputConfig",
    "ChunkConfig",
    "IndexingConfig",
    "QueryConfig",
    "StorageConfig",
    "ParallelizationConfig",
    "ModelSelectionConfig",
    
    # 枚舉類型
    "LLMType",
    "EmbeddingType",
    "VectorStoreType",
    "DeviceType",
    
    # 策略相關
    "ModelSelector",
    "ModelSelectionStrategy",
    "DefaultModelSelectionStrategy",
    "CostOptimizedSelectionStrategy",
    "AdaptiveSelectionStrategy",
    "TaskType",
    "ModelPerformanceMetrics",
]