"""中文 GraphRAG 系統配置管理模組。

本模組提供完整的配置管理功能，包括：
- 系統配置模型定義
- YAML 配置解析
- 環境變數管理
- 配置驗證和預設值處理
"""

# 從 models 模組匯入所有配置類別
from .models import (
    # 枚舉類型
    LLMType,
    EmbeddingType, 
    VectorStoreType,
    DeviceType,
    
    # 配置模型
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
    GraphRAGConfig,
)

# 從 loader 模組匯入配置載入功能
from .loader import (
    ConfigurationError,
    ConfigLoader,
    load_config,
    create_default_config,
)

# 從 strategy 模組匯入模型選擇策略
from .strategy import (
    TaskType,
    ModelPerformanceMetrics,
    ModelSelectionStrategy,
    DefaultModelSelectionStrategy,
    CostOptimizedSelectionStrategy,
    AdaptiveSelectionStrategy,
    ModelSelector,
)

# 從 env 模組匯入環境變數管理功能
from .env import (
    EnvVarError,
    EnvVarConfig,
    EnvironmentManager,
    SYSTEM_ENV_VARS,
    env_manager,
    get_env_var,
    validate_system_env_vars,
)

# 從 validation 模組匯入配置驗證功能
from .validation import (
    ConfigValidationError,
    ConfigValidationWarning,
    ConfigValidator,
    DefaultConfigProvider,
    validate_config,
    apply_default_values,
)

# 匯出所有公開的類別和函數
__all__ = [
    # 枚舉類型
    "LLMType",
    "EmbeddingType",
    "VectorStoreType", 
    "DeviceType",
    "TaskType",
    
    # 配置模型
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
    "GraphRAGConfig",
    
    # 配置載入功能
    "ConfigurationError",
    "ConfigLoader",
    "load_config",
    "create_default_config",
    
    # 模型選擇策略
    "ModelPerformanceMetrics",
    "ModelSelectionStrategy",
    "DefaultModelSelectionStrategy",
    "CostOptimizedSelectionStrategy",
    "AdaptiveSelectionStrategy",
    "ModelSelector",
    
    # 環境變數管理
    "EnvVarError",
    "EnvVarConfig", 
    "EnvironmentManager",
    "SYSTEM_ENV_VARS",
    "env_manager",
    "get_env_var",
    "validate_system_env_vars",
    
    # 配置驗證和預設值
    "ConfigValidationError",
    "ConfigValidationWarning",
    "ConfigValidator", 
    "DefaultConfigProvider",
    "validate_config",
    "apply_default_values",
]