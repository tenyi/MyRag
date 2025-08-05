"""
Embedding 服務模組

提供多種 embedding 模型的統一介面和管理功能
支援 BGE-M3、OpenAI、text2vec、m3e 等模型
包含快取、GPU 加速、記憶體優化和使用量監控功能
"""

from .base import (
    EmbeddingComputeError,
    EmbeddingModelType,
    EmbeddingResult,
    EmbeddingService,
    EmbeddingServiceError,
    ModelLoadError,
    ModelMetrics,
)

# 效能優化和管理
from .cache import (
    CacheEntry,
    CacheStrategy,
    DiskCache,
    EmbeddingCache,
    LFUStrategy,
    LRUStrategy,
    MemoryCache,
    MultiLevelCache,
    create_embedding_cache,
)
from .chinese_optimized import (
    ChineseEmbeddingConfig,
    ChineseOptimizedEmbeddingService,
    create_chinese_optimized_service,
)
from .evaluation import (
    BenchmarkConfig,
    ChineseEmbeddingEvaluator,
    EmbeddingEvaluator,
    PerformanceMetrics,
    chinese_quality_benchmark,
    quick_benchmark,
)
from .local_models import (
    LOCAL_MODELS,
    LocalEmbeddingService,
    create_local_service,
    create_m3e_service,
    create_text2vec_service,
)
from .manager import EmbeddingManager

# from .bge_m3 import (
#     BGEM3EmbeddingService,
#     create_bge_m3_service
# )

# from .openai_service import (
#     OpenAIEmbeddingService,
#     create_openai_service,
#     OPENAI_MODELS
# )


# from .gpu_acceleration import (
#     DeviceManager,
#     MemoryOptimizer,
#     BatchProcessor,
#     GPUInfo,
#     MemoryStats,
#     get_device_manager,
#     get_memory_optimizer,
#     create_batch_processor
# )

# from .monitoring import (
#     UsageMonitor,
#     UsageRecord,
#     ModelStats,
#     Alert,
#     AlertLevel,
#     get_usage_monitor,
#     record_embedding_usage
# )

__all__ = [
    # 基礎類別和介面
    "EmbeddingService",
    "EmbeddingResult",
    "EmbeddingModelType",
    "ModelMetrics",
    # 異常類別
    "EmbeddingServiceError",
    "ModelLoadError",
    "EmbeddingComputeError",
    # 管理器
    "EmbeddingManager",
    # BGE-M3 服務（暫時禁用）
    # "BGEM3EmbeddingService",
    # "create_bge_m3_service",
    # OpenAI 服務（暫時禁用）
    # "OpenAIEmbeddingService",
    # "create_openai_service",
    # "OPENAI_MODELS",
    # 本地模型服務
    "LocalEmbeddingService",
    "create_text2vec_service",
    "create_m3e_service",
    "create_local_service",
    "LOCAL_MODELS",
    # 中文優化服務
    "ChineseOptimizedEmbeddingService",
    "ChineseEmbeddingConfig",
    "create_chinese_optimized_service",
    # 效能評估
    "EmbeddingEvaluator",
    "ChineseEmbeddingEvaluator",
    "PerformanceMetrics",
    "BenchmarkConfig",
    "quick_benchmark",
    "chinese_quality_benchmark",
    # 快取系統
    "EmbeddingCache",
    "MemoryCache",
    "DiskCache",
    "MultiLevelCache",
    "CacheEntry",
    "CacheStrategy",
    "LRUStrategy",
    "LFUStrategy",
    "create_embedding_cache",
    # GPU 加速和記憶體優化（暫時禁用）
    # "DeviceManager",
    # "MemoryOptimizer",
    # "BatchProcessor",
    # "GPUInfo",
    # "MemoryStats",
    # "get_device_manager",
    # "get_memory_optimizer",
    # "create_batch_processor",
    # 使用量監控（暫時禁用）
    # "UsageMonitor",
    # "UsageRecord",
    # "ModelStats",
    # "Alert",
    # "AlertLevel",
    # "get_usage_monitor",
    # "record_embedding_usage"
]
