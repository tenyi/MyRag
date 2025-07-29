"""
Embedding 服務模組

提供多種 embedding 模型的統一介面和管理功能
支援 BGE-M3、OpenAI、text2vec、m3e 等模型
"""

from .base import (
    EmbeddingService,
    EmbeddingResult,
    EmbeddingModelType,
    ModelMetrics,
    EmbeddingServiceError,
    ModelLoadError,
    EmbeddingComputeError
)

from .manager import EmbeddingManager

from .bge_m3 import (
    BGEM3EmbeddingService,
    create_bge_m3_service
)

from .openai_service import (
    OpenAIEmbeddingService,
    create_openai_service,
    OPENAI_MODELS
)

from .local_models import (
    LocalEmbeddingService,
    create_text2vec_service,
    create_m3e_service,
    create_local_service,
    LOCAL_MODELS
)

from .chinese_optimized import (
    ChineseOptimizedEmbeddingService,
    ChineseEmbeddingConfig,
    create_chinese_optimized_service
)

from .evaluation import (
    EmbeddingEvaluator,
    ChineseEmbeddingEvaluator,
    PerformanceMetrics,
    BenchmarkConfig,
    quick_benchmark,
    chinese_quality_benchmark
)

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
    
    # BGE-M3 服務
    "BGEM3EmbeddingService",
    "create_bge_m3_service",
    
    # OpenAI 服務
    "OpenAIEmbeddingService", 
    "create_openai_service",
    "OPENAI_MODELS",
    
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
    "chinese_quality_benchmark"
]