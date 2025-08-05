"""
中文 GraphRAG 系統例外處理模組

此模組定義了系統中使用的所有自訂例外類別，
提供統一的錯誤處理和分類機制。
"""

from .base import (
    ChineseGraphRAGError,
    ConfigurationError,
    NetworkError,
    ProcessingError,
    ResourceError,
    SecurityError,
    SystemError,
    ValidationError,
)
from .consistency import (
    ConsistencyChecker,
    ConsistencyIssue,
    ConsistencyReport,
    DataConsistencyManager,
    get_consistency_manager,
)
from .handlers import (
    ErrorHandler,
    FallbackHandler,
    GlobalErrorHandler,
    HandlingStrategy,
    RetryHandler,
    get_error_handler,
)
from .incremental import (
    ChangeRecord,
    ChangeType,
    FileWatcher,
    IncrementalIndexManager,
    IncrementalIndexStorage,
    IndexStatus,
)
from .recovery import (
    CheckpointManager,
    RecoveryManager,
    StateManager,
    get_recovery_manager,
)
from .retry import (
    ExponentialBackoffPolicy,
    FixedDelayPolicy,
    LinearBackoffPolicy,
    RetryPolicy,
    async_retry_with_policy,
    retry_with_policy,
)

__all__ = [
    # 基礎例外類別
    "ChineseGraphRAGError",
    "SystemError",
    "ConfigurationError",
    "ValidationError",
    "ProcessingError",
    "ResourceError",
    "NetworkError",
    "SecurityError",
    # 錯誤處理器
    "ErrorHandler",
    "GlobalErrorHandler",
    "RetryHandler",
    "FallbackHandler",
    "HandlingStrategy",
    "get_error_handler",
    # 恢復機制
    "RecoveryManager",
    "CheckpointManager",
    "StateManager",
    "get_recovery_manager",
    # 增量索引
    "IncrementalIndexManager",
    "IncrementalIndexStorage",
    "FileWatcher",
    "ChangeRecord",
    "ChangeType",
    "IndexStatus",
    # 一致性檢查
    "DataConsistencyManager",
    "ConsistencyChecker",
    "ConsistencyReport",
    "ConsistencyIssue",
    "get_consistency_manager",
    # 重試機制
    "RetryPolicy",
    "ExponentialBackoffPolicy",
    "LinearBackoffPolicy",
    "FixedDelayPolicy",
    "retry_with_policy",
    "async_retry_with_policy",
]
