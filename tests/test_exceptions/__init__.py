"""
異常處理測試模組

測試中文 GraphRAG 系統的異常處理和恢復機制。
"""

from .test_consistency import *
from .test_incremental import *

# 匯入所有測試模組
from .test_recovery import *

__all__ = [
    # 恢復機制測試
    "TestCheckpointMetadata",
    "TestSystemState",
    "TestFileCheckpointStorage",
    "TestCheckpointManager",
    "TestStateManager",
    "TestRecoveryManager",
    "TestGlobalRecoveryManager",
    "TestRecoveryIntegration",
    # 增量索引測試
    "TestFileMetadata",
    "TestChangeRecord",
    "TestFileWatcher",
    "TestIncrementalIndexStorage",
    "TestIncrementalIndexManager",
    "TestIncrementalIndexingIntegration",
    # 一致性檢查測試
    "TestConsistencyIssue",
    "TestConsistencyReport",
    "TestFileSystemChecker",
    "TestIndexChecker",
    "TestVectorStoreChecker",
    "TestMetadataChecker",
    "TestDataConsistencyManager",
    "TestGlobalConsistencyManager",
    "TestConsistencyIntegration",
]
