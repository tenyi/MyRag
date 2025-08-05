"""向量資料庫模組

提供統一的向量儲存管理介面，支援多種向量資料庫後端
"""

from .base import (
    CollectionError,
    ConnectionError,
    HybridSearchConfig,
    RerankingMethod,
    SearchFilter,
    SearchType,
    VectorCollection,
    VectorOperationError,
    VectorSearchResult,
    VectorStore,
    VectorStoreError,
    VectorStoreType,
)
from .lancedb_store import LanceDBStore
from .manager import VectorStoreManager, create_vector_store_manager

__all__ = [
    # 基礎類別和介面
    "VectorStore",
    "VectorStoreType",
    "VectorSearchResult",
    "VectorCollection",
    # 搜尋相關類別
    "SearchType",
    "RerankingMethod",
    "HybridSearchConfig",
    "SearchFilter",
    # 異常類別
    "VectorStoreError",
    "ConnectionError",
    "CollectionError",
    "VectorOperationError",
    # 具體實作
    "LanceDBStore",
    # 管理器
    "VectorStoreManager",
    "create_vector_store_manager",
]
