"""
基礎例外類別定義

定義系統中所有例外的基礎類別和主要分類。
"""

import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ErrorSeverity(Enum):
    """錯誤嚴重程度枚舉"""

    LOW = "low"  # 低 - 不影響核心功能的警告
    MEDIUM = "medium"  # 中 - 影響部分功能但系統可繼續運行
    HIGH = "high"  # 高 - 嚴重影響系統功能
    CRITICAL = "critical"  # 嚴重 - 系統無法正常運行


class ErrorCategory(Enum):
    """錯誤類別枚舉"""

    SYSTEM = "system"  # 系統相關錯誤
    CONFIGURATION = "config"  # 配置相關錯誤
    VALIDATION = "validation"  # 驗證相關錯誤
    PROCESSING = "processing"  # 處理相關錯誤
    RESOURCE = "resource"  # 資源相關錯誤
    NETWORK = "network"  # 網路相關錯誤
    SECURITY = "security"  # 安全相關錯誤
    DATABASE = "database"  # 資料庫相關錯誤
    MODEL = "model"  # 模型相關錯誤
    EMBEDDING = "embedding"  # Embedding 相關錯誤
    QUERY = "query"  # 查詢相關錯誤
    INDEX = "index"  # 索引相關錯誤


class ChineseGraphRAGError(Exception):
    """
    中文 GraphRAG 系統基礎例外類別

    所有系統例外都應繼承此類別，提供統一的錯誤處理介面。
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化例外

        Args:
            message: 錯誤訊息
            error_code: 錯誤代碼，用於識別特定錯誤類型
            category: 錯誤類別
            severity: 錯誤嚴重程度
            details: 錯誤詳細資訊
            cause: 原始例外
            suggestions: 解決建議
            context: 錯誤發生時的上下文資訊
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.suggestions = suggestions or []
        self.context = context or {}
        self.timestamp = datetime.now()
        self.traceback_info = traceback.format_exc()

    def _generate_error_code(self) -> str:
        """生成錯誤代碼"""
        class_name = self.__class__.__name__
        return f"{class_name.upper()}_{id(self) % 10000:04d}"

    def to_dict(self) -> Dict[str, Any]:
        """將例外轉換為字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "suggestions": self.suggestions,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_info,
        }

    def add_context(self, key: str, value: Any) -> "ChineseGraphRAGError":
        """添加上下文資訊"""
        self.context[key] = value
        return self

    def add_suggestion(self, suggestion: str) -> "ChineseGraphRAGError":
        """添加解決建議"""
        self.suggestions.append(suggestion)
        return self

    def __str__(self) -> str:
        """字串表示"""
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        """詳細表示"""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"code='{self.error_code}', "
            f"category={self.category.value}, "
            f"severity={self.severity.value})"
        )


class SystemError(ChineseGraphRAGError):
    """系統層級錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ConfigurationError(ChineseGraphRAGError):
    """配置相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ValidationError(ChineseGraphRAGError):
    """驗證相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ProcessingError(ChineseGraphRAGError):
    """處理相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.PROCESSING)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ResourceError(ChineseGraphRAGError):
    """資源相關錯誤（記憶體、磁碟、CPU等）"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.RESOURCE)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class NetworkError(ChineseGraphRAGError):
    """網路相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.NETWORK)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class SecurityError(ChineseGraphRAGError):
    """安全相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.SECURITY)
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class DatabaseError(ChineseGraphRAGError):
    """資料庫相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.DATABASE)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ModelError(ChineseGraphRAGError):
    """模型相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.MODEL)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class EmbeddingError(ChineseGraphRAGError):
    """Embedding 相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.EMBEDDING)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class QueryError(ChineseGraphRAGError):
    """查詢相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.QUERY)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class IndexError(ChineseGraphRAGError):
    """索引相關錯誤"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.INDEX)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


# 特定子系統的例外類別


class DocumentProcessingError(ProcessingError):
    """文件處理錯誤"""

    pass


class EmbeddingComputeError(EmbeddingError):
    """Embedding 計算錯誤"""

    pass


class VectorStoreError(DatabaseError):
    """向量資料庫錯誤"""

    pass


class LLMError(ModelError):
    """LLM 相關錯誤"""

    pass


class GraphRAGError(ProcessingError):
    """GraphRAG 處理錯誤"""

    pass
