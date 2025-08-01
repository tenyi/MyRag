"""錯誤追蹤和報告系統。

本模組提供錯誤追蹤、異常處理和錯誤報告功能，包括：
- 錯誤分類和計數
- 異常堆疊追蹤
- 錯誤率統計
- 錯誤告警
"""

import hashlib
import traceback
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from loguru import logger


class ErrorSeverity(str, Enum):
    """錯誤嚴重程度。"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """錯誤類別。"""
    SYSTEM = "system"
    BUSINESS = "business"
    NETWORK = "network"
    DATABASE = "database"
    PROCESSING = "processing"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    EXTERNAL_API = "external_api"
    UNKNOWN = "unknown"


@dataclass
class ErrorRecord:
    """錯誤記錄。"""
    timestamp: datetime
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    module: str
    function: str
    line_number: int
    traceback_hash: str
    traceback_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    count: int = 1
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化後處理。"""
        if self.first_seen is None:
            self.first_seen = self.timestamp
        if self.last_seen is None:
            self.last_seen = self.timestamp


@dataclass
class ErrorStats:
    """錯誤統計。"""
    total_errors: int = 0
    error_rate: float = 0.0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=dict)
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=dict)
    top_errors: List[ErrorRecord] = field(default_factory=list)
    recent_errors: List[ErrorRecord] = field(default_factory=list)


class ErrorTracker:
    """錯誤追蹤器。"""
    
    def __init__(self, max_errors: int = 10000, retention_hours: int = 24):
        """初始化錯誤追蹤器。
        
        Args:
            max_errors: 最大錯誤記錄數
            retention_hours: 錯誤記錄保留小時數
        """
        self.max_errors = max_errors
        self.retention_hours = retention_hours
        self.lock = threading.RLock()
        
        # 錯誤儲存
        self.errors: Dict[str, ErrorRecord] = {}  # 按 hash 分組的錯誤
        self.recent_errors: deque = deque(maxlen=1000)  # 最近錯誤
        
        # 統計計數器
        self.error_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.severity_counts: Dict[ErrorSeverity, int] = defaultdict(int)
        self.total_error_count = 0
        
        # 告警閾值
        self.alert_thresholds = {
            ErrorSeverity.CRITICAL: 1,
            ErrorSeverity.HIGH: 5,
            ErrorSeverity.MEDIUM: 20,
            ErrorSeverity.LOW: 50
        }
        
        # 回調函數
        self.alert_callbacks: List[Callable[[ErrorRecord], None]] = []
    
    def track_error(
        self,
        error: Union[Exception, str],
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        module: Optional[str] = None,
        function: Optional[str] = None
    ) -> str:
        """追蹤錯誤。
        
        Args:
            error: 錯誤或錯誤訊息
            category: 錯誤類別
            severity: 錯誤嚴重程度
            context: 額外上下文資訊
            module: 模組名稱
            function: 函數名稱
            
        Returns:
            錯誤追蹤 ID
        """
        with self.lock:
            # 取得錯誤資訊
            if isinstance(error, Exception):
                error_type = type(error).__name__
                message = str(error)
                tb_text = traceback.format_exc()
            else:
                error_type = "ManualError"
                message = str(error)
                tb_text = ""
            
            # 取得呼叫堆疊資訊
            if not module or not function:
                frame = traceback.extract_stack()[-2]  # 跳過當前函數
                if not module:
                    module = frame.filename.split('/')[-1]
                if not function:
                    function = frame.name
                line_number = frame.lineno
            else:
                line_number = 0
            
            # 產生錯誤雜湊
            error_hash = self._generate_error_hash(error_type, message, module, function)
            
            # 建立或更新錯誤記錄
            now = datetime.now()
            
            if error_hash in self.errors:
                # 更新現有錯誤
                existing_error = self.errors[error_hash]
                existing_error.count += 1
                existing_error.last_seen = now
                existing_error.context.update(context or {})
            else:
                # 建立新錯誤記錄
                error_record = ErrorRecord(
                    timestamp=now,
                    error_type=error_type,
                    message=message,
                    category=category,
                    severity=severity,
                    module=module,
                    function=function,
                    line_number=line_number,
                    traceback_hash=error_hash,
                    traceback_text=tb_text,
                    context=context or {}
                )
                
                self.errors[error_hash] = error_record
                self.recent_errors.append(error_record)
            
            # 更新統計
            self.error_counts[category] += 1
            self.severity_counts[severity] += 1
            self.total_error_count += 1
            
            # 清理過期錯誤
            self._cleanup_old_errors()
            
            # 檢查告警
            self._check_alerts(self.errors[error_hash])
            
            # 記錄到日誌
            logger.bind(
                error_id=error_hash,
                category=category.value,
                severity=severity.value,
                context=context
            ).error(f"{error_type}: {message}")
            
            return error_hash
    
    def _generate_error_hash(
        self,
        error_type: str,
        message: str,
        module: str,
        function: str
    ) -> str:
        """產生錯誤雜湊。"""
        content = f"{error_type}:{message}:{module}:{function}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _cleanup_old_errors(self) -> None:
        """清理過期錯誤。"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        expired_hashes = [
            error_hash for error_hash, error_record in self.errors.items()
            if error_record.last_seen < cutoff_time
        ]
        
        for error_hash in expired_hashes:
            del self.errors[error_hash]
        
        # 限制最大錯誤數量
        if len(self.errors) > self.max_errors:
            # 移除最舊的錯誤
            sorted_errors = sorted(
                self.errors.items(),
                key=lambda x: x[1].last_seen
            )
            
            to_remove = len(self.errors) - self.max_errors
            for error_hash, _ in sorted_errors[:to_remove]:
                del self.errors[error_hash]
    
    def _check_alerts(self, error_record: ErrorRecord) -> None:
        """檢查是否需要發送告警。"""
        threshold = self.alert_thresholds.get(error_record.severity, float('inf'))
        
        if error_record.count >= threshold:
            for callback in self.alert_callbacks:
                try:
                    callback(error_record)
                except Exception as e:
                    logger.error(f"告警回調失敗: {e}")
    
    def add_alert_callback(self, callback: Callable[[ErrorRecord], None]) -> None:
        """添加告警回調函數。
        
        Args:
            callback: 告警回調函數
        """
        self.alert_callbacks.append(callback)
    
    def get_error_stats(self, hours: int = 1) -> ErrorStats:
        """取得錯誤統計。
        
        Args:
            hours: 統計時間範圍（小時）
            
        Returns:
            錯誤統計
        """
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # 篩選時間範圍內的錯誤
            recent_errors = [
                error for error in self.errors.values()
                if error.last_seen >= cutoff_time
            ]
            
            # 計算統計
            total_errors = len(recent_errors)
            error_rate = total_errors / hours if hours > 0 else 0
            
            # 按類別統計
            errors_by_category = defaultdict(int)
            for error in recent_errors:
                errors_by_category[error.category] += error.count
            
            # 按嚴重程度統計
            errors_by_severity = defaultdict(int)
            for error in recent_errors:
                errors_by_severity[error.severity] += error.count
            
            # 排序獲取最多的錯誤
            top_errors = sorted(
                recent_errors,
                key=lambda x: x.count,
                reverse=True
            )[:10]
            
            # 最近錯誤（按時間排序）
            recent_error_list = sorted(
                recent_errors,
                key=lambda x: x.last_seen,
                reverse=True
            )[:20]
            
            return ErrorStats(
                total_errors=total_errors,
                error_rate=error_rate,
                errors_by_category=dict(errors_by_category),
                errors_by_severity=dict(errors_by_severity),
                top_errors=top_errors,
                recent_errors=recent_error_list
            )
    
    def get_error_by_id(self, error_id: str) -> Optional[ErrorRecord]:
        """根據 ID 取得錯誤記錄。
        
        Args:
            error_id: 錯誤 ID
            
        Returns:
            錯誤記錄
        """
        return self.errors.get(error_id)
    
    def clear_errors(self) -> None:
        """清除所有錯誤記錄。"""
        with self.lock:
            self.errors.clear()
            self.recent_errors.clear()
            self.error_counts.clear()
            self.severity_counts.clear()
            self.total_error_count = 0
    
    def export_errors(self, format: str = "json") -> Union[str, Dict]:
        """匯出錯誤記錄。
        
        Args:
            format: 匯出格式 (json, csv)
            
        Returns:
            匯出的錯誤資料
        """
        with self.lock:
            if format.lower() == "json":
                return {
                    "total_errors": self.total_error_count,
                    "errors": [
                        {
                            "id": error_hash,
                            "timestamp": error.timestamp.isoformat(),
                            "type": error.error_type,
                            "message": error.message,
                            "category": error.category.value,
                            "severity": error.severity.value,
                            "module": error.module,
                            "function": error.function,
                            "count": error.count,
                            "first_seen": error.first_seen.isoformat() if error.first_seen else None,
                            "last_seen": error.last_seen.isoformat() if error.last_seen else None,
                            "context": error.context
                        }
                        for error_hash, error in self.errors.items()
                    ]
                }
            elif format.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # 寫入標題
                writer.writerow([
                    "ID", "Timestamp", "Type", "Message", "Category", 
                    "Severity", "Module", "Function", "Count", "First Seen", "Last Seen"
                ])
                
                # 寫入資料
                for error_hash, error in self.errors.items():
                    writer.writerow([
                        error_hash,
                        error.timestamp.isoformat(),
                        error.error_type,
                        error.message,
                        error.category.value,
                        error.severity.value,
                        error.module,
                        error.function,
                        error.count,
                        error.first_seen.isoformat() if error.first_seen else "",
                        error.last_seen.isoformat() if error.last_seen else ""
                    ])
                
                return output.getvalue()
            else:
                raise ValueError(f"不支援的匯出格式: {format}")


# 全域錯誤追蹤器實例
_error_tracker: Optional[ErrorTracker] = None
_error_tracker_lock = threading.Lock()


def get_error_tracker() -> ErrorTracker:
    """取得全域錯誤追蹤器實例。
    
    Returns:
        錯誤追蹤器實例
    """
    global _error_tracker
    
    if _error_tracker is None:
        with _error_tracker_lock:
            if _error_tracker is None:
                _error_tracker = ErrorTracker()
    
    return _error_tracker


def track_error(
    error: Union[Exception, str],
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """便利函數：追蹤錯誤。
    
    Args:
        error: 錯誤或錯誤訊息
        category: 錯誤類別
        severity: 錯誤嚴重程度
        context: 額外上下文資訊
        
    Returns:
        錯誤追蹤 ID
    """
    return get_error_tracker().track_error(error, category, severity, context)


def error_handler(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None
):
    """錯誤處理裝飾器。
    
    Args:
        category: 錯誤類別
        severity: 錯誤嚴重程度
        context: 額外上下文資訊
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                track_error(e, category, severity, context)
                raise
        return wrapper
    return decorator