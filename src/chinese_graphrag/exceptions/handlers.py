"""
錯誤處理器和處理策略

提供統一的錯誤處理機制，包括錯誤分類、處理策略和降級處理。
"""

import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from .base import ChineseGraphRAGError, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HandlingStrategy(Enum):
    """錯誤處理策略枚舉"""

    IGNORE = "ignore"  # 忽略錯誤
    LOG = "log"  # 記錄錯誤
    RETRY = "retry"  # 重試
    FALLBACK = "fallback"  # 降級處理
    ESCALATE = "escalate"  # 上報錯誤
    ABORT = "abort"  # 中止操作


class ErrorHandler(ABC):
    """錯誤處理器抽象基類"""

    def __init__(self, name: str):
        self.name = name
        self.handled_count = 0
        self.last_handled = None

    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """判斷是否可以處理此錯誤"""
        pass

    @abstractmethod
    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """處理錯誤"""
        pass

    def on_handled(self, error: Exception, result: Any):
        """錯誤處理完成後的回調"""
        self.handled_count += 1
        self.last_handled = datetime.now()
        logger.debug(f"錯誤處理器 {self.name} 處理了錯誤: {error}")


class LoggingHandler(ErrorHandler):
    """記錄錯誤處理器"""

    def __init__(self, name: str = "logging", level: int = logging.ERROR):
        super().__init__(name)
        self.level = level

    def can_handle(self, error: Exception) -> bool:
        """所有錯誤都可以記錄"""
        return True

    def handle(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """記錄錯誤"""
        if isinstance(error, ChineseGraphRAGError):
            error_info = error.to_dict()
            logger.log(self.level, f"系統錯誤: {error_info}")
        else:
            logger.log(self.level, f"未分類錯誤: {error}", exc_info=True)

        if context:
            logger.log(self.level, f"錯誤上下文: {context}")


class RetryHandler(ErrorHandler):
    """重試錯誤處理器"""

    def __init__(
        self,
        name: str = "retry",
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_errors: Optional[List[Type[Exception]]] = None,
    ):
        super().__init__(name)
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.retryable_errors = retryable_errors or [
            ConnectionError,
            TimeoutError,
            # 可重試的網路錯誤
        ]

    def can_handle(self, error: Exception) -> bool:
        """判斷錯誤是否可重試"""
        return any(isinstance(error, err_type) for err_type in self.retryable_errors)

    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """執行重試邏輯"""
        attempt = context.get("attempt", 1) if context else 1

        if attempt >= self.max_attempts:
            raise error

        import time

        wait_time = self.delay * (self.backoff_factor ** (attempt - 1))
        logger.warning(
            f"重試錯誤處理 (第 {attempt} 次), 等待 {wait_time:.1f} 秒: {error}"
        )
        time.sleep(wait_time)

        # 返回重試信號
        return {"retry": True, "attempt": attempt + 1}


class FallbackHandler(ErrorHandler):
    """降級處理錯誤處理器"""

    def __init__(
        self,
        name: str = "fallback",
        fallback_func: Optional[Callable] = None,
        fallback_errors: Optional[List[Type[Exception]]] = None,
    ):
        super().__init__(name)
        self.fallback_func = fallback_func
        self.fallback_errors = fallback_errors or [Exception]

    def can_handle(self, error: Exception) -> bool:
        """判斷是否需要降級處理"""
        return any(isinstance(error, err_type) for err_type in self.fallback_errors)

    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """執行降級處理"""
        logger.warning(f"執行降級處理: {error}")

        if self.fallback_func:
            try:
                return self.fallback_func(error, context)
            except Exception as fallback_error:
                logger.error(f"降級處理失敗: {fallback_error}")
                raise fallback_error

        # 預設降級行為：返回 None 或空結果
        return None


class EscalationHandler(ErrorHandler):
    """錯誤上報處理器"""

    def __init__(
        self,
        name: str = "escalation",
        escalation_func: Optional[Callable] = None,
        critical_errors: Optional[List[Type[Exception]]] = None,
    ):
        super().__init__(name)
        self.escalation_func = escalation_func
        self.critical_errors = critical_errors or [
            ChineseGraphRAGError,  # 系統關鍵錯誤
        ]

    def can_handle(self, error: Exception) -> bool:
        """判斷是否需要上報"""
        if isinstance(error, ChineseGraphRAGError):
            return error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        return any(isinstance(error, err_type) for err_type in self.critical_errors)

    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """執行錯誤上報"""
        logger.critical(f"錯誤上報: {error}")

        if self.escalation_func:
            try:
                self.escalation_func(error, context)
            except Exception as escalation_error:
                logger.error(f"錯誤上報失敗: {escalation_error}")

        # 繼續拋出原始錯誤
        raise error


class GlobalErrorHandler:
    """全域錯誤處理器"""

    # 添加 HandlingStrategy 作為類別屬性，以便測試可以訪問
    HandlingStrategy = HandlingStrategy

    def __init__(self):
        self.handlers: List[ErrorHandler] = []
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, datetime] = {}
        self._lock = threading.RLock()

        # 添加預設的錯誤處理器
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """設置預設的錯誤處理器"""
        # 添加重試處理器
        retry_handler = RetryHandler(
            name="default_retry",
            max_attempts=3,
            delay=1.0,
            retryable_errors=[
                ConnectionError,
                TimeoutError,
                # 可以添加更多可重試的錯誤類型
            ],
        )
        self.add_handler(retry_handler)

        # 添加降級處理器
        fallback_handler = FallbackHandler(
            name="default_fallback", fallback_errors=[Exception]  # 處理所有異常
        )
        self.add_handler(fallback_handler)

    def add_handler(self, handler: ErrorHandler):
        """添加錯誤處理器"""
        with self._lock:
            self.handlers.append(handler)
            logger.info(f"添加錯誤處理器: {handler.name}")

    def remove_handler(self, handler_name: str):
        """移除錯誤處理器"""
        with self._lock:
            self.handlers = [h for h in self.handlers if h.name != handler_name]
            logger.info(f"移除錯誤處理器: {handler_name}")

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        strategy: HandlingStrategy = HandlingStrategy.LOG,
    ) -> Any:
        """處理錯誤"""
        with self._lock:
            error_key = f"{type(error).__name__}:{str(error)[:100]}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            self.last_errors[error_key] = datetime.now()

        # 記錄錯誤統計
        logger.debug(f"錯誤處理統計 - {error_key}: {self.error_counts[error_key]} 次")

        # 根據策略選擇處理方式
        if strategy == HandlingStrategy.IGNORE:
            return None
        elif strategy == HandlingStrategy.LOG:
            # 直接記錄錯誤
            if isinstance(error, ChineseGraphRAGError):
                error_info = error.to_dict()
                logger.error(f"系統錯誤: {error_info}")
            else:
                logger.error(f"未分類錯誤: {error}", exc_info=True)

            if context:
                logger.error(f"錯誤上下文: {context}")
            return None
        elif strategy == HandlingStrategy.RETRY:
            # 使用重試處理器
            for handler in self.handlers:
                if isinstance(handler, RetryHandler) and handler.can_handle(error):
                    try:
                        result = handler.handle(error, context)
                        handler.on_handled(error, result)
                        return result
                    except Exception as handler_error:
                        logger.error(f"重試處理器失敗: {handler_error}")
                        break
        elif strategy == HandlingStrategy.FALLBACK:
            # 使用降級處理器
            for handler in self.handlers:
                if isinstance(handler, FallbackHandler) and handler.can_handle(error):
                    try:
                        result = handler.handle(error, context)
                        handler.on_handled(error, result)
                        return result
                    except Exception as handler_error:
                        logger.error(f"降級處理器失敗: {handler_error}")
                        break

        # 找到合適的處理器處理錯誤
        for handler in self.handlers:
            if handler.can_handle(error):
                try:
                    result = handler.handle(error, context)
                    handler.on_handled(error, result)

                    # 如果是重試結果，返回重試信號
                    if isinstance(result, dict) and result.get("retry"):
                        return result

                    return result
                except Exception as handler_error:
                    logger.error(f"錯誤處理器 {handler.name} 處理失敗: {handler_error}")
                    continue

        # 沒有合適的處理器時，使用預設策略
        if strategy == HandlingStrategy.ABORT or isinstance(
            error, ChineseGraphRAGError
        ):
            raise error

        logger.error(f"無法處理的錯誤: {error}", exc_info=True)
        return None

    def _execute_fallback(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ):
        """執行降級處理"""
        logger.info(f"執行降級處理: {error}")
        # 這裡可以實現具體的降級邏輯
        return {"fallback_executed": True, "error": str(error)}

    def get_error_statistics(self) -> Dict[str, Any]:
        """獲取錯誤統計資訊"""
        with self._lock:
            return {
                "total_unique_errors": len(self.error_counts),
                "total_error_count": sum(self.error_counts.values()),
                "error_counts": dict(self.error_counts),
                "recent_errors": {
                    k: v.isoformat()
                    for k, v in self.last_errors.items()
                    if v > datetime.now() - timedelta(hours=24)
                },
                "handler_stats": [
                    {
                        "name": handler.name,
                        "handled_count": handler.handled_count,
                        "last_handled": (
                            handler.last_handled.isoformat()
                            if handler.last_handled
                            else None
                        ),
                    }
                    for handler in self.handlers
                ],
            }

    def clear_statistics(self):
        """清除錯誤統計"""
        with self._lock:
            self.error_counts.clear()
            self.last_errors.clear()
            logger.info("已清除錯誤統計資訊")


# 全域錯誤處理器實例
_global_error_handler = None
_handler_lock = threading.Lock()


def get_error_handler() -> GlobalErrorHandler:
    """獲取全域錯誤處理器實例"""
    global _global_error_handler

    if _global_error_handler is None:
        with _handler_lock:
            if _global_error_handler is None:
                _global_error_handler = GlobalErrorHandler()

                # 添加預設處理器
                _global_error_handler.add_handler(LoggingHandler())
                _global_error_handler.add_handler(RetryHandler())
                _global_error_handler.add_handler(FallbackHandler())
                _global_error_handler.add_handler(EscalationHandler())

    return _global_error_handler


def setup_error_handling(
    log_level: int = logging.ERROR,
    max_retry_attempts: int = 3,
    retry_delay: float = 1.0,
    custom_handlers: Optional[List[ErrorHandler]] = None,
):
    """設置全域錯誤處理"""
    handler = get_error_handler()

    # 清除現有處理器
    handler.handlers.clear()

    # 添加基本處理器
    handler.add_handler(LoggingHandler(level=log_level))
    handler.add_handler(
        RetryHandler(max_attempts=max_retry_attempts, delay=retry_delay)
    )
    handler.add_handler(FallbackHandler())
    handler.add_handler(EscalationHandler())

    # 添加自訂處理器
    if custom_handlers:
        for custom_handler in custom_handlers:
            handler.add_handler(custom_handler)

    logger.info("全域錯誤處理已設置完成")
