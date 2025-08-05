"""
測試異常處理模組

測試自訂異常類別、錯誤處理策略和恢復機制。
"""

import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from src.chinese_graphrag.exceptions import (
    ChineseGraphRAGError,
    ConfigurationError,
    ExponentialBackoffPolicy,
    FallbackHandler,
    FixedDelayPolicy,
    GlobalErrorHandler,
    LinearBackoffPolicy,
    NetworkError,
    ProcessingError,
    ResourceError,
    RetryHandler,
    SecurityError,
    SystemError,
    ValidationError,
    async_retry_with_policy,
    get_error_handler,
    retry_with_policy,
)


class TestChineseGraphRAGError:
    """測試基礎異常類別"""

    def test_basic_error_creation(self):
        """測試基本異常建立"""
        error = ChineseGraphRAGError("測試錯誤訊息")

        assert "測試錯誤訊息" in str(error)
        assert error.message == "測試錯誤訊息"
        assert error.error_code is not None  # 會自動生成錯誤代碼
        assert error.suggestions == []

    def test_error_with_full_metadata(self):
        """測試包含完整元資料的異常"""
        from src.chinese_graphrag.exceptions.base import ErrorCategory, ErrorSeverity

        error = ChineseGraphRAGError(
            "詳細錯誤",
            error_code="ERR_001",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            details={"field": "test_field", "value": "invalid"},
            suggestions=["檢查輸入格式", "參考文件"],
        )

        assert error.error_code == "ERR_001"
        assert error.category.value == "validation"
        assert error.severity.value == "high"
        assert error.details["field"] == "test_field"
        assert len(error.suggestions) == 2

    def test_error_serialization(self):
        """測試異常序列化"""
        error = ChineseGraphRAGError(
            "序列化測試", error_code="SER_001", details={"key": "value"}
        )

        error_dict = error.to_dict()

        assert error_dict["message"] == "序列化測試"
        assert error_dict["error_code"] == "SER_001"
        assert error_dict["details"]["key"] == "value"
        assert "timestamp" in error_dict


class TestSpecificErrors:
    """測試特定異常類別"""

    def test_system_error(self):
        """測試系統錯誤"""
        error = SystemError("系統故障")
        assert isinstance(error, ChineseGraphRAGError)
        assert error.message == "系統故障"

    def test_configuration_error(self):
        """測試配置錯誤"""
        error = ConfigurationError("配置無效")
        assert isinstance(error, ChineseGraphRAGError)
        assert error.message == "配置無效"

    def test_validation_error(self):
        """測試驗證錯誤"""
        error = ValidationError("資料格式錯誤")
        assert isinstance(error, ChineseGraphRAGError)
        assert error.message == "資料格式錯誤"

    def test_processing_error(self):
        """測試處理錯誤"""
        error = ProcessingError("處理失敗")
        assert isinstance(error, ChineseGraphRAGError)
        assert error.message == "處理失敗"

    def test_resource_error(self):
        """測試資源錯誤"""
        error = ResourceError("資源不足")
        assert isinstance(error, ChineseGraphRAGError)
        assert error.message == "資源不足"

    def test_network_error(self):
        """測試網路錯誤"""
        error = NetworkError("網路連接失敗")
        assert isinstance(error, ChineseGraphRAGError)
        assert error.message == "網路連接失敗"

    def test_security_error(self):
        """測試安全錯誤"""
        error = SecurityError("權限不足")
        assert isinstance(error, ChineseGraphRAGError)
        assert error.message == "權限不足"


class TestGlobalErrorHandler:
    """測試全域錯誤處理器"""

    def test_handler_creation(self):
        """測試處理器建立"""
        handler = GlobalErrorHandler()
        assert handler is not None
        assert len(handler.handlers) > 0

    def test_error_handling_strategies(self):
        """測試錯誤處理策略"""
        handler = GlobalErrorHandler()

        # 測試日志記錄策略
        from src.chinese_graphrag.exceptions.handlers import HandlingStrategy

        error = ValidationError("測試驗證錯誤")

        with patch("logging.Logger.error") as mock_logger:
            result = handler.handle_error(error, strategy=HandlingStrategy.LOG)
            mock_logger.assert_called_once()

    def test_retry_strategy(self):
        """測試重試策略"""
        handler = GlobalErrorHandler()

        # 模擬可重試的錯誤
        error = NetworkError("網路暫時不可用")

        with patch("time.sleep"):  # 避免實際等待
            result = handler.handle_error(
                error,
                strategy=handler.HandlingStrategy.RETRY,
                context={"max_retries": 3},
            )


class TestRetryPolicies:
    """測試重試策略"""

    def test_exponential_backoff_policy(self):
        """測試指數退避策略"""
        policy = ExponentialBackoffPolicy(
            base_delay=1.0, backoff_factor=2.0, max_delay=10.0, max_attempts=5
        )

        # 測試延遲計算
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 4.0
        assert policy.get_delay(4) == 8.0
        assert policy.get_delay(5) == 10.0  # 不超過最大延遲

        # 測試是否應該重試
        assert policy.should_retry(1) is True
        assert policy.should_retry(5) is True
        assert policy.should_retry(6) is False

    def test_linear_backoff_policy(self):
        """測試線性退避策略"""
        policy = LinearBackoffPolicy(
            base_delay=1.0, increment=0.5, max_delay=5.0, max_attempts=4
        )

        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 1.5
        assert policy.get_delay(3) == 2.0
        assert policy.get_delay(4) == 2.5
        assert policy.get_delay(10) == 5.0  # 不超過最大延遲

    def test_fixed_delay_policy(self):
        """測試固定延遲策略"""
        policy = FixedDelayPolicy(delay=2.0, max_attempts=3)

        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 2.0

        assert policy.should_retry(3) is True
        assert policy.should_retry(4) is False

    def test_retry_decorator(self):
        """測試重試裝飾器"""
        policy = FixedDelayPolicy(delay=0.1, max_attempts=3)

        call_count = 0

        @retry_with_policy(policy)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("網路錯誤")
            return "成功"

        result = failing_function()
        assert result == "成功"
        assert call_count == 3

    def test_retry_decorator_failure(self):
        """測試重試裝飾器失敗情況"""
        policy = FixedDelayPolicy(delay=0.1, max_attempts=2)

        call_count = 0

        @retry_with_policy(policy)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise NetworkError("持續網路錯誤")

        with pytest.raises(NetworkError):
            always_failing_function()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_retry_decorator(self):
        """測試異步重試裝飾器"""
        policy = FixedDelayPolicy(delay=0.1, max_attempts=3)

        call_count = 0

        @async_retry_with_policy(policy)
        async def failing_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("異步網路錯誤")
            return "異步成功"

        result = await failing_async_function()
        assert result == "異步成功"
        assert call_count == 3


class TestErrorHandlerRegistry:
    """測試錯誤處理器註冊表"""

    def test_get_error_handler(self):
        """測試獲取錯誤處理器"""
        handler = get_error_handler()
        assert handler is not None
        assert isinstance(handler, GlobalErrorHandler)

    def test_singleton_pattern(self):
        """測試單例模式"""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        assert handler1 is handler2


class TestThreadSafety:
    """測試執行緒安全性"""

    def test_concurrent_error_handling(self):
        """測試並發錯誤處理"""
        handler = get_error_handler()
        results = []

        def handle_error_in_thread(thread_id: int):
            error = ProcessingError(f"執行緒 {thread_id} 錯誤")
            try:
                handler.handle_error(error)
                results.append(f"thread_{thread_id}_success")
            except Exception as e:
                results.append(f"thread_{thread_id}_error_{str(e)}")

        # 建立多個執行緒同時處理錯誤
        threads = []
        for i in range(5):
            thread = threading.Thread(target=handle_error_in_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有執行緒完成
        for thread in threads:
            thread.join()

        # 驗證所有執行緒都成功處理了錯誤
        assert len(results) == 5
        assert all("success" in result for result in results)


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """測試錯誤處理整合"""

    def test_real_world_error_scenario(self):
        """測試真實世界錯誤場景"""
        # 模擬一個複雜的錯誤處理場景
        handler = get_error_handler()

        # 第一層錯誤：配置錯誤
        config_error = ConfigurationError(
            "配置檔案格式錯誤",
            error_code="CONFIG_001",
            details={"file": "settings.yaml", "line": 42},
            suggestions=["檢查 YAML 語法", "驗證配置結構"],
        )

        # 處理配置錯誤
        with patch("logging.Logger.error") as mock_logger:
            handler.handle_error(config_error)
            mock_logger.assert_called()

        # 第二層錯誤：由配置錯誤引起的處理錯誤
        processing_error = ProcessingError(
            "無法處理文件", cause=config_error, details={"document_id": "doc_001"}
        )

        # 處理關聯錯誤
        with patch("logging.Logger.error") as mock_logger:
            handler.handle_error(processing_error)
            mock_logger.assert_called()

    def test_error_recovery_flow(self):
        """測試錯誤恢復流程"""
        from src.chinese_graphrag.exceptions.handlers import HandlingStrategy

        handler = get_error_handler()

        # 模擬需要恢復的錯誤
        resource_error = ResourceError(
            "記憶體不足", error_code="MEM_001", suggestions=["釋放資源", "重啟服務"]
        )

        # 測試恢復策略
        with patch.object(handler, "_execute_fallback") as mock_fallback:
            handler.handle_error(
                resource_error,
                strategy=HandlingStrategy.FALLBACK,
                context={"fallback_action": "cleanup_memory"},
            )
            mock_fallback.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
