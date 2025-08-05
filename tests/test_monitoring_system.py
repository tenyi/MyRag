"""監控系統測試。

測試監控系統的核心功能：
- 日誌記錄
- 錯誤追蹤
- 系統監控
- 效能指標收集
"""

import asyncio
import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chinese_graphrag.monitoring import (
    get_error_tracker,
    get_logger,
    get_metrics_collector,
    get_system_monitor,
    setup_logging,
)
from src.chinese_graphrag.monitoring.error_tracker import (
    ErrorCategory,
    ErrorSeverity,
    ErrorTracker,
    track_error,
)
from src.chinese_graphrag.monitoring.logger import LogConfig, LoggerManager
from src.chinese_graphrag.monitoring.metrics import MetricsCollector, MetricType
from src.chinese_graphrag.monitoring.system_monitor import SystemMonitor


class TestLoggerManager:
    """日誌管理器測試。"""

    def test_setup_logging_with_default_config(self, tmp_path):
        """測試使用預設配置設定日誌。"""
        log_config = LogConfig(
            log_dir=str(tmp_path),
            console_enabled=False,  # 關閉控制台輸出以便測試
            file_enabled=True,
        )

        manager = LoggerManager(log_config)
        manager.setup_logging()

        # 檢查日誌檔案是否建立
        log_file = tmp_path / log_config.log_file
        assert log_file.parent.exists()

    def test_get_logger_with_context(self, tmp_path):
        """測試取得帶上下文的日誌器。"""
        log_config = LogConfig(
            log_dir=str(tmp_path), console_enabled=False, file_enabled=True
        )

        manager = LoggerManager(log_config)
        logger = manager.get_logger("test_module")

        # 測試日誌記錄（這裡只是確保不會拋出異常）
        logger.info("測試訊息")

        # 測試添加上下文
        context_logger = manager.add_context(user_id="123", operation="test")
        context_logger.info("帶上下文的測試訊息")

    def test_update_log_level(self, tmp_path):
        """測試動態更新日誌級別。"""
        log_config = LogConfig(
            log_dir=str(tmp_path),
            level="INFO",
            console_enabled=False,
            file_enabled=True,
        )

        manager = LoggerManager(log_config)
        manager.setup_logging()

        # 更新日誌級別
        manager.update_level("DEBUG")
        assert manager.config.level == "DEBUG"

    def test_json_logging(self, tmp_path):
        """測試 JSON 格式日誌。"""
        log_config = LogConfig(
            log_dir=str(tmp_path),
            console_enabled=False,
            file_enabled=False,
            json_enabled=True,
        )

        manager = LoggerManager(log_config)
        logger = manager.get_logger("test_json")

        # 記錄測試訊息
        logger.info("JSON 測試訊息", extra={"test_key": "test_value"})

        # 檢查 JSON 檔案是否建立
        json_file = tmp_path / log_config.json_file
        assert json_file.exists()


class TestErrorTracker:
    """錯誤追蹤器測試。"""

    def test_track_exception(self):
        """測試追蹤異常。"""
        tracker = ErrorTracker()

        try:
            raise ValueError("測試錯誤")
        except Exception as e:
            error_id = tracker.track_error(
                e, category=ErrorCategory.VALIDATION, severity=ErrorSeverity.HIGH
            )

            assert error_id is not None
            assert len(error_id) == 16  # MD5 hash 的前 16 個字元

            # 檢查錯誤記錄
            error_record = tracker.get_error_by_id(error_id)
            assert error_record is not None
            assert error_record.error_type == "ValueError"
            assert error_record.message == "測試錯誤"
            assert error_record.category == ErrorCategory.VALIDATION
            assert error_record.severity == ErrorSeverity.HIGH

    def test_track_string_error(self):
        """測試追蹤字串錯誤。"""
        tracker = ErrorTracker()

        error_id = tracker.track_error(
            "手動錯誤訊息",
            category=ErrorCategory.BUSINESS,
            severity=ErrorSeverity.MEDIUM,
            context={"user_id": "123", "operation": "test"},
        )

        error_record = tracker.get_error_by_id(error_id)
        assert error_record is not None
        assert error_record.error_type == "ManualError"
        assert error_record.message == "手動錯誤訊息"
        assert error_record.context["user_id"] == "123"

    def test_error_aggregation(self):
        """測試錯誤聚合。"""
        tracker = ErrorTracker()

        # 追蹤相同的錯誤多次
        error_ids = []
        for i in range(5):
            try:
                raise RuntimeError("重複錯誤")
            except Exception as e:
                error_id = tracker.track_error(e)
                error_ids.append(error_id)

        # 所有錯誤 ID 應該相同（因為是相同的錯誤）
        assert len(set(error_ids)) == 1

        # 檢查計數
        error_record = tracker.get_error_by_id(error_ids[0])
        assert error_record.count == 5

    def test_error_stats(self):
        """測試錯誤統計。"""
        tracker = ErrorTracker()

        # 追蹤不同類型的錯誤
        tracker.track_error("錯誤1", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        tracker.track_error("錯誤2", ErrorCategory.NETWORK, ErrorSeverity.MEDIUM)
        tracker.track_error("錯誤3", ErrorCategory.SYSTEM, ErrorSeverity.LOW)

        stats = tracker.get_error_stats(hours=1)

        assert stats.total_errors == 3
        assert stats.errors_by_category[ErrorCategory.SYSTEM] == 2
        assert stats.errors_by_category[ErrorCategory.NETWORK] == 1
        assert stats.errors_by_severity[ErrorSeverity.HIGH] == 1
        assert stats.errors_by_severity[ErrorSeverity.MEDIUM] == 1
        assert stats.errors_by_severity[ErrorSeverity.LOW] == 1

    def test_alert_callback(self):
        """測試告警回調。"""
        tracker = ErrorTracker()
        alert_called = []

        def alert_callback(error_record):
            alert_called.append(error_record)

        tracker.add_alert_callback(alert_callback)

        # 設定低閾值以觸發告警
        tracker.alert_thresholds[ErrorSeverity.MEDIUM] = 1

        # 追蹤錯誤
        tracker.track_error("測試告警", severity=ErrorSeverity.MEDIUM)

        # 檢查告警是否被觸發
        assert len(alert_called) == 1
        assert alert_called[0].message == "測試告警"

    def test_export_errors(self):
        """測試匯出錯誤。"""
        tracker = ErrorTracker()

        # 追蹤一些錯誤
        tracker.track_error("錯誤1", ErrorCategory.SYSTEM)
        tracker.track_error("錯誤2", ErrorCategory.NETWORK)

        # 匯出為 JSON
        json_data = tracker.export_errors("json")
        assert isinstance(json_data, dict)
        assert json_data["total_errors"] == 2
        assert len(json_data["errors"]) == 2

        # 匯出為 CSV
        csv_data = tracker.export_errors("csv")
        assert isinstance(csv_data, str)
        assert "ID,Timestamp,Type" in csv_data


class TestMetricsCollector:
    """指標收集器測試。"""

    def test_record_counter(self, tmp_path):
        """測試記錄計數器指標。"""
        collector = MetricsCollector(storage_dir=str(tmp_path))

        # 記錄計數器
        collector.record_counter("test.requests", 1, {"endpoint": "/api/test"})
        collector.record_counter("test.requests", 2, {"endpoint": "/api/test"})

        # 檢查指標
        summary = collector.get_metrics_summary()
        counter_key = 'test.requests:{"endpoint": "/api/test"}'
        assert counter_key in summary["counters"]
        assert summary["counters"][counter_key] == 3  # 1 + 2

    def test_record_gauge(self, tmp_path):
        """測試記錄儀表指標。"""
        collector = MetricsCollector(storage_dir=str(tmp_path))

        # 記錄儀表
        collector.record_gauge("test.cpu_usage", 45.5, {"host": "server1"})
        collector.record_gauge("test.cpu_usage", 50.2, {"host": "server1"})

        # 檢查指標（儀表只保留最新值）
        summary = collector.get_metrics_summary()
        gauge_key = 'test.cpu_usage:{"host": "server1"}'
        assert summary["gauges"][gauge_key] == 50.2

    def test_record_timer(self, tmp_path):
        """測試記錄計時器指標。"""
        collector = MetricsCollector(storage_dir=str(tmp_path))

        # 記錄計時器
        collector.record_timer("test.query_time", 0.1)
        collector.record_timer("test.query_time", 0.2)
        collector.record_timer("test.query_time", 0.15)

        # 檢查指標
        summary = collector.get_metrics_summary()
        timer_key = "test.query_time:{}"
        assert timer_key in summary["timers"]

        timer_stats = summary["timers"][timer_key]
        assert timer_stats["count"] == 3
        assert abs(timer_stats["avg_time"] - 0.15) < 1e-10  # (0.1 + 0.2 + 0.15) / 3
        assert timer_stats["min_time"] == 0.1
        assert timer_stats["max_time"] == 0.2

    def test_timer_context_manager(self, tmp_path):
        """測試計時器上下文管理器。"""
        collector = MetricsCollector(storage_dir=str(tmp_path))

        # 使用上下文管理器
        with collector.timer("test.operation"):
            time.hasher = time.time  # 確保有一些執行時間
            time.sleep(0.01)  # 短暫延遲

        # 檢查是否記錄了計時
        summary = collector.get_metrics_summary()
        timer_key = "test.operation:{}"
        assert timer_key in summary["timers"]
        assert summary["timers"][timer_key]["count"] == 1

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, tmp_path):
        """測試系統指標收集。"""
        collector = MetricsCollector(
            storage_dir=str(tmp_path), collection_interval=1  # 1秒間隔
        )

        # 手動觸發系統指標收集
        await collector._collect_system_metrics()

        # 檢查系統指標
        assert len(collector.system_metrics) > 0

        latest_metric = collector.system_metrics[-1]
        assert latest_metric.cpu_percent >= 0
        assert latest_metric.memory_percent >= 0
        assert latest_metric.process_count > 0

    @pytest.mark.asyncio
    async def test_export_metrics(self, tmp_path):
        """測試匯出指標。"""
        collector = MetricsCollector(storage_dir=str(tmp_path))

        # 記錄一些指標
        collector.record_counter("test.exports", 1)
        collector.record_gauge("test.value", 42.0)

        # 匯出指標
        result = await collector.export_metrics()
        assert result is True

        # 檢查匯出檔案
        export_files = list(tmp_path.glob("metrics_export_*.json"))
        assert len(export_files) == 1

        # 驗證匯出內容
        with open(export_files[0], "r", encoding="utf-8") as f:
            exported_data = json.load(f)

        assert "counters" in exported_data
        assert "gauges" in exported_data


class TestSystemMonitor:
    """系統監控器測試。"""

    def test_collect_system_stats(self):
        """測試收集系統統計。"""
        monitor = SystemMonitor()

        stats = monitor.collect_system_stats()

        assert stats.cpu_percent >= 0
        assert stats.memory_percent >= 0
        assert stats.disk_percent >= 0
        assert stats.process_count > 0
        assert isinstance(stats.timestamp, datetime)

    def test_system_monitoring_lifecycle(self):
        """測試系統監控生命週期。"""
        monitor = SystemMonitor(sample_interval=1)  # 1秒間隔

        # 開始監控
        monitor.start_monitoring()
        assert monitor.monitoring is True
        assert monitor.monitor_thread is not None

        # 等待一段時間讓監控收集資料
        time.sleep(2)

        # 檢查是否收集到資料
        current_stats = monitor.get_current_stats()
        assert current_stats is not None

        # 停止監控
        monitor.stop_monitoring()
        assert monitor.monitoring is False

    def test_get_process_stats(self):
        """測試取得程序統計。"""
        monitor = SystemMonitor()

        process_stats = monitor.get_process_stats(top_n=5)

        assert len(process_stats) <= 5
        if len(process_stats) > 0:
            proc = process_stats[0]
            assert proc.pid >= 0  # PID 可以是 0（如 kernel_task）
            assert proc.name is not None
            assert proc.cpu_percent >= 0

    def test_health_checks(self):
        """測試健康檢查。"""
        monitor = SystemMonitor()

        # 添加測試健康檢查
        def test_health_check():
            from src.chinese_graphrag.monitoring.system_monitor import HealthCheck

            return HealthCheck(
                name="test_check",
                status="healthy",
                message="測試通過",
                timestamp=datetime.now(),
                details={"test": True},
            )

        monitor.add_health_check("test", test_health_check)

        # 執行健康檢查
        results = monitor.run_health_checks()

        assert len(results) == 1
        assert results[0].name == "test_check"
        assert results[0].status == "healthy"

    def test_alert_system(self):
        """測試告警系統。"""
        monitor = SystemMonitor(
            alert_thresholds={
                "cpu_percent": 0.1,  # 設定很低的閾值以觸發告警
                "memory_percent": 0.1,
            }
        )

        alerts_received = []

        def alert_callback(alert_type, alert_data):
            alerts_received.append((alert_type, alert_data))

        monitor.add_alert_callback(alert_callback)

        # 手動觸發告警檢查（使用高值）
        from src.chinese_graphrag.monitoring.system_monitor import SystemStats

        high_usage_stats = SystemStats(
            timestamp=datetime.now(),
            cpu_percent=95.0,  # 高 CPU 使用率
            memory_percent=90.0,  # 高記憶體使用率
            memory_used_gb=8.0,
            memory_total_gb=16.0,
            disk_percent=50.0,
            disk_used_gb=100.0,
            disk_total_gb=200.0,
            network_sent_mb=10.0,
            network_recv_mb=5.0,
            process_count=100,
        )

        monitor._check_system_alerts(high_usage_stats)

        # 檢查是否收到告警
        assert len(alerts_received) >= 1


class TestIntegration:
    """整合測試。"""

    def test_monitoring_system_integration(self, tmp_path):
        """測試監控系統整合。"""
        # 設定日誌
        setup_logging()

        # 取得各個監控元件
        logger = get_logger("integration_test")
        error_tracker = get_error_tracker()
        metrics_collector = get_metrics_collector()
        system_monitor = get_system_monitor()

        # 測試日誌記錄
        logger.info("整合測試開始")

        # 測試錯誤追蹤
        error_id = track_error("整合測試錯誤", ErrorCategory.SYSTEM, ErrorSeverity.LOW)
        assert error_id is not None

        # 測試指標記錄
        metrics_collector.record_counter("integration.test_runs", 1)
        metrics_collector.record_gauge("integration.test_value", 100.0)

        # 測試系統監控
        stats = system_monitor.collect_system_stats()
        assert stats is not None

        # 檢查所有元件都正常工作
        error_stats = error_tracker.get_error_stats()
        assert error_stats.total_errors >= 1

        metrics_summary = metrics_collector.get_metrics_summary()
        assert len(metrics_summary["counters"]) >= 1
        assert len(metrics_summary["gauges"]) >= 1


# 測試夾具
@pytest.fixture
def temp_log_dir(tmp_path):
    """建立臨時日誌目錄。"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


@pytest.fixture
def temp_metrics_dir(tmp_path):
    """建立臨時指標目錄。"""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    return metrics_dir


if __name__ == "__main__":
    pytest.main([__file__])
