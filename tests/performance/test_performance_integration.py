"""
效能優化整合測試

測試所有效能優化模組的整合功能
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.chinese_graphrag.performance import (
    BatchOptimizer,
    BenchmarkRunner,
    CostOptimizer,
    OptimizationConfig,
    OptimizerManager,
    PerformanceMonitor,
    QueryOptimizer,
)


class TestPerformanceIntegration:
    """效能優化整合測試"""

    @pytest.fixture
    def temp_storage(self):
        """臨時儲存目錄"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def optimization_config(self, temp_storage):
        """測試用優化配置"""
        return OptimizationConfig(
            batch_enabled=True,
            batch_size=16,
            max_batch_size=64,
            parallel_workers=2,
            memory_threshold_mb=512.0,
            query_cache_enabled=True,
            cache_ttl_seconds=300,
            cache_max_size=1000,
            preload_enabled=True,
            cost_tracking_enabled=True,
            budget_limit_usd=100.0,
            quality_threshold=0.8,
            monitoring_enabled=True,
            monitoring_interval=1.0,
            alert_thresholds={
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "error_rate": 5.0,
            },
            storage_path=temp_storage,
        )

    @pytest_asyncio.fixture
    async def optimizer_manager(self, optimization_config):
        """優化管理器實例"""
        manager = OptimizerManager(optimization_config)
        await manager.initialize()
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_optimizer_manager_initialization(self, optimizer_manager):
        """測試優化管理器初始化"""
        # 檢查初始化狀態
        assert optimizer_manager._initialized
        assert optimizer_manager.batch_optimizer is not None
        assert optimizer_manager.query_optimizer is not None
        assert optimizer_manager.cost_optimizer is not None
        assert optimizer_manager.performance_monitor is not None
        assert optimizer_manager.benchmark_runner is not None

        # 檢查效能狀態
        status = optimizer_manager.get_performance_status()
        assert status["initialized"]
        assert status["optimizers"]["batch_optimizer"]
        assert status["optimizers"]["query_optimizer"]
        assert status["optimizers"]["cost_optimizer"]
        assert status["optimizers"]["performance_monitor"]

    @pytest.mark.asyncio
    async def test_batch_processing_optimization(self, optimizer_manager):
        """測試批次處理優化"""

        # 模擬處理函數
        async def mock_process_func(item):
            await asyncio.sleep(0.01)  # 模擬處理時間
            return f"processed_{item}"

        # 測試資料
        test_items = list(range(50))

        # 執行批次優化
        start_time = time.time()
        results = await optimizer_manager.optimize_batch_processing(
            items=test_items, process_func=mock_process_func
        )
        end_time = time.time()

        # 驗證結果
        assert len(results) == len(test_items)
        assert all(f"processed_{i}" in results for i in test_items)

        # 檢查統計資料
        status = optimizer_manager.get_performance_status()
        assert status["statistics"]["batch_optimizations"] > 0

        # 批次處理應該比逐一處理更快
        processing_time = end_time - start_time
        assert processing_time < 1.0  # 應該在1秒內完成

    @pytest.mark.asyncio
    async def test_query_optimization(self, optimizer_manager):
        """測試查詢優化"""
        # 模擬查詢函數
        call_count = 0

        async def mock_query_func(query):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # 模擬查詢時間
            return f"result_for_{query}"

        test_query = "test_query_123"

        # 第一次查詢（應該執行函數）
        result1 = await optimizer_manager.optimize_query(
            query=test_query, query_func=mock_query_func
        )
        assert result1 == f"result_for_{test_query}"
        assert call_count == 1

        # 第二次查詢（應該從快取取得）
        start_time = time.time()
        result2 = await optimizer_manager.optimize_query(
            query=test_query, query_func=mock_query_func
        )
        end_time = time.time()

        assert result2 == f"result_for_{test_query}"
        assert call_count == 1  # 函數不應該再次被呼叫
        assert (end_time - start_time) < 0.05  # 快取查詢應該很快

        # 檢查快取統計
        status = optimizer_manager.get_performance_status()
        assert status["statistics"]["cache_hits"] > 0

    @pytest.mark.asyncio
    async def test_cost_optimization(self, optimizer_manager):
        """測試成本優化"""
        # 測試模型使用優化
        recommendation = await optimizer_manager.optimize_model_usage(
            model_name="gpt-3.5-turbo", input_tokens=1000, operation_type="embedding"
        )

        # 驗證建議格式
        assert isinstance(recommendation, dict)
        assert (
            "recommended_model" in recommendation or "recommendation" in recommendation
        )

        # 測試多次使用以觸發成本追蹤
        for i in range(5):
            await optimizer_manager.optimize_model_usage(
                model_name="gpt-4",
                input_tokens=500 + i * 100,
                operation_type="inference",
            )

        # 檢查成本統計
        if optimizer_manager.cost_optimizer:
            if (
                hasattr(optimizer_manager.cost_optimizer, "usage_tracker")
                and optimizer_manager.cost_optimizer.usage_tracker
            ):
                stats = optimizer_manager.cost_optimizer.usage_tracker.get_usage_stats(
                    "today"
                )
                assert "total_cost" in stats
                assert "model_breakdown" in stats
            else:
                # 使用成本分析方法作為替代
                stats = optimizer_manager.cost_optimizer.get_cost_analysis("today")
                assert "total_cost" in stats or "cost_breakdown" in stats

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, optimizer_manager):
        """測試效能監控"""
        # 啟動監控
        await optimizer_manager.start()

        # 等待收集一些資料
        await asyncio.sleep(2.5)  # 等待至少2個監控週期

        # 檢查監控資料
        current_metrics = optimizer_manager.performance_monitor.get_current_metrics()
        assert current_metrics is not None
        assert current_metrics.cpu_usage >= 0
        assert current_metrics.memory_usage > 0

        # 檢查歷史資料
        history = optimizer_manager.performance_monitor.get_metrics_history(
            1
        )  # 1分鐘內
        assert len(history) > 0

        # 檢查效能統計
        stats = optimizer_manager.performance_monitor.get_performance_stats(1)
        assert "system" in stats
        assert "cpu" in stats["system"]
        assert "memory" in stats["system"]

    @pytest.mark.asyncio
    async def test_benchmark_execution(self, optimizer_manager):
        """測試基準測試執行"""

        # 定義測試函數
        async def fast_function():
            await asyncio.sleep(0.01)
            return "fast_result"

        async def slow_function():
            await asyncio.sleep(0.05)
            return "slow_result"

        # 執行比較基準測試
        test_configs = [
            {"name": "fast_test", "func": fast_function, "params": {}},
            {"name": "slow_test", "func": slow_function, "params": {}},
        ]

        results = await optimizer_manager.run_performance_benchmark(
            test_configs=test_configs, iterations=5
        )

        # 驗證結果
        assert "fast_test" in results
        assert "slow_test" in results

        fast_result = results["fast_test"]
        slow_result = results["slow_test"]

        # 快速函數應該有更高的吞吐量和更低的延遲
        assert fast_result.throughput > slow_result.throughput
        assert fast_result.latency_ms < slow_result.latency_ms
        assert fast_result.success_rate == 1.0
        assert slow_result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_optimization_report(self, optimizer_manager):
        """測試優化報告生成"""

        # 執行一些優化操作
        async def process_func(x):
            return f"item_{x}"

        await optimizer_manager.optimize_batch_processing(
            items=[1, 2, 3], process_func=process_func
        )

        async def query_func(q):
            return f"result_{q}"

        await optimizer_manager.optimize_query(
            query="test_query", query_func=query_func
        )

        await optimizer_manager.optimize_model_usage(
            model_name="test-model", input_tokens=100
        )

        # 生成報告
        report = optimizer_manager.get_optimization_report(duration_minutes=1)

        # 驗證報告結構
        assert "period_minutes" in report
        assert "timestamp" in report
        assert "summary" in report

        summary = report["summary"]
        assert "batch_optimizations" in summary
        assert summary["batch_optimizations"] > 0

    @pytest.mark.asyncio
    async def test_config_persistence(self, temp_storage, optimization_config):
        """測試配置持久化"""
        config_path = Path(temp_storage) / "test_config.json"

        # 建立管理器並儲存配置
        manager = OptimizerManager(optimization_config)
        manager.save_config(str(config_path))

        # 驗證配置檔案存在
        assert config_path.exists()

        # 載入配置並建立新管理器
        loaded_manager = OptimizerManager.load_config(str(config_path))

        # 驗證配置相同
        assert loaded_manager.config.batch_enabled == optimization_config.batch_enabled
        assert loaded_manager.config.batch_size == optimization_config.batch_size
        assert (
            loaded_manager.config.query_cache_enabled
            == optimization_config.query_cache_enabled
        )
        assert (
            loaded_manager.config.cost_tracking_enabled
            == optimization_config.cost_tracking_enabled
        )

    @pytest.mark.asyncio
    async def test_context_manager(self, optimization_config):
        """測試異步上下文管理器"""
        async with OptimizerManager(optimization_config) as manager:
            # 管理器應該已初始化並啟動
            assert manager._initialized
            assert manager._running

            # 執行一些操作
            status = manager.get_performance_status()
            assert status["initialized"]

        # 退出上下文後應該已停止
        assert not manager._running

    @pytest.mark.asyncio
    async def test_alert_handling(self, optimizer_manager):
        """測試警報處理"""
        # 模擬高記憶體使用警報
        await optimizer_manager._handle_alert("memory_usage", 90.0, 85.0)

        # 檢查批次大小是否已調整
        if optimizer_manager.batch_optimizer:
            # 批次大小應該已減少
            assert (
                optimizer_manager.batch_optimizer.current_batch_size
                <= optimizer_manager.batch_optimizer.default_batch_size
            )

        # 模擬高 CPU 使用警報
        original_workers = (
            optimizer_manager.batch_optimizer.parallel_workers
            if optimizer_manager.batch_optimizer
            else 0
        )
        await optimizer_manager._handle_alert("cpu_usage", 85.0, 80.0)

        # 檢查並行工作者數量是否已調整
        if optimizer_manager.batch_optimizer:
            assert (
                optimizer_manager.batch_optimizer.parallel_workers <= original_workers
            )

    @pytest.mark.asyncio
    async def test_error_handling(self, optimizer_manager):
        """測試錯誤處理"""

        # 測試批次處理中的錯誤
        async def failing_process_func(item):
            if item == 2:
                raise ValueError(f"Processing failed for item {item}")
            return f"processed_{item}"

        # 執行包含錯誤的批次處理
        results = await optimizer_manager.optimize_batch_processing(
            items=[1, 2, 3, 4], process_func=failing_process_func
        )

        # 應該處理錯誤並繼續處理其他項目
        successful_results = [
            r for r in results if r is not None and not isinstance(r, Exception)
        ]
        assert len(successful_results) == 3  # 除了項目2之外都應該成功

        # 測試查詢優化中的錯誤
        async def failing_query_func(query):
            raise RuntimeError("Query failed")

        try:
            await optimizer_manager.optimize_query(
                query="failing_query", query_func=failing_query_func
            )
            assert False, "應該拋出異常"
        except RuntimeError:
            pass  # 預期的異常


if __name__ == "__main__":
    # 執行測試
    pytest.main([__file__, "-v"])
