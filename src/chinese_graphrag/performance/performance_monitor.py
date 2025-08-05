"""
效能監控和基準測試系統

提供系統效能監控、基準測試、效能分析和報告功能
"""

import asyncio
import gc
import json
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil
from loguru import logger

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """效能指標"""

    # 系統資源
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_total: float = 0.0
    disk_usage: float = 0.0

    # GPU 資源（如果可用）
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    gpu_memory_total: float = 0.0

    # 應用程式效能
    response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate: float = 0.0

    # 快取效能
    cache_hit_rate: float = 0.0
    cache_size: int = 0

    # 資料庫效能
    db_query_time_ms: float = 0.0
    db_connection_count: int = 0

    # 自訂指標
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    # 時間戳
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "timestamp": self.timestamp,
            "system": {
                "cpu_usage": self.cpu_usage,
                "memory_usage": self.memory_usage,
                "memory_total": self.memory_total,
                "disk_usage": self.disk_usage,
            },
            "gpu": (
                {
                    "gpu_usage": self.gpu_usage,
                    "gpu_memory_usage": self.gpu_memory_usage,
                    "gpu_memory_total": self.gpu_memory_total,
                }
                if GPU_AVAILABLE
                else None
            ),
            "application": {
                "response_time_ms": self.response_time_ms,
                "throughput_ops_per_sec": self.throughput_ops_per_sec,
                "error_rate": self.error_rate,
            },
            "cache": {
                "cache_hit_rate": self.cache_hit_rate,
                "cache_size": self.cache_size,
            },
            "database": {
                "db_query_time_ms": self.db_query_time_ms,
                "db_connection_count": self.db_connection_count,
            },
            "custom": self.custom_metrics,
        }


@dataclass
class BenchmarkResult:
    """基準測試結果"""

    test_name: str
    start_time: float
    end_time: float
    duration: float

    # 測試參數
    test_params: Dict[str, Any] = field(default_factory=dict)

    # 效能指標
    throughput: float = 0.0
    latency_ms: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_peak_usage: float = 0.0

    # 成功率
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    # 詳細統計
    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    error_details: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """成功率"""
        return (
            self.successful_operations / self.total_operations
            if self.total_operations > 0
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "test_params": self.test_params,
            "performance": {
                "throughput": self.throughput,
                "latency_ms": self.latency_ms,
                "memory_peak_mb": self.memory_peak_mb,
                "cpu_peak_usage": self.cpu_peak_usage,
            },
            "reliability": {
                "total_operations": self.total_operations,
                "successful_operations": self.successful_operations,
                "failed_operations": self.failed_operations,
                "success_rate": self.success_rate,
            },
            "statistics": {
                "latency_percentiles": self.latency_percentiles,
                "error_count": len(self.error_details),
            },
        }


class PerformanceMonitor:
    """效能監控器"""

    def __init__(
        self,
        collection_interval: float = 5.0,
        history_size: int = 1000,
        storage_path: Optional[str] = None,
    ):
        """初始化效能監控器

        Args:
            collection_interval: 資料收集間隔（秒）
            history_size: 歷史資料保留數量
            storage_path: 資料儲存路徑
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.storage_path = (
            Path(storage_path) if storage_path else Path("logs/performance")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 效能資料
        self._metrics_history: deque = deque(maxlen=history_size)
        self._metrics_lock = threading.RLock()

        # 監控狀態
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

        # 自訂指標收集器
        self._custom_collectors: Dict[str, Callable] = {}

        # 警報設定
        self._alert_thresholds: Dict[str, Tuple[float, Callable]] = {}

        # 統計資料
        self._stats_cache: Dict[str, Any] = {}
        self._last_stats_update = 0.0

        logger.info(f"效能監控器初始化完成，收集間隔: {collection_interval}s")

    def start_monitoring(self):
        """開始效能監控"""
        if self._monitoring:
            logger.warning("效能監控已在執行中")
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("效能監控已啟動")

    def stop_monitoring(self):
        """停止效能監控"""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

        logger.info("效能監控已停止")

    async def _monitoring_loop(self):
        """監控循環"""
        try:
            while self._monitoring:
                metrics = await self._collect_metrics()

                with self._metrics_lock:
                    self._metrics_history.append(metrics)

                # 檢查警報
                await self._check_alerts(metrics)

                # 儲存資料
                await self._save_metrics(metrics)

                await asyncio.sleep(self.collection_interval)

        except asyncio.CancelledError:
            logger.info("效能監控循環已取消")
        except Exception as e:
            logger.error(f"效能監控循環錯誤: {e}")

    async def _collect_metrics(self) -> PerformanceMetrics:
        """收集效能指標"""
        metrics = PerformanceMetrics()

        try:
            # 系統資源
            metrics.cpu_usage = psutil.cpu_percent(interval=0.1)

            memory = psutil.virtual_memory()
            metrics.memory_usage = memory.used / (1024 * 1024)  # MB
            metrics.memory_total = memory.total / (1024 * 1024)  # MB

            disk = psutil.disk_usage("/")
            metrics.disk_usage = (disk.used / disk.total) * 100

            # GPU 資源
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # 使用第一個 GPU
                        metrics.gpu_usage = gpu.load * 100
                        metrics.gpu_memory_usage = gpu.memoryUsed
                        metrics.gpu_memory_total = gpu.memoryTotal
                except Exception as e:
                    logger.debug(f"GPU 資訊收集失敗: {e}")

            # 收集自訂指標
            for name, collector in self._custom_collectors.items():
                try:
                    value = (
                        await collector()
                        if asyncio.iscoroutinefunction(collector)
                        else collector()
                    )
                    metrics.custom_metrics[name] = value
                except Exception as e:
                    logger.warning(f"自訂指標 {name} 收集失敗: {e}")

        except Exception as e:
            logger.error(f"效能指標收集失敗: {e}")

        return metrics

    async def _check_alerts(self, metrics: PerformanceMetrics):
        """檢查警報條件"""
        for metric_name, (threshold, callback) in self._alert_thresholds.items():
            try:
                value = getattr(metrics, metric_name, None)
                if value is None and metric_name in metrics.custom_metrics:
                    value = metrics.custom_metrics[metric_name]

                if value is not None and value > threshold:
                    await callback(metric_name, value, threshold)

            except Exception as e:
                logger.warning(f"警報檢查失敗 {metric_name}: {e}")

    async def _save_metrics(self, metrics: PerformanceMetrics):
        """儲存效能指標"""
        try:
            date_str = datetime.fromtimestamp(metrics.timestamp).strftime("%Y-%m-%d")
            log_file = self.storage_path / f"performance_{date_str}.jsonl"

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"效能指標儲存失敗: {e}")

    def register_custom_collector(self, name: str, collector: Callable):
        """註冊自訂指標收集器

        Args:
            name: 指標名稱
            collector: 收集器函數
        """
        self._custom_collectors[name] = collector
        logger.info(f"註冊自訂指標收集器: {name}")

    def set_alert_threshold(
        self, metric_name: str, threshold: float, callback: Callable
    ):
        """設定警報閾值

        Args:
            metric_name: 指標名稱
            threshold: 警報閾值
            callback: 警報回調函數
        """
        self._alert_thresholds[metric_name] = (threshold, callback)
        logger.info(f"設定警報閾值: {metric_name} > {threshold}")

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """取得當前效能指標"""
        with self._metrics_lock:
            return self._metrics_history[-1] if self._metrics_history else None

    def get_metrics_history(
        self, duration_minutes: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """取得效能指標歷史

        Args:
            duration_minutes: 歷史時間範圍（分鐘）

        Returns:
            效能指標列表
        """
        with self._metrics_lock:
            if duration_minutes is None:
                return list(self._metrics_history)

            cutoff_time = time.time() - (duration_minutes * 60)
            return [m for m in self._metrics_history if m.timestamp >= cutoff_time]

    def get_performance_stats(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """取得效能統計資料

        Args:
            duration_minutes: 統計時間範圍（分鐘）

        Returns:
            效能統計資料
        """
        # 檢查快取
        cache_key = f"stats_{duration_minutes}"
        if (
            cache_key in self._stats_cache
            and time.time() - self._last_stats_update < 30
        ):  # 30秒快取
            return self._stats_cache[cache_key]

        history = self.get_metrics_history(duration_minutes)
        if not history:
            return {}

        # 計算統計資料
        stats = {
            "period_minutes": duration_minutes,
            "sample_count": len(history),
            "system": self._calculate_system_stats(history),
            "trends": self._calculate_trends(history),
        }

        # 更新快取
        self._stats_cache[cache_key] = stats
        self._last_stats_update = time.time()

        return stats

    def _calculate_system_stats(
        self, history: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """計算系統統計資料"""
        if not history:
            return {}

        cpu_values = [m.cpu_usage for m in history]
        memory_values = [m.memory_usage for m in history]

        return {
            "cpu": {
                "avg": statistics.mean(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "p95": (
                    statistics.quantiles(cpu_values, n=20)[18]
                    if len(cpu_values) > 1
                    else cpu_values[0]
                ),
            },
            "memory": {
                "avg_mb": statistics.mean(memory_values),
                "min_mb": min(memory_values),
                "max_mb": max(memory_values),
                "p95_mb": (
                    statistics.quantiles(memory_values, n=20)[18]
                    if len(memory_values) > 1
                    else memory_values[0]
                ),
            },
        }

    def _calculate_trends(self, history: List[PerformanceMetrics]) -> Dict[str, str]:
        """計算效能趨勢"""
        if len(history) < 10:
            return {"status": "insufficient_data"}

        # 分析最近和之前的資料
        recent_data = history[-5:]
        previous_data = history[-10:-5]

        recent_cpu = statistics.mean([m.cpu_usage for m in recent_data])
        previous_cpu = statistics.mean([m.cpu_usage for m in previous_data])

        recent_memory = statistics.mean([m.memory_usage for m in recent_data])
        previous_memory = statistics.mean([m.memory_usage for m in previous_data])

        # 判斷趨勢
        cpu_trend = (
            "increasing"
            if recent_cpu > previous_cpu * 1.1
            else "decreasing" if recent_cpu < previous_cpu * 0.9 else "stable"
        )
        memory_trend = (
            "increasing"
            if recent_memory > previous_memory * 1.1
            else "decreasing" if recent_memory < previous_memory * 0.9 else "stable"
        )

        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "overall_status": (
                "degrading"
                if cpu_trend == "increasing" and memory_trend == "increasing"
                else "stable"
            ),
        }


class BenchmarkRunner:
    """基準測試執行器"""

    def __init__(self, monitor: Optional[PerformanceMonitor] = None):
        """初始化基準測試執行器

        Args:
            monitor: 效能監控器
        """
        self.monitor = monitor
        self._benchmark_results: List[BenchmarkResult] = []
        self._running_benchmarks: Dict[str, BenchmarkResult] = {}

        logger.info("基準測試執行器初始化完成")

    async def run_benchmark(
        self,
        test_name: str,
        test_func: Callable,
        test_params: Dict[str, Any],
        iterations: int = 1,
        warmup_iterations: int = 0,
    ) -> BenchmarkResult:
        """執行基準測試

        Args:
            test_name: 測試名稱
            test_func: 測試函數
            test_params: 測試參數
            iterations: 測試迭代次數
            warmup_iterations: 預熱迭代次數

        Returns:
            基準測試結果
        """
        logger.info(f"開始基準測試: {test_name}")

        # 建立測試結果
        result = BenchmarkResult(
            test_name=test_name,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            test_params=test_params,
        )

        self._running_benchmarks[test_name] = result

        try:
            # 預熱
            if warmup_iterations > 0:
                logger.info(f"執行預熱迭代: {warmup_iterations} 次")
                for _ in range(warmup_iterations):
                    await self._run_single_iteration(test_func, test_params)

            # 正式測試
            latencies = []
            memory_usage = []
            cpu_usage = []

            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            for i in range(iterations):
                iteration_start = time.time()

                try:
                    # 執行測試
                    await self._run_single_iteration(test_func, test_params)

                    iteration_end = time.time()
                    latency = (iteration_end - iteration_start) * 1000  # 轉換為毫秒
                    latencies.append(latency)

                    result.successful_operations += 1

                    # 記錄資源使用
                    current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                    memory_usage.append(current_memory - initial_memory)
                    cpu_usage.append(psutil.cpu_percent(interval=0.1))

                except Exception as e:
                    result.failed_operations += 1
                    result.error_details.append(str(e))
                    logger.warning(f"測試迭代 {i+1} 失敗: {e}")

                result.total_operations += 1

                # 進度報告
                if (i + 1) % max(1, iterations // 10) == 0:
                    progress = (i + 1) / iterations * 100
                    logger.info(f"測試進度: {progress:.1f}% ({i+1}/{iterations})")

            # 計算統計資料
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time

            if latencies:
                result.latency_ms = statistics.mean(latencies)
                result.latency_percentiles = {
                    "p50": statistics.median(latencies),
                    "p95": (
                        statistics.quantiles(latencies, n=20)[18]
                        if len(latencies) > 1
                        else latencies[0]
                    ),
                    "p99": (
                        statistics.quantiles(latencies, n=100)[98]
                        if len(latencies) > 1
                        else latencies[0]
                    ),
                }

            if memory_usage:
                result.memory_peak_mb = max(memory_usage)

            if cpu_usage:
                result.cpu_peak_usage = max(cpu_usage)

            # 計算吞吐量
            if result.duration > 0:
                result.throughput = result.successful_operations / result.duration

            # 儲存結果
            self._benchmark_results.append(result)

            logger.info(
                f"基準測試完成: {test_name}, "
                f"成功率: {result.success_rate:.1%}, "
                f"平均延遲: {result.latency_ms:.2f}ms, "
                f"吞吐量: {result.throughput:.2f} ops/s"
            )

            return result

        finally:
            del self._running_benchmarks[test_name]

    async def _run_single_iteration(
        self, test_func: Callable, test_params: Dict[str, Any]
    ):
        """執行單次測試迭代"""
        if asyncio.iscoroutinefunction(test_func):
            return await test_func(**test_params)
        else:
            return test_func(**test_params)

    async def run_comparative_benchmark(
        self, test_configs: List[Dict[str, Any]], iterations: int = 10
    ) -> Dict[str, BenchmarkResult]:
        """執行比較基準測試

        Args:
            test_configs: 測試配置列表
            iterations: 每個測試的迭代次數

        Returns:
            比較測試結果
        """
        results = {}

        for config in test_configs:
            test_name = config["name"]
            test_func = config["func"]
            test_params = config.get("params", {})

            result = await self.run_benchmark(
                test_name=test_name,
                test_func=test_func,
                test_params=test_params,
                iterations=iterations,
            )

            results[test_name] = result

        # 生成比較報告
        self._generate_comparison_report(results)

        return results

    def _generate_comparison_report(self, results: Dict[str, BenchmarkResult]):
        """生成比較報告"""
        logger.info("基準測試比較報告:")
        logger.info("=" * 60)

        # 按吞吐量排序
        sorted_results = sorted(
            results.items(), key=lambda x: x[1].throughput, reverse=True
        )

        for i, (name, result) in enumerate(sorted_results, 1):
            logger.info(f"{i}. {name}:")
            logger.info(f"   吞吐量: {result.throughput:.2f} ops/s")
            logger.info(f"   平均延遲: {result.latency_ms:.2f}ms")
            logger.info(f"   成功率: {result.success_rate:.1%}")
            logger.info(f"   記憶體峰值: {result.memory_peak_mb:.1f}MB")

    def get_benchmark_history(
        self, test_name: Optional[str] = None
    ) -> List[BenchmarkResult]:
        """取得基準測試歷史

        Args:
            test_name: 特定測試名稱

        Returns:
            基準測試結果列表
        """
        if test_name:
            return [r for r in self._benchmark_results if r.test_name == test_name]
        return self._benchmark_results.copy()

    def export_results(self, output_path: str, format: str = "json"):
        """匯出測試結果

        Args:
            output_path: 輸出路徑
            format: 輸出格式 ("json", "csv")
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            results_data = [result.to_dict() for result in self._benchmark_results]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)

        elif format == "csv":
            import csv

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                if self._benchmark_results:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "test_name",
                            "duration",
                            "throughput",
                            "latency_ms",
                            "success_rate",
                            "memory_peak_mb",
                            "cpu_peak_usage",
                        ],
                    )
                    writer.writeheader()

                    for result in self._benchmark_results:
                        writer.writerow(
                            {
                                "test_name": result.test_name,
                                "duration": result.duration,
                                "throughput": result.throughput,
                                "latency_ms": result.latency_ms,
                                "success_rate": result.success_rate,
                                "memory_peak_mb": result.memory_peak_mb,
                                "cpu_peak_usage": result.cpu_peak_usage,
                            }
                        )

        logger.info(f"測試結果已匯出: {output_file}")
