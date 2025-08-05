"""
批次處理優化器

提供智慧批次處理、並行處理和資源管理功能
"""

import asyncio
import gc
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil
from loguru import logger

from ..embeddings.gpu_acceleration import get_device_manager, get_memory_optimizer


@dataclass
class BatchProcessingConfig:
    """批次處理配置"""

    # 基本批次設定
    initial_batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 512

    # 並行處理設定
    max_workers: int = 4
    use_process_pool: bool = False
    enable_async: bool = True

    # 記憶體管理
    memory_threshold: float = 0.8  # 記憶體使用閾值
    memory_limit_mb: Optional[int] = None
    enable_gc: bool = True
    gc_frequency: int = 10  # 每處理多少批次執行一次垃圾回收

    # 效能調整
    adaptive_batch_size: bool = True
    performance_target_ms: float = 1000.0  # 目標處理時間（毫秒）
    batch_size_adjustment_factor: float = 0.1

    # 錯誤處理
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_fallback: bool = True

    # 監控和日誌
    enable_progress_tracking: bool = True
    log_batch_stats: bool = True
    stats_collection_interval: int = 100


@dataclass
class BatchStats:
    """批次處理統計資料"""

    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0

    total_processing_time: float = 0.0
    avg_batch_processing_time: float = 0.0
    min_batch_processing_time: float = float("inf")
    max_batch_processing_time: float = 0.0

    memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0

    throughput_items_per_second: float = 0.0

    batch_size_history: List[int] = field(default_factory=list)
    processing_time_history: List[float] = field(default_factory=list)

    def update_batch_result(
        self, batch_size: int, processing_time: float, success: bool
    ):
        """更新批次結果統計"""
        self.total_batches += 1

        if success:
            self.successful_batches += 1
            self.processed_items += batch_size
        else:
            self.failed_batches += 1
            self.failed_items += batch_size

        self.total_processing_time += processing_time
        self.batch_size_history.append(batch_size)
        self.processing_time_history.append(processing_time)

        # 更新統計
        self.avg_batch_processing_time = self.total_processing_time / self.total_batches
        self.min_batch_processing_time = min(
            self.min_batch_processing_time, processing_time
        )
        self.max_batch_processing_time = max(
            self.max_batch_processing_time, processing_time
        )

        # 計算吞吐量
        if self.total_processing_time > 0:
            self.throughput_items_per_second = (
                self.processed_items / self.total_processing_time
            )

    def get_summary(self) -> Dict[str, Any]:
        """取得統計摘要"""
        success_rate = (
            self.successful_batches / self.total_batches
            if self.total_batches > 0
            else 0
        )

        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "success_rate": success_rate,
            "total_batches": self.total_batches,
            "avg_batch_processing_time": self.avg_batch_processing_time,
            "throughput_items_per_second": self.throughput_items_per_second,
            "peak_memory_usage_mb": self.peak_memory_usage_mb,
            "avg_batch_size": (
                sum(self.batch_size_history) / len(self.batch_size_history)
                if self.batch_size_history
                else 0
            ),
        }


class BatchOptimizer:
    """批次處理優化器

    提供智慧批次處理、動態批次大小調整、並行處理和資源管理功能
    """

    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """初始化批次優化器

        Args:
            config: 批次處理配置
        """
        self.config = config or BatchProcessingConfig()
        self.stats = BatchStats()

        # 初始化資源管理器
        self.memory_optimizer = get_memory_optimizer()
        self.device_manager = get_device_manager()

        # 批次大小管理
        self.default_batch_size = self.config.initial_batch_size
        self.current_batch_size = self.config.initial_batch_size
        self._batch_size_lock = threading.Lock()

        # 並行工作者管理
        self.parallel_workers = self.config.max_workers

        # 執行器管理
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._process_executor: Optional[ProcessPoolExecutor] = None

        # 效能監控
        self._performance_history: List[Tuple[int, float]] = (
            []
        )  # (batch_size, processing_time)
        self._last_gc_batch = 0

        logger.info(f"批次優化器初始化完成，初始批次大小: {self.current_batch_size}")

    def __enter__(self):
        """進入上下文管理器"""
        self._initialize_executors()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        self._cleanup_executors()

    def _initialize_executors(self):
        """初始化執行器"""
        if self.config.use_process_pool:
            self._process_executor = ProcessPoolExecutor(
                max_workers=self.config.max_workers
            )
            logger.info(f"初始化程序池執行器，工作程序數: {self.config.max_workers}")
        else:
            self._thread_executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )
            logger.info(
                f"初始化執行緒池執行器，工作執行緒數: {self.config.max_workers}"
            )

    def _cleanup_executors(self):
        """清理執行器"""
        if self._thread_executor:
            self._thread_executor.shutdown(wait=True)
            self._thread_executor = None

        if self._process_executor:
            self._process_executor.shutdown(wait=True)
            self._process_executor = None

        logger.info("執行器清理完成")

    def calculate_optimal_batch_size(
        self,
        estimated_memory_per_item: int = 10,
        target_processing_time: Optional[float] = None,
    ) -> int:
        """計算最佳批次大小

        Args:
            estimated_memory_per_item: 每個項目的估計記憶體使用量（MB）
            target_processing_time: 目標處理時間（秒）

        Returns:
            最佳批次大小
        """
        with self._batch_size_lock:
            # 基於記憶體限制計算批次大小
            memory_stats = self.memory_optimizer.get_memory_stats()
            available_memory_mb = (
                memory_stats.system_total * (1 - self.config.memory_threshold)
                - memory_stats.process_used
            )

            if self.config.memory_limit_mb:
                available_memory_mb = min(
                    available_memory_mb, self.config.memory_limit_mb
                )

            memory_based_batch_size = max(
                self.config.min_batch_size,
                min(
                    self.config.max_batch_size,
                    int(available_memory_mb / estimated_memory_per_item),
                ),
            )

            # 基於效能歷史調整批次大小
            if self.config.adaptive_batch_size and self._performance_history:
                performance_adjusted_size = self._adjust_batch_size_by_performance(
                    memory_based_batch_size,
                    target_processing_time
                    or self.config.performance_target_ms / 1000.0,
                )
            else:
                performance_adjusted_size = memory_based_batch_size

            optimal_size = max(
                self.config.min_batch_size,
                min(self.config.max_batch_size, performance_adjusted_size),
            )

            logger.debug(
                f"計算最佳批次大小: {optimal_size} "
                f"(記憶體限制: {memory_based_batch_size}, "
                f"效能調整: {performance_adjusted_size})"
            )

            return optimal_size

    def _adjust_batch_size_by_performance(
        self, base_batch_size: int, target_time: float
    ) -> int:
        """基於效能歷史調整批次大小"""
        if len(self._performance_history) < 3:
            return base_batch_size

        # 分析最近的效能資料
        recent_history = self._performance_history[-10:]  # 最近10次的資料

        # 計算平均處理時間和批次大小的關係
        avg_time_per_item = sum(time / size for size, time in recent_history) / len(
            recent_history
        )

        # 基於目標時間計算理想批次大小
        ideal_batch_size = int(target_time / avg_time_per_item)

        # 使用調整因子進行漸進式調整
        adjustment = int(
            (ideal_batch_size - base_batch_size)
            * self.config.batch_size_adjustment_factor
        )
        adjusted_size = base_batch_size + adjustment

        return max(
            self.config.min_batch_size, min(self.config.max_batch_size, adjusted_size)
        )

    async def process_in_batches_async(
        self,
        items: List[Any],
        process_func: Callable,
        estimated_memory_per_item: int = 10,
        show_progress: bool = True,
    ) -> List[Any]:
        """異步批次處理

        Args:
            items: 要處理的項目列表
            process_func: 處理函數
            estimated_memory_per_item: 每個項目的估計記憶體使用量（MB）
            show_progress: 是否顯示進度

        Returns:
            處理結果列表
        """
        if not items:
            return []

        self.stats.total_items = len(items)
        results = []
        processed_items = 0

        logger.info(f"開始異步批次處理 {len(items)} 個項目")

        while processed_items < len(items):
            # 計算當前批次大小
            batch_size = self.calculate_optimal_batch_size(estimated_memory_per_item)
            batch_start = processed_items
            batch_end = min(processed_items + batch_size, len(items))
            current_batch = items[batch_start:batch_end]

            start_time = time.time()
            success = False

            try:
                # 記憶體監控
                with self.memory_optimizer.memory_monitor(
                    f"批次處理 {batch_start}-{batch_end}"
                ):
                    batch_result = await process_func(current_batch)
                    results.extend(batch_result)
                    success = True
                    processed_items += len(current_batch)

                processing_time = time.time() - start_time
                self.stats.update_batch_result(
                    len(current_batch), processing_time, success
                )
                self._performance_history.append((len(current_batch), processing_time))

                if show_progress:
                    progress = processed_items / len(items) * 100
                    logger.info(
                        f"批次處理進度: {progress:.1f}% "
                        f"({processed_items}/{len(items)})"
                    )

                # 垃圾回收
                if (
                    self.config.enable_gc
                    and self.stats.total_batches - self._last_gc_batch
                    >= self.config.gc_frequency
                ):
                    gc.collect()
                    self._last_gc_batch = self.stats.total_batches

            except Exception as e:
                processing_time = time.time() - start_time
                self.stats.update_batch_result(
                    len(current_batch), processing_time, success
                )

                logger.error(f"批次處理失敗: {e}")

                # 錯誤處理和重試邏輯
                if (
                    self.config.enable_fallback
                    and batch_size > self.config.min_batch_size
                ):
                    logger.info(
                        f"嘗試減小批次大小重試: {batch_size} -> {batch_size // 2}"
                    )
                    self.current_batch_size = max(
                        self.config.min_batch_size, batch_size // 2
                    )
                    continue
                else:
                    logger.error(f"跳過失敗的批次: {batch_start}-{batch_end}")
                    processed_items += len(current_batch)

            # 更新記憶體統計
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            self.stats.memory_usage_mb = current_memory
            self.stats.peak_memory_usage_mb = max(
                self.stats.peak_memory_usage_mb, current_memory
            )

        logger.info(f"異步批次處理完成，處理了 {processed_items} 個項目")
        return results

    def process_in_batches_parallel(
        self,
        items: List[Any],
        process_func: Callable,
        estimated_memory_per_item: int = 10,
        show_progress: bool = True,
    ) -> List[Any]:
        """並行批次處理

        Args:
            items: 要處理的項目列表
            process_func: 處理函數
            estimated_memory_per_item: 每個項目的估計記憶體使用量（MB）
            show_progress: 是否顯示進度

        Returns:
            處理結果列表
        """
        if not items:
            return []

        self.stats.total_items = len(items)

        # 計算批次大小和批次數量
        batch_size = self.calculate_optimal_batch_size(estimated_memory_per_item)
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        logger.info(f"開始並行批次處理 {len(items)} 個項目，分為 {len(batches)} 個批次")

        results = []
        executor = (
            self._process_executor
            if self.config.use_process_pool
            else self._thread_executor
        )

        if not executor:
            raise RuntimeError(
                "執行器未初始化，請使用 with 語句或手動調用 _initialize_executors()"
            )

        # 提交所有批次任務
        future_to_batch = {
            executor.submit(process_func, batch): (i, batch)
            for i, batch in enumerate(batches)
        }

        processed_batches = 0

        # 收集結果
        for future in as_completed(future_to_batch):
            batch_index, batch = future_to_batch[future]
            start_time = time.time()

            try:
                batch_result = future.result()
                results.extend(batch_result)
                success = True

                processing_time = time.time() - start_time
                self.stats.update_batch_result(len(batch), processing_time, success)

                processed_batches += 1

                if show_progress:
                    progress = processed_batches / len(batches) * 100
                    logger.info(
                        f"並行處理進度: {progress:.1f}% ({processed_batches}/{len(batches)})"
                    )

            except Exception as e:
                processing_time = time.time() - start_time
                self.stats.update_batch_result(len(batch), processing_time, False)
                logger.error(f"批次 {batch_index} 處理失敗: {e}")

        logger.info(f"並行批次處理完成，處理了 {len(results)} 個結果")
        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """取得效能統計資料"""
        stats_summary = self.stats.get_summary()

        # 添加額外的效能指標
        if self._performance_history:
            recent_history = self._performance_history[-20:]  # 最近20次
            avg_processing_time = sum(time for _, time in recent_history) / len(
                recent_history
            )
            avg_batch_size = sum(size for size, _ in recent_history) / len(
                recent_history
            )

            stats_summary.update(
                {
                    "recent_avg_processing_time": avg_processing_time,
                    "recent_avg_batch_size": avg_batch_size,
                    "performance_trend": self._calculate_performance_trend(),
                }
            )

        return stats_summary

    def _calculate_performance_trend(self) -> str:
        """計算效能趨勢"""
        if len(self._performance_history) < 10:
            return "insufficient_data"

        # 比較最近5次和之前5次的平均處理時間
        recent_times = [time for _, time in self._performance_history[-5:]]
        previous_times = [time for _, time in self._performance_history[-10:-5]]

        recent_avg = sum(recent_times) / len(recent_times)
        previous_avg = sum(previous_times) / len(previous_times)

        if recent_avg < previous_avg * 0.95:
            return "improving"
        elif recent_avg > previous_avg * 1.05:
            return "degrading"
        else:
            return "stable"

    def reset_stats(self):
        """重設統計資料"""
        self.stats = BatchStats()
        self._performance_history.clear()
        self._last_gc_batch = 0
        logger.info("批次處理統計資料已重設")

    def optimize_for_memory(self):
        """針對記憶體使用進行優化"""
        # 強制垃圾回收
        gc.collect()

        # 調整批次大小
        memory_stats = self.memory_optimizer.get_memory_stats()
        if memory_stats.system_usage_ratio > self.config.memory_threshold:
            with self._batch_size_lock:
                self.current_batch_size = max(
                    self.config.min_batch_size, self.current_batch_size // 2
                )
            logger.info(f"記憶體優化：調整批次大小至 {self.current_batch_size}")

        # 清理效能歷史（保留最近的資料）
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-50:]

    def get_recommended_config(
        self, workload_type: str = "balanced"
    ) -> BatchProcessingConfig:
        """取得推薦的配置

        Args:
            workload_type: 工作負載類型 ("memory_intensive", "cpu_intensive", "balanced")

        Returns:
            推薦的批次處理配置
        """
        base_config = BatchProcessingConfig()

        if workload_type == "memory_intensive":
            base_config.initial_batch_size = 16
            base_config.max_batch_size = 128
            base_config.memory_threshold = 0.7
            base_config.gc_frequency = 5
            base_config.enable_gc = True

        elif workload_type == "cpu_intensive":
            base_config.initial_batch_size = 64
            base_config.max_batch_size = 512
            base_config.max_workers = min(8, psutil.cpu_count())
            base_config.use_process_pool = True

        else:  # balanced
            base_config.initial_batch_size = 32
            base_config.max_batch_size = 256
            base_config.max_workers = min(4, psutil.cpu_count())
            base_config.memory_threshold = 0.8

        return base_config
