"""
Embedding 模型使用量監控系統

提供詳細的使用量統計、效能監控和成本追蹤功能
支援多模型監控、實時警報和歷史資料分析
"""

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from loguru import logger


class AlertLevel(Enum):
    """警報等級"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class UsageRecord:
    """使用量記錄"""

    timestamp: float
    model_name: str
    operation: str  # 'embed_texts', 'embed_single_text', 'compute_similarity'
    input_count: int  # 輸入項目數量
    input_tokens: int  # 輸入 token 數量（估算）
    output_dimensions: int  # 輸出向量維度
    processing_time: float  # 處理時間（秒）
    memory_used: float  # 記憶體使用量（MB）
    device: str  # 使用的裝置
    success: bool  # 是否成功
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def throughput(self) -> float:
        """吞吐量（項目/秒）"""
        return (
            self.input_count / self.processing_time if self.processing_time > 0 else 0.0
        )

    @property
    def tokens_per_second(self) -> float:
        """Token 處理速度（tokens/秒）"""
        return (
            self.input_tokens / self.processing_time
            if self.processing_time > 0
            else 0.0
        )


@dataclass
class ModelStats:
    """模型統計資訊"""

    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_count: int = 0
    total_input_tokens: int = 0
    total_processing_time: float = 0.0
    total_memory_used: float = 0.0
    first_used: Optional[float] = None
    last_used: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """成功率"""
        return (
            self.successful_requests / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

    @property
    def average_processing_time(self) -> float:
        """平均處理時間"""
        return (
            self.total_processing_time / self.successful_requests
            if self.successful_requests > 0
            else 0.0
        )

    @property
    def average_throughput(self) -> float:
        """平均吞吐量"""
        return (
            self.total_input_count / self.total_processing_time
            if self.total_processing_time > 0
            else 0.0
        )

    @property
    def average_memory_per_request(self) -> float:
        """平均每請求記憶體使用量"""
        return (
            self.total_memory_used / self.successful_requests
            if self.successful_requests > 0
            else 0.0
        )


@dataclass
class Alert:
    """警報資訊"""

    timestamp: float
    level: AlertLevel
    model_name: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


class UsageMonitor:
    """使用量監控器

    負責收集、儲存和分析 embedding 模型的使用量資料
    """

    def __init__(
        self,
        storage_dir: Union[str, Path] = "./logs/embedding_usage",
        max_records_in_memory: int = 10000,
        save_interval: int = 300,  # 5 分鐘
        enable_alerts: bool = True,
    ):
        """初始化使用量監控器

        Args:
            storage_dir: 資料儲存目錄
            max_records_in_memory: 記憶體中最大記錄數
            save_interval: 儲存間隔（秒）
            enable_alerts: 是否啟用警報
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_records_in_memory = max_records_in_memory
        self.save_interval = save_interval
        self.enable_alerts = enable_alerts

        # 資料儲存
        self.usage_records: deque[UsageRecord] = deque(maxlen=max_records_in_memory)
        self.model_stats: Dict[str, ModelStats] = {}
        self.alerts: List[Alert] = []

        # 實時統計
        self.hourly_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.daily_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: defaultdict(int)
        )

        # 警報規則
        self.alert_rules: Dict[str, Dict[str, Any]] = {
            "high_error_rate": {
                "threshold": 0.1,  # 10% 錯誤率
                "window_minutes": 15,
                "enabled": True,
            },
            "slow_processing": {
                "threshold": 30.0,  # 30 秒
                "window_minutes": 5,
                "enabled": True,
            },
            "high_memory_usage": {
                "threshold": 1000.0,  # 1GB
                "window_minutes": 10,
                "enabled": True,
            },
            "request_spike": {
                "threshold_multiplier": 3.0,  # 3 倍於平均值
                "window_minutes": 5,
                "enabled": True,
            },
        }

        # 執行緒鎖
        self._lock = threading.RLock()

        # 自動儲存任務
        self._save_task: Optional[asyncio.Task] = None
        self._stop_save_task = False

        # 載入歷史資料
        asyncio.create_task(self._load_historical_data())

        # 啟動自動儲存
        self._start_auto_save()

        logger.info(f"初始化使用量監控器，儲存目錄: {self.storage_dir}")

    def record_usage(
        self,
        model_name: str,
        operation: str,
        input_count: int,
        processing_time: float,
        memory_used: float = 0.0,
        device: str = "unknown",
        success: bool = True,
        error_message: Optional[str] = None,
        **metadata,
    ) -> None:
        """記錄使用量

        Args:
            model_name: 模型名稱
            operation: 操作類型
            input_count: 輸入項目數量
            processing_time: 處理時間
            memory_used: 記憶體使用量
            device: 使用的裝置
            success: 是否成功
            error_message: 錯誤訊息
            **metadata: 額外元數據
        """
        with self._lock:
            timestamp = time.time()

            # 估算 token 數量（簡化估算）
            estimated_tokens = input_count * 50  # 假設平均每個項目 50 tokens

            # 建立使用量記錄
            record = UsageRecord(
                timestamp=timestamp,
                model_name=model_name,
                operation=operation,
                input_count=input_count,
                input_tokens=estimated_tokens,
                output_dimensions=metadata.get("output_dimensions", 0),
                processing_time=processing_time,
                memory_used=memory_used,
                device=device,
                success=success,
                error_message=error_message,
                metadata=metadata,
            )

            # 添加到記錄中
            self.usage_records.append(record)

            # 更新模型統計
            self._update_model_stats(record)

            # 更新實時統計
            self._update_realtime_stats(record)

            # 檢查警報
            if self.enable_alerts:
                self._check_alerts(record)

            logger.debug(
                f"記錄使用量: {model_name} - {operation}, "
                f"項目數: {input_count}, 時間: {processing_time:.2f}s"
            )

    def _update_model_stats(self, record: UsageRecord) -> None:
        """更新模型統計資訊"""
        model_name = record.model_name

        if model_name not in self.model_stats:
            self.model_stats[model_name] = ModelStats(
                model_name=model_name, first_used=record.timestamp
            )

        stats = self.model_stats[model_name]
        stats.total_requests += 1
        stats.total_input_count += record.input_count
        stats.total_input_tokens += record.input_tokens
        stats.last_used = record.timestamp

        if record.success:
            stats.successful_requests += 1
            stats.total_processing_time += record.processing_time
            stats.total_memory_used += record.memory_used
        else:
            stats.failed_requests += 1

    def _update_realtime_stats(self, record: UsageRecord) -> None:
        """更新實時統計"""
        # 小時統計
        hour_key = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d-%H")
        hour_stats = self.hourly_stats[hour_key]
        hour_stats["total_requests"] += 1
        hour_stats["total_input_count"] += record.input_count
        hour_stats["total_processing_time"] += record.processing_time

        if record.success:
            hour_stats["successful_requests"] += 1
        else:
            hour_stats["failed_requests"] += 1

        # 日統計
        day_key = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d")
        day_stats = self.daily_stats[day_key]
        day_stats["total_requests"] += 1
        day_stats["total_input_count"] += record.input_count
        day_stats["total_processing_time"] += record.processing_time

        if record.success:
            day_stats["successful_requests"] += 1
        else:
            day_stats["failed_requests"] += 1

    def _check_alerts(self, record: UsageRecord) -> None:
        """檢查警報條件"""
        current_time = time.time()

        # 高錯誤率警報
        if self.alert_rules["high_error_rate"]["enabled"] and not record.success:
            self._check_error_rate_alert(record, current_time)

        # 慢處理警報
        if (
            self.alert_rules["slow_processing"]["enabled"]
            and record.success
            and record.processing_time
            > self.alert_rules["slow_processing"]["threshold"]
        ):
            self._create_alert(
                AlertLevel.WARNING,
                record.model_name,
                f"處理時間過長: {record.processing_time:.2f}s",
                {
                    "processing_time": record.processing_time,
                    "operation": record.operation,
                },
            )

        # 高記憶體使用警報
        if (
            self.alert_rules["high_memory_usage"]["enabled"]
            and record.memory_used > self.alert_rules["high_memory_usage"]["threshold"]
        ):
            self._create_alert(
                AlertLevel.WARNING,
                record.model_name,
                f"記憶體使用量過高: {record.memory_used:.1f}MB",
                {"memory_used": record.memory_used, "operation": record.operation},
            )

    def _check_error_rate_alert(self, record: UsageRecord, current_time: float) -> None:
        """檢查錯誤率警報"""
        window_seconds = self.alert_rules["high_error_rate"]["window_minutes"] * 60
        threshold = self.alert_rules["high_error_rate"]["threshold"]

        # 計算時間窗口內的錯誤率
        window_start = current_time - window_seconds
        recent_records = [
            r
            for r in self.usage_records
            if r.timestamp >= window_start and r.model_name == record.model_name
        ]

        if len(recent_records) >= 10:  # 至少需要 10 個樣本
            error_count = sum(1 for r in recent_records if not r.success)
            error_rate = error_count / len(recent_records)

            if error_rate > threshold:
                self._create_alert(
                    AlertLevel.ERROR,
                    record.model_name,
                    f"錯誤率過高: {error_rate:.1%}",
                    {
                        "error_rate": error_rate,
                        "error_count": error_count,
                        "total_requests": len(recent_records),
                        "window_minutes": self.alert_rules["high_error_rate"][
                            "window_minutes"
                        ],
                    },
                )

    def _create_alert(
        self, level: AlertLevel, model_name: str, message: str, details: Dict[str, Any]
    ) -> None:
        """建立警報"""
        alert = Alert(
            timestamp=time.time(),
            level=level,
            model_name=model_name,
            message=message,
            details=details,
        )

        self.alerts.append(alert)

        # 記錄警報日誌
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }[level]

        # 安全地獲取級別字符串
        if hasattr(level, "value"):
            level_str = level.value.upper()
        else:
            level_str = str(level).upper()
        log_func(f"[{level_str}] {model_name}: {message}")

        # 保持警報列表在合理大小
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]

    def get_model_stats(
        self, model_name: Optional[str] = None
    ) -> Union[ModelStats, Dict[str, ModelStats]]:
        """取得模型統計資訊

        Args:
            model_name: 模型名稱，如果為 None 則返回所有模型

        Returns:
            模型統計資訊
        """
        with self._lock:
            if model_name:
                return self.model_stats.get(model_name, ModelStats(model_name))
            else:
                return self.model_stats.copy()

    def get_usage_summary(
        self, time_range_hours: int = 24, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """取得使用量摘要

        Args:
            time_range_hours: 時間範圍（小時）
            model_name: 模型名稱篩選

        Returns:
            使用量摘要
        """
        with self._lock:
            current_time = time.time()
            start_time = current_time - (time_range_hours * 3600)

            # 篩選記錄
            filtered_records = [
                r
                for r in self.usage_records
                if r.timestamp >= start_time
                and (not model_name or r.model_name == model_name)
            ]

            if not filtered_records:
                return {
                    "time_range_hours": time_range_hours,
                    "model_filter": model_name,
                    "total_requests": 0,
                    "message": "沒有找到符合條件的記錄",
                }

            # 計算統計資訊
            total_requests = len(filtered_records)
            successful_requests = sum(1 for r in filtered_records if r.success)
            failed_requests = total_requests - successful_requests

            total_input_count = sum(r.input_count for r in filtered_records)
            total_processing_time = sum(
                r.processing_time for r in filtered_records if r.success
            )
            total_memory_used = sum(
                r.memory_used for r in filtered_records if r.success
            )

            # 按模型分組統計
            model_breakdown = defaultdict(
                lambda: {
                    "requests": 0,
                    "successful": 0,
                    "failed": 0,
                    "input_count": 0,
                    "processing_time": 0.0,
                    "memory_used": 0.0,
                }
            )

            for record in filtered_records:
                stats = model_breakdown[record.model_name]
                stats["requests"] += 1
                stats["input_count"] += record.input_count

                if record.success:
                    stats["successful"] += 1
                    stats["processing_time"] += record.processing_time
                    stats["memory_used"] += record.memory_used
                else:
                    stats["failed"] += 1

            # 按裝置分組統計
            device_breakdown = defaultdict(int)
            for record in filtered_records:
                device_breakdown[record.device] += 1

            # 按操作分組統計
            operation_breakdown = defaultdict(int)
            for record in filtered_records:
                operation_breakdown[record.operation] += 1

            return {
                "time_range_hours": time_range_hours,
                "model_filter": model_name,
                "summary": {
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "success_rate": (
                        successful_requests / total_requests
                        if total_requests > 0
                        else 0.0
                    ),
                    "total_input_count": total_input_count,
                    "total_processing_time": total_processing_time,
                    "total_memory_used": total_memory_used,
                    "average_processing_time": (
                        total_processing_time / successful_requests
                        if successful_requests > 0
                        else 0.0
                    ),
                    "average_throughput": (
                        total_input_count / total_processing_time
                        if total_processing_time > 0
                        else 0.0
                    ),
                },
                "model_breakdown": dict(model_breakdown),
                "device_breakdown": dict(device_breakdown),
                "operation_breakdown": dict(operation_breakdown),
            }

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        model_name: Optional[str] = None,
        unresolved_only: bool = True,
        limit: int = 100,
    ) -> List[Alert]:
        """取得警報列表

        Args:
            level: 警報等級篩選
            model_name: 模型名稱篩選
            unresolved_only: 是否只返回未解決的警報
            limit: 返回數量限制

        Returns:
            警報列表
        """
        with self._lock:
            filtered_alerts = []

            for alert in reversed(self.alerts):  # 最新的在前
                if level and alert.level != level:
                    continue
                if model_name and alert.model_name != model_name:
                    continue
                if unresolved_only and alert.resolved:
                    continue

                filtered_alerts.append(alert)

                if len(filtered_alerts) >= limit:
                    break

            return filtered_alerts

    def resolve_alert(self, alert_index: int) -> bool:
        """解決警報

        Args:
            alert_index: 警報索引

        Returns:
            是否成功解決
        """
        with self._lock:
            if 0 <= alert_index < len(self.alerts):
                alert = self.alerts[alert_index]
                if not alert.resolved:
                    alert.resolved = True
                    alert.resolved_timestamp = time.time()
                    logger.info(f"警報已解決: {alert.model_name} - {alert.message}")
                    return True
            return False

    def get_performance_trends(
        self,
        model_name: str,
        hours: int = 24,
        granularity: str = "hour",  # "hour" or "day"
    ) -> Dict[str, Any]:
        """取得效能趨勢資料

        Args:
            model_name: 模型名稱
            hours: 時間範圍（小時）
            granularity: 時間粒度

        Returns:
            趨勢資料
        """
        with self._lock:
            current_time = time.time()
            start_time = current_time - (hours * 3600)

            # 篩選記錄
            records = [
                r
                for r in self.usage_records
                if r.timestamp >= start_time
                and r.model_name == model_name
                and r.success
            ]

            if not records:
                return {"error": f"沒有找到模型 {model_name} 的記錄"}

            # 按時間分組
            time_buckets = defaultdict(list)

            for record in records:
                if granularity == "hour":
                    bucket_key = datetime.fromtimestamp(record.timestamp).strftime(
                        "%Y-%m-%d %H:00"
                    )
                else:  # day
                    bucket_key = datetime.fromtimestamp(record.timestamp).strftime(
                        "%Y-%m-%d"
                    )

                time_buckets[bucket_key].append(record)

            # 計算趨勢資料
            trend_data = []

            for time_key in sorted(time_buckets.keys()):
                bucket_records = time_buckets[time_key]

                total_requests = len(bucket_records)
                total_input_count = sum(r.input_count for r in bucket_records)
                total_processing_time = sum(r.processing_time for r in bucket_records)
                total_memory_used = sum(r.memory_used for r in bucket_records)

                avg_processing_time = total_processing_time / total_requests
                avg_throughput = (
                    total_input_count / total_processing_time
                    if total_processing_time > 0
                    else 0
                )
                avg_memory_per_request = total_memory_used / total_requests

                trend_data.append(
                    {
                        "time": time_key,
                        "requests": total_requests,
                        "input_count": total_input_count,
                        "avg_processing_time": avg_processing_time,
                        "avg_throughput": avg_throughput,
                        "avg_memory_per_request": avg_memory_per_request,
                    }
                )

            return {
                "model_name": model_name,
                "time_range_hours": hours,
                "granularity": granularity,
                "data_points": len(trend_data),
                "trend_data": trend_data,
            }

    async def _load_historical_data(self) -> None:
        """載入歷史資料"""
        try:
            # 載入模型統計
            stats_file = self.storage_dir / "model_stats.json"
            if stats_file.exists():
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats_data = json.load(f)

                for model_name, stats_dict in stats_data.items():
                    self.model_stats[model_name] = ModelStats(**stats_dict)

                logger.info(f"載入了 {len(self.model_stats)} 個模型的歷史統計")

            # 載入最近的使用記錄
            records_file = self.storage_dir / "recent_records.json"
            if records_file.exists():
                with open(records_file, "r", encoding="utf-8") as f:
                    records_data = json.load(f)

                for record_dict in records_data:
                    record = UsageRecord(**record_dict)
                    self.usage_records.append(record)

                logger.info(f"載入了 {len(self.usage_records)} 條歷史記錄")

        except Exception as e:
            logger.error(f"載入歷史資料失敗: {e}")

    def _start_auto_save(self) -> None:
        """啟動自動儲存任務"""
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._auto_save_loop())

    async def _auto_save_loop(self) -> None:
        """自動儲存循環"""
        while not self._stop_save_task:
            try:
                await asyncio.sleep(self.save_interval)
                await self.save_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自動儲存失敗: {e}")

    async def save_data(self) -> None:
        """儲存資料到磁碟"""
        try:
            with self._lock:
                # 儲存模型統計
                stats_file = self.storage_dir / "model_stats.json"
                stats_data = {
                    name: asdict(stats) for name, stats in self.model_stats.items()
                }

                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(stats_data, f, ensure_ascii=False, indent=2)

                # 儲存最近的記錄
                records_file = self.storage_dir / "recent_records.json"
                records_data = [asdict(record) for record in list(self.usage_records)]

                with open(records_file, "w", encoding="utf-8") as f:
                    json.dump(records_data, f, ensure_ascii=False, indent=2)

                # 儲存警報
                alerts_file = self.storage_dir / "alerts.json"
                alerts_data = []
                for alert in self.alerts:
                    alert_dict = asdict(alert)
                    alert_dict["level"] = (
                        alert.level.value
                        if hasattr(alert.level, "value")
                        else str(alert.level)
                    )
                    alerts_data.append(alert_dict)

                with open(alerts_file, "w", encoding="utf-8") as f:
                    json.dump(alerts_data, f, ensure_ascii=False, indent=2)

                logger.debug("使用量資料已儲存")

        except Exception as e:
            logger.error(f"儲存使用量資料失敗: {e}")

    def export_report(
        self,
        output_file: Union[str, Path],
        time_range_hours: int = 168,  # 一週
        format: str = "json",  # "json" or "csv"
    ) -> bool:
        """匯出使用量報告

        Args:
            output_file: 輸出檔案路徑
            time_range_hours: 時間範圍（小時）
            format: 輸出格式

        Returns:
            是否成功匯出
        """
        try:
            summary = self.get_usage_summary(time_range_hours)

            if format == "json":
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

            elif format == "csv":
                import csv

                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)

                    # 寫入摘要資訊
                    writer.writerow(["指標", "數值"])
                    for key, value in summary["summary"].items():
                        writer.writerow([key, value])

                    writer.writerow([])  # 空行

                    # 寫入模型分解
                    writer.writerow(["模型統計"])
                    writer.writerow(
                        [
                            "模型名稱",
                            "請求數",
                            "成功數",
                            "失敗數",
                            "輸入數量",
                            "處理時間",
                            "記憶體使用",
                        ]
                    )

                    for model_name, stats in summary["model_breakdown"].items():
                        writer.writerow(
                            [
                                model_name,
                                stats["requests"],
                                stats["successful"],
                                stats["failed"],
                                stats["input_count"],
                                f"{stats['processing_time']:.2f}",
                                f"{stats['memory_used']:.1f}",
                            ]
                        )

            else:
                raise ValueError(f"不支援的格式: {format}")

            logger.info(f"使用量報告已匯出: {output_file}")
            return True

        except Exception as e:
            logger.error(f"匯出使用量報告失敗: {e}")
            return False

    def stop(self) -> None:
        """停止監控器"""
        self._stop_save_task = True
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()

        # 最後儲存一次資料
        asyncio.create_task(self.save_data())

        logger.info("使用量監控器已停止")


# 全域監控器實例
_usage_monitor: Optional[UsageMonitor] = None


def get_usage_monitor(**kwargs) -> UsageMonitor:
    """取得全域使用量監控器實例"""
    global _usage_monitor
    if _usage_monitor is None:
        _usage_monitor = UsageMonitor(**kwargs)
    return _usage_monitor


def record_embedding_usage(
    model_name: str, operation: str, input_count: int, processing_time: float, **kwargs
) -> None:
    """記錄 embedding 使用量的便利函數"""
    monitor = get_usage_monitor()
    monitor.record_usage(
        model_name=model_name,
        operation=operation,
        input_count=input_count,
        processing_time=processing_time,
        **kwargs,
    )
