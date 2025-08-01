"""
效能監控和指標收集系統

提供系統效能監控、指標收集、統計分析和報告功能
"""

import time
import psutil
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import json
import asyncio
from contextlib import contextmanager
from enum import Enum

from .logger import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """指標類型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricRecord:
    """指標記錄"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class SystemMetrics:
    """系統指標"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    thread_count: int


@dataclass
class PerformanceStats:
    """效能統計"""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, duration: float):
        """更新統計資料"""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.count
        self.recent_times.append(duration)
        
        # 計算百分位數
        if self.recent_times:
            sorted_times = sorted(self.recent_times)
            n = len(sorted_times)
            self.p50 = sorted_times[int(n * 0.5)]
            self.p95 = sorted_times[int(n * 0.95)]
            self.p99 = sorted_times[int(n * 0.99)]


class MetricsCollector:
    """指標收集器"""
    
    def __init__(self, 
                 storage_dir: str = "logs/metrics",
                 collection_interval: int = 60,
                 max_records: int = 10000):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.collection_interval = collection_interval
        self.max_records = max_records
        
        # 指標儲存
        self.metrics: Dict[str, List[MetricRecord]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        
        # 系統指標
        self.system_metrics: List[SystemMetrics] = []
        
        # 控制變數
        self._stop_collection = False
        self._collection_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # 初始化系統監控
        self._init_system_monitoring()
        
        logger.info(f"指標收集器初始化完成，儲存目錄: {self.storage_dir}")
        
    def _init_system_monitoring(self):
        """初始化系統監控"""
        try:
            # 初始化網路統計
            self._initial_net_io = psutil.net_io_counters()
            self._last_net_check = time.time()
        except Exception as e:
            logger.warning(f"初始化系統監控失敗: {e}")
            self._initial_net_io = None
            
    def start_collection(self):
        """開始指標收集"""
        if self._collection_task is None or self._collection_task.done():
            self._stop_collection = False
            self._collection_task = asyncio.create_task(self._collection_loop())
            logger.info("指標收集已開始")
            
    async def _collection_loop(self):
        """指標收集循環"""
        while not self._stop_collection:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"指標收集錯誤: {e}")
                await asyncio.sleep(5)  # 錯誤後短暫等待
                
    async def _collect_system_metrics(self):
        """收集系統指標"""
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 記憶體使用率
            memory = psutil.virtual_memory()
            
            # 磁碟使用率
            disk = psutil.disk_usage('/')
            
            # 網路統計
            net_sent_mb = 0
            net_recv_mb = 0
            if self._initial_net_io:
                current_net_io = psutil.net_io_counters()
                current_time = time.time()
                time_diff = current_time - self._last_net_check
                
                if time_diff > 0:
                    net_sent_mb = (current_net_io.bytes_sent - self._initial_net_io.bytes_sent) / (1024 * 1024)
                    net_recv_mb = (current_net_io.bytes_recv - self._initial_net_io.bytes_recv) / (1024 * 1024)
                
                self._last_net_check = current_time
            
            # 程序統計
            process_count = len(psutil.pids())
            thread_count = threading.active_count()
            
            # 建立系統指標記錄
            system_metric = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb,
                process_count=process_count,
                thread_count=thread_count
            )
            
            with self._lock:
                self.system_metrics.append(system_metric)
                
                # 限制記錄數量
                if len(self.system_metrics) > self.max_records:
                    self.system_metrics = self.system_metrics[-self.max_records:]
                    
            # 記錄到指標系統
            self.record_gauge("system.cpu_percent", cpu_percent, {"unit": "percent"})
            self.record_gauge("system.memory_percent", memory.percent, {"unit": "percent"})
            self.record_gauge("system.disk_usage_percent", disk.percent, {"unit": "percent"})
            
        except Exception as e:
            logger.error(f"收集系統指標失敗: {e}")
            
    def record_counter(self, name: str, value: float = 1, labels: Dict[str, str] = None):
        """記錄計數器指標"""
        labels = labels or {}
        with self._lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True)}"
            self.counters[key] += value
            
            record = MetricRecord(
                name=name,
                value=self.counters[key],
                metric_type=MetricType.COUNTER,
                timestamp=datetime.now(),
                labels=labels
            )
            self.metrics[name].append(record)
            self._trim_metrics(name)
            
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """記錄儀表指標"""
        labels = labels or {}
        with self._lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True)}"
            self.gauges[key] = value
            
            record = MetricRecord(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=datetime.now(),
                labels=labels
            )
            self.metrics[name].append(record)
            self._trim_metrics(name)
            
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """記錄直方圖指標"""
        labels = labels or {}
        with self._lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True)}"
            self.histograms[key].append(value)
            
            record = MetricRecord(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=datetime.now(),
                labels=labels
            )
            self.metrics[name].append(record)
            self._trim_metrics(name)
            
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """記錄計時器指標"""
        labels = labels or {}
        with self._lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True)}"
            self.timers[key].update(duration)
            
            record = MetricRecord(
                name=name,
                value=duration,
                metric_type=MetricType.TIMER,
                timestamp=datetime.now(),
                labels=labels
            )
            self.metrics[name].append(record)
            self._trim_metrics(name)
            
    @contextmanager
    def timer(self, name: str, labels: Dict[str, str] = None):
        """計時器上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timer(name, duration, labels)
            
    def _trim_metrics(self, name: str):
        """修剪指標記錄"""
        if len(self.metrics[name]) > self.max_records:
            self.metrics[name] = self.metrics[name][-self.max_records:]
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """取得指標摘要"""
        with self._lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timers": {},
                "system_metrics": None
            }
            
            # 計時器統計
            for name, stats in self.timers.items():
                summary["timers"][name] = {
                    "count": stats.count,
                    "avg_time": stats.avg_time,
                    "min_time": stats.min_time,
                    "max_time": stats.max_time,
                    "p50": stats.p50,
                    "p95": stats.p95,
                    "p99": stats.p99
                }
                
            # 最新系統指標
            if self.system_metrics:
                latest_system = self.system_metrics[-1]
                summary["system_metrics"] = {
                    "cpu_percent": latest_system.cpu_percent,
                    "memory_percent": latest_system.memory_percent,
                    "disk_usage_percent": latest_system.disk_usage_percent,
                    "process_count": latest_system.process_count,
                    "thread_count": latest_system.thread_count
                }
                
        return summary
        
    async def export_metrics(self, output_file: Optional[str] = None) -> bool:
        """匯出指標資料"""
        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.storage_dir / f"metrics_export_{timestamp}.json"
            else:
                output_file = Path(output_file)
                
            summary = self.get_metrics_summary()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"指標資料已匯出: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"匯出指標資料失敗: {e}")
            return False
            
    def stop_collection(self):
        """停止指標收集"""
        self._stop_collection = True
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            
        logger.info("指標收集已停止")


# 全域指標收集器
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(**kwargs) -> MetricsCollector:
    """取得全域指標收集器實例"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(**kwargs)
    return _metrics_collector


def record_counter(name: str, value: float = 1, labels: Dict[str, str] = None):
    """記錄計數器指標的便利函數"""
    collector = get_metrics_collector()
    collector.record_counter(name, value, labels)


def record_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """記錄儀表指標的便利函數"""
    collector = get_metrics_collector()
    collector.record_gauge(name, value, labels)


def record_timer(name: str, duration: float, labels: Dict[str, str] = None):
    """記錄計時器指標的便利函數"""
    collector = get_metrics_collector()
    collector.record_timer(name, duration, labels)


@contextmanager
def timer(name: str, labels: Dict[str, str] = None):
    """計時器上下文管理器的便利函數"""
    collector = get_metrics_collector()
    with collector.timer(name, labels):
        yield