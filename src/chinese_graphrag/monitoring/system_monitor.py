"""系統監控模組。

本模組提供系統資源監控和效能追蹤功能，包括：
- CPU 和記憶體使用率監控
- 磁碟 I/O 監控
- 網路監控
- 系統健康檢查
- 資源使用告警
"""

import os
import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
from loguru import logger


@dataclass
class SystemStats:
    """系統統計資料。"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    load_average: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典。"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "disk_percent": self.disk_percent,
            "disk_used_gb": self.disk_used_gb,
            "disk_total_gb": self.disk_total_gb,
            "network_sent_mb": self.network_sent_mb,
            "network_recv_mb": self.network_recv_mb,
            "process_count": self.process_count,
            "load_average": self.load_average,
        }


@dataclass
class ProcessStats:
    """程序統計資料。"""

    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    status: str
    create_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典。"""
        return {
            "pid": self.pid,
            "name": self.name,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_mb": self.memory_mb,
            "status": self.status,
            "create_time": self.create_time.isoformat(),
        }


@dataclass
class HealthCheck:
    """健康檢查結果。"""

    name: str
    status: str  # "healthy", "warning", "critical"
    message: str
    timestamp: datetime
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典。"""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class SystemMonitor:
    """系統監控器。"""

    def __init__(
        self,
        sample_interval: int = 60,
        history_size: int = 1440,  # 24小時（每分鐘一個樣本）
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        """初始化系統監控器。

        Args:
            sample_interval: 採樣間隔（秒）
            history_size: 歷史資料大小
            alert_thresholds: 告警閾值
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.lock = threading.RLock()

        # 預設告警閾值
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "load_average": 5.0,
        }

        # 監控資料
        self.stats_history: List[SystemStats] = []
        self.current_stats: Optional[SystemStats] = None

        # 監控執行緒
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 健康檢查
        self.health_checks: Dict[str, Callable[[], HealthCheck]] = {}

        # 告警回調
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        # 網路統計基準值
        self._network_baseline: Optional[psutil._common.snetio] = None
        self._last_network_time = time.time()

    def start_monitoring(self) -> None:
        """開始監控。"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("系統監控已啟動")

    def stop_monitoring(self) -> None:
        """停止監控。"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("系統監控已停止")

    def _monitor_loop(self) -> None:
        """監控主迴圈。"""
        while self.monitoring:
            try:
                # 收集系統統計
                stats = self.collect_system_stats()

                with self.lock:
                    # 更新當前統計
                    self.current_stats = stats

                    # 添加到歷史記錄
                    self.stats_history.append(stats)

                    # 限制歷史記錄大小
                    if len(self.stats_history) > self.history_size:
                        self.stats_history.pop(0)

                # 檢查告警
                self._check_system_alerts(stats)

                # 等待下一個採樣
                time.sleep(self.sample_interval)

            except Exception as e:
                logger.error(f"監控迴圈錯誤: {e}")
                time.sleep(self.sample_interval)

    def collect_system_stats(self) -> SystemStats:
        """收集系統統計資料。"""
        now = datetime.now()

        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 記憶體使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)

        # 磁碟使用率
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)

        # 網路統計
        network_sent_mb, network_recv_mb = self._get_network_stats()

        # 程序數量
        process_count = len(psutil.pids())

        # 負載平均（僅 Unix 系統）
        load_average = None
        if hasattr(os, "getloadavg"):
            try:
                load_average = os.getloadavg()
            except (OSError, AttributeError):
                pass

        return SystemStats(
            timestamp=now,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_count=process_count,
            load_average=load_average,
        )

    def _get_network_stats(self) -> Tuple[float, float]:
        """取得網路統計（MB/s）。"""
        current_time = time.time()
        current_network = psutil.net_io_counters()

        if self._network_baseline is None:
            self._network_baseline = current_network
            self._last_network_time = current_time
            return 0.0, 0.0

        # 計算時間差
        time_diff = current_time - self._last_network_time
        if time_diff <= 0:
            return 0.0, 0.0

        # 計算傳輸速率（MB/s）
        sent_mb = (
            (current_network.bytes_sent - self._network_baseline.bytes_sent)
            / (1024**2)
            / time_diff
        )
        recv_mb = (
            (current_network.bytes_recv - self._network_baseline.bytes_recv)
            / (1024**2)
            / time_diff
        )

        # 更新基準值
        self._network_baseline = current_network
        self._last_network_time = current_time

        return max(0, sent_mb), max(0, recv_mb)

    def _check_system_alerts(self, stats: SystemStats) -> None:
        """檢查系統告警。"""
        alerts = []

        # CPU 告警
        if stats.cpu_percent > self.alert_thresholds.get("cpu_percent", 100):
            alerts.append(
                {
                    "type": "cpu_high",
                    "message": f"CPU 使用率過高: {stats.cpu_percent:.1f}%",
                    "value": stats.cpu_percent,
                    "threshold": self.alert_thresholds["cpu_percent"],
                }
            )

        # 記憶體告警
        if stats.memory_percent > self.alert_thresholds.get("memory_percent", 100):
            alerts.append(
                {
                    "type": "memory_high",
                    "message": f"記憶體使用率過高: {stats.memory_percent:.1f}%",
                    "value": stats.memory_percent,
                    "threshold": self.alert_thresholds["memory_percent"],
                }
            )

        # 磁碟告警
        if stats.disk_percent > self.alert_thresholds.get("disk_percent", 100):
            alerts.append(
                {
                    "type": "disk_high",
                    "message": f"磁碟使用率過高: {stats.disk_percent:.1f}%",
                    "value": stats.disk_percent,
                    "threshold": self.alert_thresholds["disk_percent"],
                }
            )

        # 負載平均告警
        if stats.load_average and stats.load_average[0] > self.alert_thresholds.get(
            "load_average", 100
        ):
            alerts.append(
                {
                    "type": "load_high",
                    "message": f"系統負載過高: {stats.load_average[0]:.2f}",
                    "value": stats.load_average[0],
                    "threshold": self.alert_thresholds["load_average"],
                }
            )

        # 發送告警
        for alert in alerts:
            self._send_alert(alert["type"], alert)

    def _send_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """發送告警。"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"告警回調失敗: {e}")

        # 記錄到日誌
        logger.warning(f"系統告警: {alert_data['message']}")

    def get_current_stats(self) -> Optional[SystemStats]:
        """取得當前統計資料。"""
        return self.current_stats

    def get_stats_history(self, hours: int = 1) -> List[SystemStats]:
        """取得歷史統計資料。

        Args:
            hours: 取得多少小時的歷史資料

        Returns:
            歷史統計資料列表
        """
        with self.lock:
            if not self.stats_history:
                return []

            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [
                stats for stats in self.stats_history if stats.timestamp >= cutoff_time
            ]

    def get_process_stats(self, top_n: int = 10) -> List[ProcessStats]:
        """取得程序統計資料。

        Args:
            top_n: 取得前 N 個程序

        Returns:
            程序統計資料列表
        """
        processes = []

        try:
            for proc in psutil.process_iter(
                [
                    "pid",
                    "name",
                    "cpu_percent",
                    "memory_percent",
                    "memory_info",
                    "status",
                    "create_time",
                ]
            ):
                try:
                    info = proc.info
                    if info["memory_info"]:
                        memory_mb = info["memory_info"].rss / (1024**2)
                    else:
                        memory_mb = 0

                    create_time = (
                        datetime.fromtimestamp(info["create_time"])
                        if info["create_time"]
                        else datetime.now()
                    )

                    processes.append(
                        ProcessStats(
                            pid=info["pid"],
                            name=info["name"] or "Unknown",
                            cpu_percent=info["cpu_percent"] or 0,
                            memory_percent=info["memory_percent"] or 0,
                            memory_mb=memory_mb,
                            status=info["status"] or "unknown",
                            create_time=create_time,
                        )
                    )
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue
                except Exception as e:
                    # 記錄但不中斷整個過程
                    logger.debug(f"獲取程序資訊時發生錯誤: {e}")
                    continue
        except Exception as e:
            logger.error(f"獲取程序列表時發生錯誤: {e}")
            return []

        # 按 CPU 使用率排序
        processes.sort(key=lambda x: x.cpu_percent, reverse=True)
        return processes[:top_n]

    def add_health_check(
        self, name: str, check_func: Callable[[], HealthCheck]
    ) -> None:
        """添加健康檢查。

        Args:
            name: 檢查名稱
            check_func: 檢查函數
        """
        self.health_checks[name] = check_func

    def run_health_checks(self) -> List[HealthCheck]:
        """執行所有健康檢查。

        Returns:
            健康檢查結果列表
        """
        results = []

        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results.append(result)
            except Exception as e:
                results.append(
                    HealthCheck(
                        name=name,
                        status="critical",
                        message=f"健康檢查失敗: {e}",
                        timestamp=datetime.now(),
                        details={"error": str(e)},
                    )
                )

        return results

    def add_alert_callback(
        self, callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """添加告警回調函數。

        Args:
            callback: 告警回調函數
        """
        self.alert_callbacks.append(callback)

    def get_system_info(self) -> Dict[str, Any]:
        """取得系統資訊。

        Returns:
            系統資訊字典
        """
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_total_gb": psutil.disk_usage("/").total / (1024**3),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
        }


# 全域系統監控器實例
_system_monitor: Optional[SystemMonitor] = None
_system_monitor_lock = threading.Lock()


def get_system_monitor() -> SystemMonitor:
    """取得全域系統監控器實例。

    Returns:
        系統監控器實例
    """
    global _system_monitor

    if _system_monitor is None:
        with _system_monitor_lock:
            if _system_monitor is None:
                _system_monitor = SystemMonitor()

    return _system_monitor


def start_system_monitoring() -> None:
    """便利函數：開始系統監控。"""
    get_system_monitor().start_monitoring()


def stop_system_monitoring() -> None:
    """便利函數：停止系統監控。"""
    if _system_monitor:
        _system_monitor.stop_monitoring()
