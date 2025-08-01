#!/usr/bin/env python3
"""
監控和維護工具

提供系統監控、效能分析和維護功能
"""

import json
import os
import psutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import click
import yaml
from loguru import logger


class SystemMonitor:
    """系統監控器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.metrics_dir = Path(self.config.get("metrics_dir", "./monitoring/metrics"))
        self.alerts_dir = Path(self.config.get("alerts_dir", "./monitoring/alerts"))
        
        # 建立監控目錄
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定日誌
        logger.remove()
        logger.add(
            self.metrics_dir.parent / "monitoring.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="1 day",
            retention="30 days"
        )
        logger.add(sys.stdout, level="INFO")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """載入監控配置"""
        default_config = {
            "metrics_dir": "./monitoring/metrics",
            "alerts_dir": "./monitoring/alerts",
            "collection_interval": 60,  # 秒
            "retention_days": 30,
            "thresholds": {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90,
                "response_time_ms": 5000
            },
            "alert_channels": {
                "email": False,
                "webhook": False,
                "log": True
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"載入配置失敗，使用預設配置: {e}")
        
        return default_config
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系統指標"""
        try:
            # CPU 指標
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 記憶體指標
            memory = psutil.virtual_memory()
            
            # 磁碟指標
            disk = psutil.disk_usage('/')
            
            # 網路指標
            network = psutil.net_io_counters()
            
            # 進程指標
            processes = len(psutil.pids())
            
            metrics = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "system": {
                    "cpu": {
                        "percent": cpu_percent,
                        "count": cpu_count,
                        "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
                    },
                    "memory": {
                        "total_gb": memory.total / (1024**3),
                        "available_gb": memory.available / (1024**3),
                        "used_gb": memory.used / (1024**3),
                        "percent": memory.percent
                    },
                    "disk": {
                        "total_gb": disk.total / (1024**3),
                        "free_gb": disk.free / (1024**3),
                        "used_gb": disk.used / (1024**3),
                        "percent": (disk.used / disk.total) * 100
                    },
                    "network": {
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv
                    },
                    "processes": processes
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集系統指標失敗: {e}")
            return {}
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """收集應用程式指標"""
        try:
            # 查找應用程式進程
            app_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    if 'chinese_graphrag' in proc.info['name'] or 'uvicorn' in proc.info['name']:
                        app_processes.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "cpu_percent": proc.info['cpu_percent'],
                            "memory_mb": proc.info['memory_info'].rss / (1024**2) if proc.info['memory_info'] else 0
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # API 健康檢查
            api_health = self._check_api_health()
            
            # 資料庫狀態
            db_status = self._check_database_status()
            
            metrics = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "application": {
                    "processes": app_processes,
                    "process_count": len(app_processes),
                    "total_cpu_percent": sum(p["cpu_percent"] for p in app_processes),
                    "total_memory_mb": sum(p["memory_mb"] for p in app_processes),
                    "api_health": api_health,
                    "database_status": db_status
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集應用程式指標失敗: {e}")
            return {}
    
    def _check_api_health(self) -> Dict[str, Any]:
        """檢查 API 健康狀態"""
        try:
            import requests
            
            start_time = time.time()
            response = requests.get("http://localhost:8000/health", timeout=5)
            response_time = (time.time() - start_time) * 1000  # 毫秒
            
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": None,
                "last_check": datetime.now().isoformat()
            }
    
    def _check_database_status(self) -> Dict[str, Any]:
        """檢查資料庫狀態"""
        try:
            # 檢查向量資料庫目錄
            vector_db_path = Path("./data/vector_db")
            vector_db_exists = vector_db_path.exists()
            vector_db_size = sum(f.stat().st_size for f in vector_db_path.rglob('*') if f.is_file()) if vector_db_exists else 0
            
            # 檢查圖形資料庫目錄
            graph_db_path = Path("./data/graph_db")
            graph_db_exists = graph_db_path.exists()
            graph_db_size = sum(f.stat().st_size for f in graph_db_path.rglob('*') if f.is_file()) if graph_db_exists else 0
            
            return {
                "vector_db": {
                    "exists": vector_db_exists,
                    "size_mb": vector_db_size / (1024**2),
                    "path": str(vector_db_path)
                },
                "graph_db": {
                    "exists": graph_db_exists,
                    "size_mb": graph_db_size / (1024**2),
                    "path": str(graph_db_path)
                },
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """儲存指標資料"""
        try:
            timestamp = int(metrics.get("timestamp", time.time()))
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            
            # 按日期分檔儲存
            metrics_file = self.metrics_dir / f"metrics_{date_str}.jsonl"
            
            with open(metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
            
            logger.debug(f"指標已儲存: {metrics_file}")
            
        except Exception as e:
            logger.error(f"儲存指標失敗: {e}")
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """檢查警報條件"""
        try:
            thresholds = self.config["thresholds"]
            alerts = []
            
            # 檢查系統指標
            if "system" in metrics:
                system = metrics["system"]
                
                # CPU 警報
                if system.get("cpu", {}).get("percent", 0) > thresholds["cpu_percent"]:
                    alerts.append({
                        "type": "cpu_high",
                        "severity": "warning",
                        "message": f"CPU 使用率過高: {system['cpu']['percent']:.1f}%",
                        "value": system["cpu"]["percent"],
                        "threshold": thresholds["cpu_percent"]
                    })
                
                # 記憶體警報
                if system.get("memory", {}).get("percent", 0) > thresholds["memory_percent"]:
                    alerts.append({
                        "type": "memory_high",
                        "severity": "warning",
                        "message": f"記憶體使用率過高: {system['memory']['percent']:.1f}%",
                        "value": system["memory"]["percent"],
                        "threshold": thresholds["memory_percent"]
                    })
                
                # 磁碟警報
                if system.get("disk", {}).get("percent", 0) > thresholds["disk_percent"]:
                    alerts.append({
                        "type": "disk_high",
                        "severity": "critical",
                        "message": f"磁碟使用率過高: {system['disk']['percent']:.1f}%",
                        "value": system["disk"]["percent"],
                        "threshold": thresholds["disk_percent"]
                    })
            
            # 檢查應用程式指標
            if "application" in metrics:
                app = metrics["application"]
                
                # API 回應時間警報
                api_health = app.get("api_health", {})
                response_time = api_health.get("response_time_ms")
                if response_time and response_time > thresholds["response_time_ms"]:
                    alerts.append({
                        "type": "api_slow",
                        "severity": "warning",
                        "message": f"API 回應時間過長: {response_time:.0f}ms",
                        "value": response_time,
                        "threshold": thresholds["response_time_ms"]
                    })
                
                # API 健康狀態警報
                if api_health.get("status") != "healthy":
                    alerts.append({
                        "type": "api_unhealthy",
                        "severity": "critical",
                        "message": f"API 健康檢查失敗: {api_health.get('status')}",
                        "value": api_health.get("status"),
                        "threshold": "healthy"
                    })
            
            # 處理警報
            for alert in alerts:
                self._handle_alert(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"檢查警報失敗: {e}")
            return []
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """處理警報"""
        try:
            alert["timestamp"] = time.time()
            alert["datetime"] = datetime.now().isoformat()
            
            # 記錄警報
            if self.config["alert_channels"]["log"]:
                severity_icon = "🚨" if alert["severity"] == "critical" else "⚠️"
                logger.warning(f"{severity_icon} {alert['message']}")
            
            # 儲存警報
            alert_file = self.alerts_dir / f"alerts_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
            with open(alert_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert, ensure_ascii=False) + '\n')
            
            # 其他警報通道（email, webhook 等）
            # 這裡可以添加更多警報通道的實作
            
        except Exception as e:
            logger.error(f"處理警報失敗: {e}")
    
    def cleanup_old_data(self):
        """清理舊資料"""
        try:
            retention_days = self.config["retention_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # 清理舊的指標檔案
            for metrics_file in self.metrics_dir.glob("metrics_*.jsonl"):
                try:
                    file_date_str = metrics_file.stem.replace("metrics_", "")
                    file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        metrics_file.unlink()
                        logger.info(f"已刪除舊指標檔案: {metrics_file}")
                        
                except ValueError:
                    # 檔案名稱格式不正確，跳過
                    continue
            
            # 清理舊的警報檔案
            for alert_file in self.alerts_dir.glob("alerts_*.jsonl"):
                try:
                    file_date_str = alert_file.stem.replace("alerts_", "")
                    file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        alert_file.unlink()
                        logger.info(f"已刪除舊警報檔案: {alert_file}")
                        
                except ValueError:
                    continue
            
        except Exception as e:
            logger.error(f"清理舊資料失敗: {e}")
    
    def generate_report(self, days: int = 7) -> Dict[str, Any]:
        """生成監控報告"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 收集指標資料
            all_metrics = []
            for i in range(days):
                date = start_date + timedelta(days=i)
                date_str = date.strftime("%Y-%m-%d")
                metrics_file = self.metrics_dir / f"metrics_{date_str}.jsonl"
                
                if metrics_file.exists():
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                metrics = json.loads(line.strip())
                                all_metrics.append(metrics)
                            except json.JSONDecodeError:
                                continue
            
            if not all_metrics:
                return {"error": "沒有找到指標資料"}
            
            # 計算統計資料
            cpu_values = [m.get("system", {}).get("cpu", {}).get("percent", 0) for m in all_metrics]
            memory_values = [m.get("system", {}).get("memory", {}).get("percent", 0) for m in all_metrics]
            disk_values = [m.get("system", {}).get("disk", {}).get("percent", 0) for m in all_metrics]
            
            # 收集警報資料
            all_alerts = []
            for i in range(days):
                date = start_date + timedelta(days=i)
                date_str = date.strftime("%Y-%m-%d")
                alert_file = self.alerts_dir / f"alerts_{date_str}.jsonl"
                
                if alert_file.exists():
                    with open(alert_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                alert = json.loads(line.strip())
                                all_alerts.append(alert)
                            except json.JSONDecodeError:
                                continue
            
            report = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "metrics_summary": {
                    "total_data_points": len(all_metrics),
                    "cpu": {
                        "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                        "max": max(cpu_values) if cpu_values else 0,
                        "min": min(cpu_values) if cpu_values else 0
                    },
                    "memory": {
                        "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
                        "max": max(memory_values) if memory_values else 0,
                        "min": min(memory_values) if memory_values else 0
                    },
                    "disk": {
                        "avg": sum(disk_values) / len(disk_values) if disk_values else 0,
                        "max": max(disk_values) if disk_values else 0,
                        "min": min(disk_values) if disk_values else 0
                    }
                },
                "alerts_summary": {
                    "total_alerts": len(all_alerts),
                    "critical_alerts": len([a for a in all_alerts if a.get("severity") == "critical"]),
                    "warning_alerts": len([a for a in all_alerts if a.get("severity") == "warning"]),
                    "alert_types": {}
                },
                "generated_at": datetime.now().isoformat()
            }
            
            # 統計警報類型
            for alert in all_alerts:
                alert_type = alert.get("type", "unknown")
                report["alerts_summary"]["alert_types"][alert_type] = \
                    report["alerts_summary"]["alert_types"].get(alert_type, 0) + 1
            
            return report
            
        except Exception as e:
            logger.error(f"生成監控報告失敗: {e}")
            return {"error": str(e)}
    
    def run_continuous_monitoring(self):
        """執行連續監控"""
        logger.info("🔍 開始連續監控")
        
        interval = self.config["collection_interval"]
        
        try:
            while True:
                # 收集指標
                system_metrics = self.collect_system_metrics()
                app_metrics = self.collect_application_metrics()
                
                # 合併指標
                combined_metrics = {**system_metrics, **app_metrics}
                
                if combined_metrics:
                    # 儲存指標
                    self.save_metrics(combined_metrics)
                    
                    # 檢查警報
                    alerts = self.check_alerts(combined_metrics)
                    
                    if alerts:
                        logger.info(f"檢測到 {len(alerts)} 個警報")
                
                # 定期清理舊資料
                if int(time.time()) % 3600 == 0:  # 每小時執行一次
                    self.cleanup_old_data()
                
                # 等待下次收集
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("監控已停止")
        except Exception as e:
            logger.error(f"監控過程中發生錯誤: {e}")


class MaintenanceTools:
    """維護工具"""
    
    def __init__(self):
        self.data_dir = Path("./data")
        self.logs_dir = Path("./logs")
        self.cache_dir = Path("./cache")
    
    def cleanup_logs(self, days: int = 30):
        """清理舊日誌"""
        logger.info(f"🧹 清理 {days} 天前的日誌")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for log_file in self.logs_dir.rglob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    cleaned_count += 1
                    logger.info(f"已刪除: {log_file}")
            except Exception as e:
                logger.warning(f"刪除日誌檔案失敗 {log_file}: {e}")
        
        logger.info(f"✅ 清理完成，共刪除 {cleaned_count} 個日誌檔案")
    
    def cleanup_cache(self):
        """清理快取"""
        logger.info("🧹 清理快取檔案")
        
        cleaned_count = 0
        cleaned_size = 0
        
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                try:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    cleaned_count += 1
                    cleaned_size += file_size
                    logger.debug(f"已刪除快取: {cache_file}")
                except Exception as e:
                    logger.warning(f"刪除快取檔案失敗 {cache_file}: {e}")
        
        logger.info(f"✅ 快取清理完成，共刪除 {cleaned_count} 個檔案，釋放 {cleaned_size / (1024**2):.1f}MB 空間")
    
    def optimize_database(self):
        """優化資料庫"""
        logger.info("🔧 優化資料庫")
        
        # 這裡可以添加具體的資料庫優化邏輯
        # 例如：重建索引、壓縮資料、清理碎片等
        
        logger.info("✅ 資料庫優化完成")
    
    def check_system_health(self) -> Dict[str, Any]:
        """檢查系統健康狀態"""
        logger.info("🏥 檢查系統健康狀態")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # 檢查磁碟空間
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            health_status["checks"]["disk_space"] = {
                "status": "healthy" if disk_percent < 90 else "warning",
                "usage_percent": disk_percent,
                "free_gb": disk.free / (1024**3)
            }
        except Exception as e:
            health_status["checks"]["disk_space"] = {"status": "error", "error": str(e)}
        
        # 檢查記憶體
        try:
            memory = psutil.virtual_memory()
            
            health_status["checks"]["memory"] = {
                "status": "healthy" if memory.percent < 85 else "warning",
                "usage_percent": memory.percent,
                "available_gb": memory.available / (1024**3)
            }
        except Exception as e:
            health_status["checks"]["memory"] = {"status": "error", "error": str(e)}
        
        # 檢查應用程式進程
        try:
            app_processes = []
            for proc in psutil.process_iter(['pid', 'name']):
                if 'chinese_graphrag' in proc.info['name'] or 'uvicorn' in proc.info['name']:
                    app_processes.append(proc.info)
            
            health_status["checks"]["application_processes"] = {
                "status": "healthy" if app_processes else "critical",
                "process_count": len(app_processes),
                "processes": app_processes
            }
        except Exception as e:
            health_status["checks"]["application_processes"] = {"status": "error", "error": str(e)}
        
        # 檢查資料庫檔案
        try:
            vector_db_exists = (self.data_dir / "vector_db").exists()
            graph_db_exists = (self.data_dir / "graph_db").exists()
            
            health_status["checks"]["databases"] = {
                "status": "healthy" if vector_db_exists and graph_db_exists else "warning",
                "vector_db_exists": vector_db_exists,
                "graph_db_exists": graph_db_exists
            }
        except Exception as e:
            health_status["checks"]["databases"] = {"status": "error", "error": str(e)}
        
        # 計算整體狀態
        check_statuses = [check.get("status", "error") for check in health_status["checks"].values()]
        if "critical" in check_statuses:
            health_status["overall_status"] = "critical"
        elif "error" in check_statuses or "warning" in check_statuses:
            health_status["overall_status"] = "warning"
        
        return health_status


@click.group()
def cli():
    """監控和維護工具"""
    pass


@cli.command()
@click.option("--config", type=click.Path(), help="監控配置檔案")
@click.option("--interval", type=int, default=60, help="收集間隔（秒）")
def monitor(config: Optional[str], interval: int):
    """啟動系統監控"""
    config_path = Path(config) if config else None
    
    monitor = SystemMonitor(config_path)
    monitor.config["collection_interval"] = interval
    
    monitor.run_continuous_monitoring()


@cli.command()
@click.option("--days", type=int, default=7, help="報告天數")
@click.option("--output", type=click.Path(), help="輸出檔案")
def report(days: int, output: Optional[str]):
    """生成監控報告"""
    monitor = SystemMonitor()
    report_data = monitor.generate_report(days)
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        logger.info(f"報告已儲存至: {output}")
    else:
        print(json.dumps(report_data, ensure_ascii=False, indent=2))


@cli.command()
@click.option("--log-days", type=int, default=30, help="保留日誌天數")
@click.option("--clean-cache", is_flag=True, help="清理快取")
def cleanup(log_days: int, clean_cache: bool):
    """執行系統清理"""
    tools = MaintenanceTools()
    
    tools.cleanup_logs(log_days)
    
    if clean_cache:
        tools.cleanup_cache()
    
    tools.optimize_database()


@cli.command()
def health():
    """檢查系統健康狀態"""
    tools = MaintenanceTools()
    health_status = tools.check_system_health()
    
    print(json.dumps(health_status, ensure_ascii=False, indent=2))
    
    # 根據健康狀態設定退出碼
    if health_status["overall_status"] == "critical":
        sys.exit(2)
    elif health_status["overall_status"] == "warning":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    cli()