#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
監控管理 API 路由

提供系統監控、指標收集和狀態查看功能。
"""

import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from ...config.settings import Settings
from ...monitoring.logger import get_logger
from ..models import (
    SystemMetrics, ApplicationMetrics, MonitoringResponse, DataResponse,
    create_success_response, create_error_response
)


# 設定日誌
logger = get_logger(__name__)

# 創建路由器
router = APIRouter()

# 監控資料儲存（生產環境中應使用時序資料庫）
metrics_history: List[Dict[str, Any]] = []
application_stats = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "total_response_time": 0.0,
    "active_connections": 0,
    "index_count": 0,
    "last_index_time": None,
    "start_time": datetime.now()
}


@router.get("/monitoring/metrics", response_model=MonitoringResponse, summary="取得系統監控指標")
async def get_metrics(settings: Settings = Depends()) -> MonitoringResponse:
    """取得當前系統監控指標。
    
    包括系統資源使用情況和應用程式運行狀態。
    
    Args:
        settings: 應用程式設定
        
    Returns:
        系統和應用程式監控指標
    """
    try:
        # 收集系統指標
        system_metrics = await _collect_system_metrics()
        
        # 收集應用指標
        app_metrics = _collect_application_metrics()
        
        # 儲存歷史資料
        current_time = datetime.now()
        metrics_entry = {
            "timestamp": current_time,
            "system_metrics": system_metrics.dict(),
            "application_metrics": app_metrics.dict()
        }
        metrics_history.append(metrics_entry)
        
        # 只保留最近 24 小時的資料
        cutoff_time = current_time - timedelta(hours=24)
        global metrics_history
        metrics_history = [
            m for m in metrics_history 
            if m["timestamp"] > cutoff_time
        ]
        
        logger.debug("收集監控指標完成")
        
        return MonitoringResponse(
            success=True,
            message="成功取得監控指標",
            system_metrics=system_metrics,
            application_metrics=app_metrics
        )
        
    except Exception as e:
        logger.error(f"取得監控指標失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取得監控指標失敗: {str(e)}"
        )


@router.get("/monitoring/history", response_model=DataResponse, summary="取得監控歷史資料")
async def get_metrics_history(
    hours: int = 1,
    interval_minutes: int = 5
) -> DataResponse:
    """取得指定時間範圍內的監控歷史資料。
    
    Args:
        hours: 回溯小時數（預設 1 小時）
        interval_minutes: 資料間隔分鐘數（預設 5 分鐘）
        
    Returns:
        監控歷史資料
    """
    try:
        current_time = datetime.now()
        start_time = current_time - timedelta(hours=hours)
        
        # 過濾指定時間範圍的資料
        filtered_history = [
            m for m in metrics_history
            if start_time <= m["timestamp"] <= current_time
        ]
        
        # 依時間間隔取樣
        if interval_minutes > 0:
            sampled_history = []
            last_sampled_time = None
            
            for entry in sorted(filtered_history, key=lambda x: x["timestamp"]):
                if (last_sampled_time is None or 
                    (entry["timestamp"] - last_sampled_time).total_seconds() >= interval_minutes * 60):
                    sampled_history.append(entry)
                    last_sampled_time = entry["timestamp"]
            
            filtered_history = sampled_history
        
        logger.info(f"取得 {len(filtered_history)} 筆監控歷史資料")
        
        return create_success_response(
            data=filtered_history,
            message=f"成功取得 {len(filtered_history)} 筆監控歷史資料",
            meta={
                "time_range": {
                    "start_time": start_time.isoformat(),
                    "end_time": current_time.isoformat(),
                    "hours": hours
                },
                "interval_minutes": interval_minutes,
                "total_entries": len(filtered_history)
            }
        )
        
    except Exception as e:
        logger.error(f"取得監控歷史資料失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取得監控歷史資料失敗: {str(e)}"
        )


@router.get("/monitoring/alerts", response_model=DataResponse, summary="取得系統警告")
async def get_alerts(
    severity: Optional[str] = None,
    limit: int = 50
) -> DataResponse:
    """取得系統警告和異常事件。
    
    Args:
        severity: 警告級別過濾（info, warning, error, critical）
        limit: 回傳結果數量限制
        
    Returns:
        系統警告列表
    """
    try:
        # 檢查系統狀態並產生警告
        alerts = await _check_system_alerts()
        
        # 應用嚴重程度過濾
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        # 限制結果數量
        alerts = alerts[:limit]
        
        logger.info(f"取得 {len(alerts)} 個系統警告")
        
        return create_success_response(
            data=alerts,
            message=f"找到 {len(alerts)} 個警告",
            meta={
                "severity_filter": severity,
                "limit": limit
            }
        )
        
    except Exception as e:
        logger.error(f"取得系統警告失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取得系統警告失敗: {str(e)}"
        )


@router.get("/monitoring/performance", response_model=DataResponse, summary="取得效能統計")
async def get_performance_stats() -> DataResponse:
    """取得系統效能統計資訊。
    
    Returns:
        系統效能統計資料
    """
    try:
        # 計算效能統計
        current_time = datetime.now()
        uptime = (current_time - application_stats["start_time"]).total_seconds()
        
        # 平均回應時間
        avg_response_time = 0.0
        if application_stats["successful_queries"] > 0:
            avg_response_time = (
                application_stats["total_response_time"] / 
                application_stats["successful_queries"]
            )
        
        # 成功率
        total_queries = application_stats["total_queries"]
        success_rate = 0.0
        if total_queries > 0:
            success_rate = (application_stats["successful_queries"] / total_queries) * 100
        
        # QPS（每秒查詢數）
        qps = 0.0
        if uptime > 0:
            qps = total_queries / uptime
        
        performance_stats = {
            "uptime_seconds": uptime,
            "uptime_formatted": _format_uptime(uptime),
            "total_queries": total_queries,
            "successful_queries": application_stats["successful_queries"],
            "failed_queries": application_stats["failed_queries"],
            "success_rate_percent": round(success_rate, 2),
            "average_response_time_ms": round(avg_response_time * 1000, 2),
            "queries_per_second": round(qps, 2),
            "active_connections": application_stats["active_connections"],
            "index_count": application_stats["index_count"],
            "last_index_time": application_stats["last_index_time"]
        }
        
        # 最近 1 小時的資料統計
        one_hour_ago = current_time - timedelta(hours=1)
        recent_metrics = [
            m for m in metrics_history
            if m["timestamp"] > one_hour_ago
        ]
        
        if recent_metrics:
            cpu_values = [m["system_metrics"]["cpu_usage"] for m in recent_metrics]
            memory_values = [m["system_metrics"]["memory_usage"] for m in recent_metrics]
            
            performance_stats["recent_performance"] = {
                "avg_cpu_usage": round(sum(cpu_values) / len(cpu_values), 2),
                "max_cpu_usage": round(max(cpu_values), 2),
                "avg_memory_usage": round(sum(memory_values) / len(memory_values), 2),
                "max_memory_usage": round(max(memory_values), 2),
                "sample_count": len(recent_metrics)
            }
        
        return create_success_response(
            data=performance_stats,
            message="成功取得效能統計資訊"
        )
        
    except Exception as e:
        logger.error(f"取得效能統計失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取得效能統計失敗: {str(e)}"
        )


@router.post("/monitoring/test", response_model=DataResponse, summary="執行系統測試")
async def run_system_test(
    test_type: str = "basic",
    settings: Settings = Depends()
) -> DataResponse:
    """執行系統測試以驗證各項功能。
    
    Args:
        test_type: 測試類型（basic, comprehensive, performance）
        settings: 應用程式設定
        
    Returns:
        測試結果
    """
    try:
        test_results = {}
        start_time = time.time()
        
        if test_type == "basic":
            test_results = await _run_basic_tests()
        elif test_type == "comprehensive":
            test_results = await _run_comprehensive_tests()
        elif test_type == "performance":
            test_results = await _run_performance_tests()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支援的測試類型: {test_type}"
            )
        
        test_duration = time.time() - start_time
        
        # 計算測試總結
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results.values() if r.get("status") == "passed")
        failed_tests = total_tests - passed_tests
        
        overall_status = "passed" if failed_tests == 0 else "failed"
        
        result = {
            "test_type": test_type,
            "overall_status": overall_status,
            "duration_seconds": round(test_duration, 3),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0
            },
            "test_results": test_results
        }
        
        logger.info(
            f"系統測試完成: {test_type}, "
            f"總計 {total_tests} 個測試，"
            f"通過 {passed_tests} 個，失敗 {failed_tests} 個"
        )
        
        return create_success_response(
            data=result,
            message=f"系統測試完成：{overall_status}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"執行系統測試失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"執行系統測試失敗: {str(e)}"
        )


async def _collect_system_metrics() -> SystemMetrics:
    """收集系統資源指標。
    
    Returns:
        系統指標物件
    """
    # CPU 使用率
    cpu_usage = psutil.cpu_percent(interval=1)
    
    # 記憶體使用率
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    
    # 磁碟使用率
    disk = psutil.disk_usage('/')
    disk_usage = disk.percent
    
    # 網路 I/O
    network_io = psutil.net_io_counters()
    network_stats = {
        "bytes_sent": network_io.bytes_sent,
        "bytes_recv": network_io.bytes_recv,
        "packets_sent": network_io.packets_sent,
        "packets_recv": network_io.packets_recv
    }
    
    # 程序數量
    process_count = len(psutil.pids())
    
    return SystemMetrics(
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        disk_usage=disk_usage,
        network_io=network_stats,
        process_count=process_count
    )


def _collect_application_metrics() -> ApplicationMetrics:
    """收集應用程式指標。
    
    Returns:
        應用程式指標物件
    """
    # 計算平均回應時間
    avg_response_time = 0.0
    if application_stats["successful_queries"] > 0:
        avg_response_time = (
            application_stats["total_response_time"] / 
            application_stats["successful_queries"]
        )
    
    return ApplicationMetrics(
        total_queries=application_stats["total_queries"],
        successful_queries=application_stats["successful_queries"],
        failed_queries=application_stats["failed_queries"],
        average_response_time=avg_response_time,
        active_connections=application_stats["active_connections"],
        index_count=application_stats["index_count"],
        last_index_time=application_stats["last_index_time"]
    )


async def _check_system_alerts() -> List[Dict[str, Any]]:
    """檢查系統警告。
    
    Returns:
        警告列表
    """
    alerts = []
    current_time = datetime.now()
    
    # 檢查 CPU 使用率
    cpu_usage = psutil.cpu_percent()
    if cpu_usage > 90:
        alerts.append({
            "id": f"cpu_high_{int(current_time.timestamp())}",
            "type": "resource",
            "severity": "critical",
            "title": "CPU 使用率過高",
            "description": f"CPU 使用率達到 {cpu_usage:.1f}%",
            "timestamp": current_time,
            "value": cpu_usage,
            "threshold": 90
        })
    elif cpu_usage > 80:
        alerts.append({
            "id": f"cpu_warning_{int(current_time.timestamp())}",
            "type": "resource",
            "severity": "warning",
            "title": "CPU 使用率較高",
            "description": f"CPU 使用率達到 {cpu_usage:.1f}%",
            "timestamp": current_time,
            "value": cpu_usage,
            "threshold": 80
        })
    
    # 檢查記憶體使用率
    memory = psutil.virtual_memory()
    if memory.percent > 95:
        alerts.append({
            "id": f"memory_critical_{int(current_time.timestamp())}",
            "type": "resource",
            "severity": "critical",
            "title": "記憶體使用率極高",
            "description": f"記憶體使用率達到 {memory.percent:.1f}%",
            "timestamp": current_time,
            "value": memory.percent,
            "threshold": 95
        })
    elif memory.percent > 85:
        alerts.append({
            "id": f"memory_warning_{int(current_time.timestamp())}",
            "type": "resource",
            "severity": "warning",
            "title": "記憶體使用率較高",
            "description": f"記憶體使用率達到 {memory.percent:.1f}%",
            "timestamp": current_time,
            "value": memory.percent,
            "threshold": 85
        })
    
    # 檢查磁碟使用率
    disk = psutil.disk_usage('/')
    if disk.percent > 95:
        alerts.append({
            "id": f"disk_critical_{int(current_time.timestamp())}",
            "type": "resource",
            "severity": "critical",
            "title": "磁碟空間不足",
            "description": f"磁碟使用率達到 {disk.percent:.1f}%",
            "timestamp": current_time,
            "value": disk.percent,
            "threshold": 95
        })
    elif disk.percent > 85:
        alerts.append({
            "id": f"disk_warning_{int(current_time.timestamp())}",
            "type": "resource",
            "severity": "warning",
            "title": "磁碟空間偏低",
            "description": f"磁碟使用率達到 {disk.percent:.1f}%",
            "timestamp": current_time,
            "value": disk.percent,
            "threshold": 85
        })
    
    # 檢查失敗率
    total_queries = application_stats["total_queries"]
    if total_queries > 0:
        failure_rate = (application_stats["failed_queries"] / total_queries) * 100
        if failure_rate > 50:
            alerts.append({
                "id": f"failure_rate_high_{int(current_time.timestamp())}",
                "type": "application",
                "severity": "critical",
                "title": "查詢失敗率過高",
                "description": f"查詢失敗率達到 {failure_rate:.1f}%",
                "timestamp": current_time,
                "value": failure_rate,
                "threshold": 50
            })
        elif failure_rate > 20:
            alerts.append({
                "id": f"failure_rate_warning_{int(current_time.timestamp())}",
                "type": "application",
                "severity": "warning",
                "title": "查詢失敗率較高",
                "description": f"查詢失敗率達到 {failure_rate:.1f}%",
                "timestamp": current_time,
                "value": failure_rate,
                "threshold": 20
            })
    
    return alerts


async def _run_basic_tests() -> Dict[str, Any]:
    """執行基本系統測試。
    
    Returns:
        測試結果字典
    """
    results = {}
    
    # 測試系統資源
    try:
        psutil.cpu_percent()
        psutil.virtual_memory()
        psutil.disk_usage('/')
        results["system_resources"] = {
            "status": "passed",
            "message": "系統資源檢查正常"
        }
    except Exception as e:
        results["system_resources"] = {
            "status": "failed",
            "message": f"系統資源檢查失敗: {str(e)}"
        }
    
    # 測試日誌系統
    try:
        logger.info("測試日誌系統")
        results["logging"] = {
            "status": "passed",
            "message": "日誌系統工作正常"
        }
    except Exception as e:
        results["logging"] = {
            "status": "failed",
            "message": f"日誌系統測試失敗: {str(e)}"
        }
    
    return results


async def _run_comprehensive_tests() -> Dict[str, Any]:
    """執行全面系統測試。
    
    Returns:
        測試結果字典
    """
    results = await _run_basic_tests()
    
    # TODO: 新增更多全面測試
    results["database_connection"] = {
        "status": "passed",
        "message": "資料庫連接測試通過（模擬）"
    }
    
    results["embedding_service"] = {
        "status": "passed",
        "message": "Embedding 服務測試通過（模擬）"
    }
    
    return results


async def _run_performance_tests() -> Dict[str, Any]:
    """執行效能測試。
    
    Returns:
        測試結果字典
    """
    results = {}
    
    # 測試回應時間
    start_time = time.time()
    # 模擬一些工作
    await _collect_system_metrics()
    response_time = time.time() - start_time
    
    results["response_time"] = {
        "status": "passed" if response_time < 1.0 else "failed",
        "message": f"回應時間: {response_time:.3f}s",
        "value": response_time,
        "threshold": 1.0
    }
    
    return results


def _format_uptime(uptime_seconds: float) -> str:
    """格式化運行時間。
    
    Args:
        uptime_seconds: 運行時間（秒）
        
    Returns:
        格式化後的運行時間字串
    """
    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    
    if days > 0:
        return f"{days}天 {hours}小時 {minutes}分鐘"
    elif hours > 0:
        return f"{hours}小時 {minutes}分鐘"
    elif minutes > 0:
        return f"{minutes}分鐘 {seconds}秒"
    else:
        return f"{seconds}秒"


# 工具函數：更新應用統計資料
def update_query_stats(success: bool, response_time: float):
    """更新查詢統計資料。
    
    Args:
        success: 查詢是否成功
        response_time: 回應時間
    """
    application_stats["total_queries"] += 1
    if success:
        application_stats["successful_queries"] += 1
        application_stats["total_response_time"] += response_time
    else:
        application_stats["failed_queries"] += 1


def update_connection_count(delta: int):
    """更新活躍連接數。
    
    Args:
        delta: 連接數變化量
    """
    application_stats["active_connections"] = max(0, application_stats["active_connections"] + delta)


def update_index_stats(count: int):
    """更新索引統計資料。
    
    Args:
        count: 索引數量
    """
    application_stats["index_count"] = count
    application_stats["last_index_time"] = datetime.now()