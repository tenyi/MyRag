#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
健康檢查 API 路由

提供系統健康狀態檢查功能。
"""

import time
import platform
import psutil
from typing import Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...config.settings import Settings
from ...monitoring.logger import get_logger


# 設定日誌
logger = get_logger(__name__)

# 創建路由器
router = APIRouter()


class HealthResponse(BaseModel):
    """健康檢查回應模型。"""
    status: str
    timestamp: float
    version: str
    system_info: Dict[str, Any]
    dependencies: Dict[str, str]
    performance: Dict[str, Any]


class DetailedHealthResponse(BaseModel):
    """詳細健康檢查回應模型。"""
    status: str
    timestamp: float
    version: str
    system_info: Dict[str, Any]
    dependencies: Dict[str, str]
    performance: Dict[str, Any]
    services: Dict[str, Any]
    configuration: Dict[str, Any]


@router.get("/health", response_model=HealthResponse, summary="系統健康檢查")
async def health_check() -> HealthResponse:
    """基本健康檢查端點。
    
    檢查系統基本狀態，適用於負載均衡器健康檢查。
    
    Returns:
        系統健康狀態資訊
    """
    try:
        # 取得系統資訊
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "hostname": platform.node()
        }
        
        # 取得效能指標
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        performance = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        # 檢查依賴項
        dependencies = await check_dependencies()
        
        # 判斷整體健康狀態
        status = "healthy"
        if cpu_percent > 90 or memory.percent > 90:
            status = "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=time.time(),
            version="1.0.0",
            system_info=system_info,
            dependencies=dependencies,
            performance=performance
        )
        
    except Exception as e:
        logger.error(f"健康檢查失敗: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            version="1.0.0",
            system_info={},
            dependencies={},
            performance={}
        )


@router.get("/health/detailed", response_model=DetailedHealthResponse, summary="詳細健康檢查")
async def detailed_health_check(settings: Settings = Depends()) -> DetailedHealthResponse:
    """詳細健康檢查端點。
    
    提供系統完整的健康狀態資訊，包括服務狀態和配置資訊。
    
    Args:
        settings: 應用程式設定
        
    Returns:
        詳細的系統健康狀態資訊
    """
    try:
        # 基本健康檢查
        basic_health = await health_check()
        
        # 檢查服務狀態
        services = await check_services(settings)
        
        # 檢查配置
        configuration = check_configuration(settings)
        
        return DetailedHealthResponse(
            status=basic_health.status,
            timestamp=basic_health.timestamp,
            version=basic_health.version,
            system_info=basic_health.system_info,
            dependencies=basic_health.dependencies,
            performance=basic_health.performance,
            services=services,
            configuration=configuration
        )
        
    except Exception as e:
        logger.error(f"詳細健康檢查失敗: {e}")
        return DetailedHealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            version="1.0.0",
            system_info={},
            dependencies={},
            performance={},
            services={},
            configuration={}
        )


@router.get("/health/live", summary="存活檢查")
async def liveness_check() -> Dict[str, Any]:
    """存活檢查端點。
    
    簡單的存活檢查，回傳最小化的狀態資訊。
    適用於 Kubernetes liveness probe。
    
    Returns:
        存活狀態
    """
    return {
        "status": "alive",
        "timestamp": time.time()
    }


@router.get("/health/ready", summary="就緒檢查")
async def readiness_check() -> Dict[str, Any]:
    """就緒檢查端點。
    
    檢查系統是否已準備好處理請求。
    適用於 Kubernetes readiness probe。
    
    Returns:
        就緒狀態
    """
    try:
        # 檢查關鍵依賴項
        dependencies = await check_dependencies()
        
        # 檢查是否所有關鍵服務都正常
        critical_services = ["database", "vector_store", "embedding_service"]
        all_ready = all(
            dependencies.get(service, "unknown") == "healthy" 
            for service in critical_services
        )
        
        status = "ready" if all_ready else "not_ready"
        
        return {
            "status": status,
            "timestamp": time.time(),
            "dependencies": dependencies
        }
        
    except Exception as e:
        logger.error(f"就緒檢查失敗: {e}")
        return {
            "status": "not_ready",
            "timestamp": time.time(),
            "error": str(e)
        }


async def check_dependencies() -> Dict[str, str]:
    """檢查系統依賴項狀態。
    
    Returns:
        依賴項狀態字典
    """
    dependencies = {}
    
    try:
        # 檢查資料庫連接
        # TODO: 實作實際的資料庫連接檢查
        dependencies["database"] = "healthy"
        
        # 檢查向量資料庫
        # TODO: 實作向量資料庫連接檢查
        dependencies["vector_store"] = "healthy"
        
        # 檢查 embedding 服務
        # TODO: 實作 embedding 服務檢查
        dependencies["embedding_service"] = "healthy"
        
        # 檢查 LLM 服務
        # TODO: 實作 LLM 服務檢查
        dependencies["llm_service"] = "healthy"
        
    except Exception as e:
        logger.error(f"依賴項檢查失敗: {e}")
        dependencies["error"] = str(e)
        
    return dependencies


async def check_services(settings: Settings) -> Dict[str, Any]:
    """檢查各項服務狀態。
    
    Args:
        settings: 應用程式設定
        
    Returns:
        服務狀態字典
    """
    services = {}
    
    try:
        # 檢查索引服務
        services["indexing"] = {
            "status": "operational",
            "last_index_time": None,  # TODO: 實作實際檢查
            "index_count": 0  # TODO: 實作實際檢查
        }
        
        # 檢查查詢服務
        services["query"] = {
            "status": "operational",
            "last_query_time": None,  # TODO: 實作實際檢查
            "query_count": 0  # TODO: 實作實際檢查
        }
        
        # 檢查監控服務
        services["monitoring"] = {
            "status": "operational",
            "metrics_collected": True,  # TODO: 實作實際檢查
            "alerts_active": 0  # TODO: 實作實際檢查
        }
        
    except Exception as e:
        logger.error(f"服務檢查失敗: {e}")
        services["error"] = str(e)
        
    return services


def check_configuration(settings: Settings) -> Dict[str, Any]:
    """檢查系統配置狀態。
    
    Args:
        settings: 應用程式設定
        
    Returns:
        配置狀態字典
    """
    try:
        return {
            "config_loaded": True,
            "config_valid": True,  # TODO: 實作配置驗證
            "environment": getattr(settings, 'environment', 'development'),
            "debug_mode": getattr(settings, 'debug', False),
            "log_level": getattr(settings, 'log_level', 'INFO')
        }
    except Exception as e:
        logger.error(f"配置檢查失敗: {e}")
        return {
            "config_loaded": False,
            "error": str(e)
        }