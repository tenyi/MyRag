#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI 應用程式主檔案

提供 Chinese GraphRAG 系統的 REST API 介面。
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from ..config.settings import Settings
from ..monitoring.logger import get_logger
from . import API_PREFIX, API_VERSION

# 設定日誌
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理。

    Args:
        app: FastAPI 應用程式實例
    """
    # 應用程式啟動
    logger.info("Chinese GraphRAG API 正在啟動...")

    try:
        # TODO: 初始化服務（資料庫連接、模型載入等）
        logger.info("正在初始化服務...")

        # 初始化配置
        settings = Settings()
        app.state.settings = settings

        # 初始化其他服務
        # await initialize_services(settings)

        logger.info("Chinese GraphRAG API 啟動完成")

        yield

    except Exception as e:
        logger.error(f"API 啟動失敗: {e}")
        raise
    finally:
        # 應用程式關閉
        logger.info("Chinese GraphRAG API 正在關閉...")

        # TODO: 清理資源
        logger.info("正在清理資源...")

        logger.info("Chinese GraphRAG API 已關閉")


def create_app() -> FastAPI:
    """創建 FastAPI 應用程式實例。

    Returns:
        配置完成的 FastAPI 應用程式
    """
    app = FastAPI(
        title="Chinese GraphRAG API",
        description="針對中文文件優化的知識圖譜檢索增強生成系統 API",
        version=API_VERSION,
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=f"{API_PREFIX}/redoc",
        openapi_url=f"{API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # 新增中介軟體
    setup_middleware(app)

    # 註冊路由
    register_routes(app)

    # 設定例外處理器
    setup_exception_handlers(app)

    return app


def setup_middleware(app: FastAPI):
    """設定中介軟體。

    Args:
        app: FastAPI 應用程式實例
    """
    # CORS 中介軟體
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生產環境中應該限制特定域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 信任主機中介軟體
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["*"]  # 生產環境中應該限制特定主機
    )

    # 請求處理時間中介軟體
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """新增請求處理時間標頭。"""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # 請求日誌中介軟體
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """記錄請求資訊。"""
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            f"請求開始: {request.method} {request.url.path} " f"來自 {client_ip}"
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            logger.info(
                f"請求完成: {request.method} {request.url.path} "
                f"狀態 {response.status_code} "
                f"處理時間 {process_time:.3f}s"
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"請求錯誤: {request.method} {request.url.path} "
                f"錯誤 {str(e)} "
                f"處理時間 {process_time:.3f}s"
            )
            raise


def register_routes(app: FastAPI):
    """註冊 API 路由。

    Args:
        app: FastAPI 應用程式實例
    """
    # 匯入並註冊各個路由模組
    from .routes import config, health, index, monitoring, query

    # 健康檢查路由（不加前綴，方便負載均衡器檢查）
    app.include_router(health.router)

    # API 路由（加上前綴）
    app.include_router(index.router, prefix=API_PREFIX, tags=["indexing"])
    app.include_router(query.router, prefix=API_PREFIX, tags=["query"])
    app.include_router(config.router, prefix=API_PREFIX, tags=["configuration"])
    app.include_router(monitoring.router, prefix=API_PREFIX, tags=["monitoring"])


def setup_exception_handlers(app: FastAPI):
    """設定例外處理器。

    Args:
        app: FastAPI 應用程式實例
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 例外處理器。"""
        logger.warning(
            f"HTTP 例外: {exc.status_code} {exc.detail} " f"路徑: {request.url.path}"
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "type": "http_error",
                },
                "success": False,
                "timestamp": time.time(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """一般例外處理器。"""
        logger.error(
            f"未處理的例外: {str(exc)} " f"路徑: {request.url.path}", exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "內部伺服器錯誤",
                    "type": "internal_error",
                    "detail": str(exc) if app.debug else None,
                },
                "success": False,
                "timestamp": time.time(),
            },
        )


def get_settings() -> Settings:
    """取得應用程式設定的依賴項。

    Returns:
        應用程式設定實例
    """
    # 這個函數將作為 FastAPI 的依賴項使用
    return Settings()


# 自訂 OpenAPI 規格
def custom_openapi(app: FastAPI):
    """自訂 OpenAPI 規格。

    Args:
        app: FastAPI 應用程式實例

    Returns:
        自訂的 OpenAPI 規格
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Chinese GraphRAG API",
        version=API_VERSION,
        description="""
        ## 關於
        
        Chinese GraphRAG API 是一個針對中文文件優化的知識圖譜檢索增強生成系統的 REST API 介面。
        
        ## 功能特色
        
        - 🔍 **智慧查詢**: 支援自然語言查詢和結構化查詢
        - 📚 **文件索引**: 批次處理中文文件並建立知識圖譜
        - 🧠 **中文優化**: 專門針對中文語言特性進行優化
        - 📊 **監控管理**: 提供系統狀態監控和配置管理
        - 🚀 **高效能**: 支援並行處理和快取機制
        
        ## 使用方式
        
        1. 使用 `/health` 端點檢查系統狀態
        2. 使用 `/index` 端點建立文件索引
        3. 使用 `/query` 端點執行查詢
        4. 使用 `/config` 端點管理系統配置
        5. 使用 `/monitoring` 端點查看系統指標
        
        ## 認證
        
        目前 API 不需要認證，但建議在生產環境中啟用適當的認證機制。
        """,
        routes=app.routes,
    )

    # 新增自訂資訊
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# 創建應用程式實例
app = create_app()

# 設定自訂 OpenAPI
app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]


if __name__ == "__main__":
    import uvicorn

    # 開發模式啟動
    uvicorn.run(
        "chinese_graphrag.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
