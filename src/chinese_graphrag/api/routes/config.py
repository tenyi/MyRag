#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理 API 路由

提供系統配置查看和更新功能。
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from ...config.settings import Settings
from ...monitoring.logger import get_logger
from ..models import (
    ConfigResponse,
    ConfigUpdateRequest,
    DataResponse,
    create_error_response,
    create_success_response,
)

# 設定日誌
logger = get_logger(__name__)

# 創建路由器
router = APIRouter()


@router.get("/config", response_model=ConfigResponse, summary="取得系統配置")
async def get_config(
    section: Optional[str] = None, settings: Settings = Depends()
) -> ConfigResponse:
    """取得系統配置資訊。

    Args:
        section: 配置區段名稱（可選），如果不指定則回傳所有配置
        settings: 應用程式設定

    Returns:
        系統配置資訊
    """
    try:
        # 取得所有配置
        config_data = settings.dict()

        # 過濾敏感資訊
        config_data = _filter_sensitive_data(config_data)

        # 如果指定了區段，只回傳該區段
        if section:
            if section not in config_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"找不到配置區段: {section}",
                )
            config_data = {section: config_data[section]}

        logger.info(f"取得配置資訊，區段: {section or '全部'}")

        return ConfigResponse(
            success=True, message=f"成功取得配置資訊", data=config_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取得配置資訊失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取得配置資訊失敗: {str(e)}",
        )


@router.put("/config", response_model=ConfigResponse, summary="更新系統配置")
async def update_config(
    request: ConfigUpdateRequest, settings: Settings = Depends()
) -> ConfigResponse:
    """更新系統配置。

    Args:
        request: 配置更新請求
        settings: 應用程式設定

    Returns:
        更新後的配置資訊
    """
    try:
        # 驗證配置區段
        if not request.config_section:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="必須指定配置區段"
            )

        # 驗證配置資料
        validation_result = _validate_config_data(
            request.config_section, request.config_data
        )

        if not validation_result["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"配置資料驗證失敗: {validation_result['error']}",
            )

        # 如果只是驗證，不實際更新
        if request.validate_only:
            logger.info(f"配置驗證通過: {request.config_section}")
            return ConfigResponse(
                success=True,
                message="配置驗證通過",
                data={
                    "section": request.config_section,
                    "validation_result": validation_result,
                    "validate_only": True,
                },
            )

        # TODO: 實作實際的配置更新邏輯
        # 這裡應該更新實際的配置檔案或資料庫
        logger.info(f"更新配置: {request.config_section}")

        # 模擬更新後的配置
        updated_config = {request.config_section: request.config_data}

        return ConfigResponse(
            success=True,
            message=f"成功更新配置區段: {request.config_section}",
            data=updated_config,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新配置失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新配置失敗: {str(e)}",
        )


@router.get("/config/schema", response_model=DataResponse, summary="取得配置結構描述")
async def get_config_schema() -> DataResponse:
    """取得系統配置的結構描述（Schema）。

    Returns:
        配置結構描述資訊
    """
    try:
        # 定義配置結構描述
        config_schema = {
            "embedding": {
                "description": "Embedding 服務配置",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Embedding 模型名稱",
                        "default": "BAAI/bge-m3",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "批次處理大小",
                        "default": 32,
                        "minimum": 1,
                        "maximum": 512,
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "最大序列長度",
                        "default": 512,
                        "minimum": 128,
                        "maximum": 2048,
                    },
                },
            },
            "vector_store": {
                "description": "向量資料庫配置",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "向量資料庫提供者",
                        "enum": ["lancedb", "chroma", "pinecone"],
                        "default": "lancedb",
                    },
                    "connection_string": {
                        "type": "string",
                        "description": "資料庫連接字串",
                        "sensitive": True,
                    },
                    "table_name": {
                        "type": "string",
                        "description": "資料表名稱",
                        "default": "documents",
                    },
                },
            },
            "llm": {
                "description": "大語言模型配置",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "LLM 提供者",
                        "enum": ["openai", "anthropic", "azure", "local"],
                        "default": "openai",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "模型名稱",
                        "default": "gpt-3.5-turbo",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API 金鑰",
                        "sensitive": True,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "生成溫度",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "最大生成 token 數",
                        "default": 1000,
                        "minimum": 1,
                        "maximum": 4096,
                    },
                },
            },
            "indexing": {
                "description": "索引建立配置",
                "properties": {
                    "batch_size": {
                        "type": "integer",
                        "description": "索引批次大小",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 1000,
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "文本分塊大小",
                        "default": 1000,
                        "minimum": 100,
                        "maximum": 5000,
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "分塊重疊大小",
                        "default": 200,
                        "minimum": 0,
                        "maximum": 1000,
                    },
                    "enable_ocr": {
                        "type": "boolean",
                        "description": "是否啟用 OCR",
                        "default": False,
                    },
                },
            },
            "monitoring": {
                "description": "監控和日誌配置",
                "properties": {
                    "log_level": {
                        "type": "string",
                        "description": "日誌級別",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        "default": "INFO",
                    },
                    "enable_metrics": {
                        "type": "boolean",
                        "description": "是否啟用指標收集",
                        "default": True,
                    },
                    "metrics_port": {
                        "type": "integer",
                        "description": "指標服務端口",
                        "default": 8001,
                        "minimum": 1024,
                        "maximum": 65535,
                    },
                },
            },
        }

        return create_success_response(
            data=config_schema, message="成功取得配置結構描述"
        )

    except Exception as e:
        logger.error(f"取得配置結構描述失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取得配置結構描述失敗: {str(e)}",
        )


@router.post("/config/validate", response_model=DataResponse, summary="驗證配置")
async def validate_config(request: ConfigUpdateRequest) -> DataResponse:
    """驗證配置資料的有效性。

    Args:
        request: 配置更新請求（僅用於驗證）

    Returns:
        驗證結果
    """
    try:
        validation_result = _validate_config_data(
            request.config_section, request.config_data
        )

        if validation_result["valid"]:
            return create_success_response(
                data=validation_result, message="配置驗證通過"
            )
        else:
            return create_error_response(
                message=f"配置驗證失敗: {validation_result['error']}",
                error_code="VALIDATION_FAILED",
                error_details=validation_result,
            )

    except Exception as e:
        logger.error(f"配置驗證失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"配置驗證失敗: {str(e)}",
        )


@router.post("/config/reset", response_model=ConfigResponse, summary="重設配置為預設值")
async def reset_config(
    section: Optional[str] = None, settings: Settings = Depends()
) -> ConfigResponse:
    """重設配置為預設值。

    Args:
        section: 要重設的配置區段（可選），如果不指定則重設所有配置
        settings: 應用程式設定

    Returns:
        重設後的配置資訊
    """
    try:
        # TODO: 實作實際的配置重設邏輯
        logger.info(f"重設配置，區段: {section or '全部'}")

        # 模擬預設配置
        default_config = _get_default_config()

        if section:
            if section not in default_config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"找不到配置區段: {section}",
                )
            reset_config_data = {section: default_config[section]}
        else:
            reset_config_data = default_config

        return ConfigResponse(
            success=True,
            message=f"成功重設配置{f'區段: {section}' if section else ''}",
            data=reset_config_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重設配置失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重設配置失敗: {str(e)}",
        )


def _filter_sensitive_data(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """過濾敏感配置資料。

    Args:
        config_data: 原始配置資料

    Returns:
        過濾後的配置資料
    """
    filtered_data = config_data.copy()

    # 定義需要過濾的敏感欄位
    sensitive_fields = [
        "api_key",
        "secret_key",
        "password",
        "token",
        "connection_string",
        "database_url",
    ]

    def filter_dict(data):
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in sensitive_fields):
                    result[key] = "***隱藏***" if value else None
                else:
                    result[key] = filter_dict(value)
            return result
        elif isinstance(data, list):
            return [filter_dict(item) for item in data]
        else:
            return data

    return filter_dict(filtered_data)


def _validate_config_data(section: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
    """驗證配置資料。

    Args:
        section: 配置區段
        config_data: 配置資料

    Returns:
        驗證結果
    """
    try:
        # TODO: 實作實際的配置驗證邏輯
        # 這裡應該根據配置結構描述進行詳細驗證

        validation_errors = []

        # 基本驗證
        if not config_data:
            validation_errors.append("配置資料不能為空")

        # 區段特定驗證
        if section == "embedding":
            if "batch_size" in config_data:
                batch_size = config_data["batch_size"]
                if (
                    not isinstance(batch_size, int)
                    or batch_size < 1
                    or batch_size > 512
                ):
                    validation_errors.append("batch_size 必須是 1-512 之間的整數")

        elif section == "llm":
            if "temperature" in config_data:
                temperature = config_data["temperature"]
                if (
                    not isinstance(temperature, (int, float))
                    or temperature < 0
                    or temperature > 2
                ):
                    validation_errors.append("temperature 必須是 0-2 之間的數值")

        # 回傳驗證結果
        if validation_errors:
            return {
                "valid": False,
                "error": "; ".join(validation_errors),
                "errors": validation_errors,
            }
        else:
            return {"valid": True, "message": "配置驗證通過"}

    except Exception as e:
        return {"valid": False, "error": f"驗證過程中發生錯誤: {str(e)}"}


def _get_default_config() -> Dict[str, Any]:
    """取得預設配置。

    Returns:
        預設配置資料
    """
    return {
        "embedding": {"model_name": "BAAI/bge-m3", "batch_size": 32, "max_length": 512},
        "vector_store": {"provider": "lancedb", "table_name": "documents"},
        "llm": {
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "indexing": {
            "batch_size": 100,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "enable_ocr": False,
        },
        "monitoring": {
            "log_level": "INFO",
            "enable_metrics": True,
            "metrics_port": 8001,
        },
    }
