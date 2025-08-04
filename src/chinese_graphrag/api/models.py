#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API 回應模型

定義所有 API 端點的請求和回應模型。
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class ResponseStatus(str, Enum):
    """回應狀態列舉。"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    PROCESSING = "processing"


class BaseResponse(BaseModel):
    """基礎回應模型。"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field("", description="回應訊息")
    timestamp: datetime = Field(default_factory=datetime.now, description="回應時間戳")
    request_id: Optional[str] = Field(None, description="請求 ID")


class ErrorResponse(BaseResponse):
    """錯誤回應模型。"""
    success: bool = Field(False, description="操作失敗")
    error_code: Optional[str] = Field(None, description="錯誤代碼")
    error_details: Optional[Dict[str, Any]] = Field(None, description="錯誤詳情")
    
    
class DataResponse(BaseResponse):
    """資料回應模型。"""
    success: bool = Field(True, description="操作成功")
    data: Any = Field(..., description="回應資料")
    meta: Optional[Dict[str, Any]] = Field(None, description="元資料")


class PaginatedResponse(DataResponse):
    """分頁回應模型。"""
    pagination: Dict[str, Any] = Field(..., description="分頁資訊")


class TaskResponse(BaseResponse):
    """任務回應模型。"""
    task_id: str = Field(..., description="任務 ID")
    status: ResponseStatus = Field(..., description="任務狀態")
    progress: Optional[float] = Field(None, description="進度百分比 (0-100)")
    result: Optional[Any] = Field(None, description="任務結果")
    
    @validator('progress')
    def validate_progress(cls, v):
        """驗證進度百分比。"""
        if v is not None and not (0 <= v <= 100):
            raise ValueError('進度必須在 0-100 之間')
        return v


# 索引相關模型
class IndexRequest(BaseModel):
    """索引請求模型。"""
    input_path: str = Field(..., description="輸入檔案或目錄路徑")
    output_path: Optional[str] = Field(None, description="輸出目錄路徑")
    file_types: Optional[List[str]] = Field(None, description="檔案類型過濾")
    batch_size: Optional[int] = Field(32, description="批次大小")
    force_rebuild: Optional[bool] = Field(False, description="是否強制重建索引")
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """驗證批次大小。"""
        if v is not None and v <= 0:
            raise ValueError('批次大小必須大於 0')
        return v


class IndexStatus(BaseModel):
    """索引狀態模型。"""
    total_files: int = Field(..., description="總檔案數")
    processed_files: int = Field(..., description="已處理檔案數")
    failed_files: int = Field(0, description="失敗檔案數")
    current_file: Optional[str] = Field(None, description="當前處理檔案")
    estimated_remaining_time: Optional[float] = Field(None, description="預估剩餘時間（秒）")


class IndexResponse(TaskResponse):
    """索引回應模型。"""
    index_status: Optional[IndexStatus] = Field(None, description="索引狀態詳情")


# 查詢相關模型
class QueryRequest(BaseModel):
    """查詢請求模型。"""
    query: str = Field(..., description="查詢內容")
    query_type: Optional[str] = Field("auto", description="查詢類型")
    max_results: Optional[int] = Field(10, description="最大結果數")
    include_sources: Optional[bool] = Field(True, description="是否包含來源資訊")
    filters: Optional[Dict[str, Any]] = Field(None, description="查詢過濾條件")


class SimpleQueryRequest(BaseModel):
    """精簡查詢請求模型。"""
    query: str = Field(..., description="查詢內容")
    search_type: Optional[str] = Field("auto", description="搜尋類型 (auto/global/local)")
    use_llm_segmentation: Optional[bool] = Field(True, description="是否使用 LLM 分詞")
    
    @validator('max_results')
    def validate_max_results(cls, v):
        """驗證最大結果數。"""
        if v is not None and not (1 <= v <= 100):
            raise ValueError('最大結果數必須在 1-100 之間')
        return v


class QueryResult(BaseModel):
    """查詢結果模型。"""
    id: str = Field(..., description="結果 ID")
    content: str = Field(..., description="結果內容")
    score: float = Field(..., description="相關性分數")
    source: Optional[Dict[str, Any]] = Field(None, description="來源資訊")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元資料")


class QueryResponse(DataResponse):
    """查詢回應模型。"""
    data: List[QueryResult] = Field(..., description="查詢結果列表")
    query_info: Dict[str, Any] = Field(..., description="查詢資訊")


class SimpleQueryResponse(BaseResponse):
    """精簡查詢回應模型。"""
    answer: str = Field(..., description="查詢答案")
    confidence: float = Field(..., description="信心度 (0-1)")
    search_type: str = Field(..., description="使用的搜尋類型")
    response_time: float = Field(..., description="回應時間（秒）")
    reasoning_path: Optional[List[str]] = Field(None, description="推理路徑（可選）")


class BatchQueryRequest(BaseModel):
    """批次查詢請求模型。"""
    queries: List[str] = Field(..., description="查詢列表")
    query_type: Optional[str] = Field("auto", description="查詢類型")
    max_results: Optional[int] = Field(10, description="每個查詢的最大結果數")
    include_sources: Optional[bool] = Field(True, description="是否包含來源資訊")
    
    @validator('queries')
    def validate_queries(cls, v):
        """驗證查詢列表。"""
        if not v:
            raise ValueError('查詢列表不能為空')
        if len(v) > 50:
            raise ValueError('批次查詢數量不能超過 50')
        return v


class BatchQueryResponse(TaskResponse):
    """批次查詢回應模型。"""
    batch_results: Optional[List[QueryResponse]] = Field(None, description="批次查詢結果")


# 配置相關模型
class ConfigUpdateRequest(BaseModel):
    """配置更新請求模型。"""
    config_section: str = Field(..., description="配置區段")
    config_data: Dict[str, Any] = Field(..., description="配置資料")
    validate_only: Optional[bool] = Field(False, description="僅驗證不儲存")


class ConfigResponse(DataResponse):
    """配置回應模型。"""
    data: Dict[str, Any] = Field(..., description="配置資料")


# 監控相關模型
class SystemMetrics(BaseModel):
    """系統指標模型。"""
    cpu_usage: float = Field(..., description="CPU 使用率")
    memory_usage: float = Field(..., description="記憶體使用率")
    disk_usage: float = Field(..., description="磁碟使用率")
    network_io: Dict[str, float] = Field(..., description="網路 I/O")
    process_count: int = Field(..., description="程序數量")


class ApplicationMetrics(BaseModel):
    """應用指標模型。"""
    total_queries: int = Field(..., description="總查詢數")
    successful_queries: int = Field(..., description="成功查詢數")
    failed_queries: int = Field(..., description="失敗查詢數")
    average_response_time: float = Field(..., description="平均回應時間")
    active_connections: int = Field(..., description="活躍連接數")
    index_count: int = Field(..., description="索引數量")
    last_index_time: Optional[datetime] = Field(None, description="最後索引時間")


class MonitoringResponse(DataResponse):
    """監控回應模型。"""
    system_metrics: SystemMetrics = Field(..., description="系統指標")
    application_metrics: ApplicationMetrics = Field(..., description="應用指標")


# 通用工具函數
def create_success_response(
    data: Any = None,
    message: str = "操作成功",
    meta: Dict[str, Any] = None
) -> DataResponse:
    """創建成功回應。
    
    Args:
        data: 回應資料
        message: 回應訊息
        meta: 元資料
        
    Returns:
        成功的資料回應
    """
    return DataResponse(
        success=True,
        message=message,
        data=data,
        meta=meta
    )


def create_error_response(
    message: str,
    error_code: str = None,
    error_details: Dict[str, Any] = None
) -> ErrorResponse:
    """創建錯誤回應。
    
    Args:
        message: 錯誤訊息
        error_code: 錯誤代碼
        error_details: 錯誤詳情
        
    Returns:
        錯誤回應
    """
    return ErrorResponse(
        success=False,
        message=message,
        error_code=error_code,
        error_details=error_details
    )


def create_task_response(
    task_id: str,
    status: ResponseStatus,
    message: str = "",
    progress: float = None,
    result: Any = None
) -> TaskResponse:
    """創建任務回應。
    
    Args:
        task_id: 任務 ID
        status: 任務狀態
        message: 回應訊息
        progress: 進度百分比
        result: 任務結果
        
    Returns:
        任務回應
    """
    return TaskResponse(
        success=status != ResponseStatus.ERROR,
        message=message,
        task_id=task_id,
        status=status,
        progress=progress,
        result=result
    )