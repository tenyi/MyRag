# API 資料模型參考

> 生成時間：2025-08-01 21:57:27 (Asia/Taipei)
> 基於 Chinese GraphRAG API v1.3.0

## 概述

本文件詳細描述 Chinese GraphRAG API 中使用的所有資料模型，包括請求模型、回應模型和內部資料結構。

## 基礎模型

### BaseModel
所有模型的基礎類別，提供共同的功能和驗證機制。

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class BaseModel(BaseModel):
    """基礎模型類別"""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        # 允許使用 ORM 模式
        from_attributes = True
        # 序列化時包含預設值
        use_enum_values = True
        # 驗證賦值
        validate_assignment = True
```

## 回應狀態

### ResponseStatus
定義 API 回應的狀態類型。

```python
from enum import Enum

class ResponseStatus(str, Enum):
    """回應狀態枚舉"""
    SUCCESS = "success"      # 成功
    ERROR = "error"          # 錯誤
    PENDING = "pending"      # 處理中
    PARTIAL = "partial"      # 部分成功
    CANCELLED = "cancelled"  # 已取消
```

## 基礎回應模型

### BaseResponse
所有 API 回應的基礎模型。

```python
class BaseResponse(BaseModel):
    """基礎回應模型"""
    status: ResponseStatus = Field(..., description="回應狀態")
    message: str = Field(..., description="回應訊息")
    timestamp: datetime = Field(default_factory=datetime.now, description="時間戳")
    request_id: Optional[str] = Field(None, description="請求 ID")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "操作成功完成",
                "timestamp": "2025-08-01T21:57:27+08:00",
                "request_id": "req_123456789"
            }
        }
```

### ErrorResponse
錯誤回應模型。

```python
class ErrorResponse(BaseResponse):
    """錯誤回應模型"""
    error_code: str = Field(..., description="錯誤代碼")
    error_details: Optional[Dict[str, Any]] = Field(None, description="錯誤詳情")
    suggestions: Optional[List[str]] = Field(None, description="解決建議")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "message": "查詢處理失敗",
                "timestamp": "2025-08-01T21:57:27+08:00",
                "request_id": "req_123456789",
                "error_code": "QUERY_PROCESSING_ERROR",
                "error_details": {
                    "query": "無效的查詢語法",
                    "line": 1,
                    "column": 15
                },
                "suggestions": [
                    "檢查查詢語法是否正確",
                    "確認查詢參數格式",
                    "參考 API 文件中的查詢範例"
                ]
            }
        }
```

### DataResponse
資料回應模型，用於返回結構化資料。

```python
class DataResponse(BaseResponse):
    """資料回應模型"""
    data: Any = Field(..., description="回應資料")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "資料取得成功",
                "timestamp": "2025-08-01T21:57:27+08:00",
                "data": {
                    "items": [],
                    "total": 0,
                    "page": 1,
                    "per_page": 10
                }
            }
        }
```

### PaginatedResponse
分頁回應模型。

```python
class PaginatedResponse(BaseResponse):
    """分頁回應模型"""
    data: List[Any] = Field(..., description="資料列表")
    total: int = Field(..., description="總數量")
    page: int = Field(..., description="當前頁碼")
    per_page: int = Field(..., description="每頁數量")
    total_pages: int = Field(..., description="總頁數")
    has_next: bool = Field(..., description="是否有下一頁")
    has_prev: bool = Field(..., description="是否有上一頁")
```

## 索引相關模型

### IndexRequest
索引請求模型。

```python
class IndexRequest(BaseModel):
    """索引請求模型"""
    input_path: str = Field(..., description="輸入檔案或目錄路徑")
    output_path: Optional[str] = Field(None, description="輸出目錄路徑")
    file_types: Optional[List[str]] = Field(None, description="檔案類型過濾")
    batch_size: Optional[int] = Field(32, description="批次大小")
    force_rebuild: Optional[bool] = Field(False, description="是否強制重建索引")
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """驗證批次大小"""
        if v is not None and v <= 0:
            raise ValueError('批次大小必須大於 0')
        return v
    
    @validator('file_types')
    def validate_file_types(cls, v):
        """驗證檔案類型"""
        if v is not None:
            allowed_types = ['txt', 'md', 'pdf', 'docx', 'html']
            for file_type in v:
                if file_type.lower() not in allowed_types:
                    raise ValueError(f'不支援的檔案類型: {file_type}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "input_path": "./documents",
                "output_path": "./data/index",
                "file_types": ["txt", "md", "pdf"],
                "batch_size": 32,
                "force_rebuild": false
            }
        }
```

### IndexStatus
索引狀態枚舉。

```python
class IndexStatus(str, Enum):
    """索引狀態枚舉"""
    PENDING = "pending"          # 等待中
    PROCESSING = "processing"    # 處理中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 失敗
    CANCELLED = "cancelled"     # 已取消
    PAUSED = "paused"          # 已暫停
```

### IndexResponse
索引回應模型。

```python
class IndexResponse(BaseResponse):
    """索引回應模型"""
    task_id: str = Field(..., description="任務 ID")
    status: IndexStatus = Field(..., description="索引狀態")
    progress: float = Field(..., description="進度百分比 (0-100)")
    processed_files: int = Field(..., description="已處理檔案數")
    total_files: int = Field(..., description="總檔案數")
    failed_files: int = Field(0, description="失敗檔案數")
    estimated_completion: Optional[datetime] = Field(None, description="預估完成時間")
    processing_speed: Optional[float] = Field(None, description="處理速度 (檔案/秒)")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "索引處理中",
                "timestamp": "2025-08-01T21:57:27+08:00",
                "task_id": "idx_123456789",
                "status": "processing",
                "progress": 65.5,
                "processed_files": 131,
                "total_files": 200,
                "failed_files": 2,
                "estimated_completion": "2025-08-01T22:15:00+08:00",
                "processing_speed": 2.3
            }
        }
```

## 查詢相關模型

### QueryRequest
查詢請求模型。

```python
class QueryRequest(BaseModel):
    """查詢請求模型"""
    query: str = Field(..., description="查詢內容", min_length=1, max_length=1000)
    query_type: Optional[str] = Field("auto", description="查詢類型")
    max_results: Optional[int] = Field(10, description="最大結果數")
    include_sources: Optional[bool] = Field(True, description="是否包含來源資訊")
    filters: Optional[Dict[str, Any]] = Field(None, description="查詢過濾條件")
    
    @validator('query_type')
    def validate_query_type(cls, v):
        """驗證查詢類型"""
        allowed_types = ['auto', 'semantic', 'keyword', 'graph', 'hybrid']
        if v not in allowed_types:
            raise ValueError(f'不支援的查詢類型: {v}')
        return v
    
    @validator('max_results')
    def validate_max_results(cls, v):
        """驗證最大結果數"""
        if v is not None and not (1 <= v <= 100):
            raise ValueError('最大結果數必須在 1-100 之間')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "什麼是人工智慧？",
                "query_type": "semantic",
                "max_results": 10,
                "include_sources": true,
                "filters": {
                    "document_type": "academic",
                    "date_range": {
                        "start": "2023-01-01",
                        "end": "2024-12-31"
                    }
                }
            }
        }
```

### QueryResult
查詢結果模型。

```python
class QueryResult(BaseModel):
    """查詢結果模型"""
    content: str = Field(..., description="結果內容")
    score: float = Field(..., description="相關性分數")
    source: Optional[str] = Field(None, description="來源檔案")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元資料")
    entities: Optional[List[str]] = Field(None, description="相關實體")
    relationships: Optional[List[str]] = Field(None, description="相關關係")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "人工智慧是一門研究如何讓機器模擬人類智慧的科學...",
                "score": 0.95,
                "source": "documents/ai_introduction.pdf",
                "metadata": {
                    "page": 1,
                    "chapter": "第一章 人工智慧概述",
                    "author": "張三",
                    "publish_date": "2024-01-15"
                },
                "entities": ["人工智慧", "機器學習", "深度學習"],
                "relationships": ["人工智慧-包含-機器學習", "機器學習-包含-深度學習"]
            }
        }
```

### QueryResponse
查詢回應模型。

```python
class QueryResponse(BaseResponse):
    """查詢回應模型"""
    results: List[QueryResult] = Field(..., description="查詢結果列表")
    total_results: int = Field(..., description="總結果數")
    query_time: float = Field(..., description="查詢時間（秒）")
    query_type_used: str = Field(..., description="實際使用的查詢類型")
    suggestions: Optional[List[str]] = Field(None, description="相關查詢建議")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "查詢執行成功",
                "timestamp": "2025-08-01T21:57:27+08:00",
                "results": [],
                "total_results": 15,
                "query_time": 0.234,
                "query_type_used": "semantic",
                "suggestions": [
                    "機器學習的應用",
                    "深度學習技術",
                    "AI 發展歷史"
                ]
            }
        }
```

### BatchQueryRequest
批次查詢請求模型。

```python
class BatchQueryRequest(BaseModel):
    """批次查詢請求模型"""
    queries: List[str] = Field(..., description="查詢列表", min_items=1, max_items=100)
    query_type: Optional[str] = Field("auto", description="查詢類型")
    max_results_per_query: Optional[int] = Field(10, description="每個查詢的最大結果數")
    parallel_workers: Optional[int] = Field(None, description="並行工作者數量")
    
    @validator('queries')
    def validate_queries(cls, v):
        """驗證查詢列表"""
        for query in v:
            if not query.strip():
                raise ValueError('查詢內容不能為空')
            if len(query) > 1000:
                raise ValueError('單個查詢長度不能超過 1000 字元')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "queries": [
                    "什麼是人工智慧？",
                    "機器學習的主要算法有哪些？",
                    "深度學習在圖像識別中的應用"
                ],
                "query_type": "semantic",
                "max_results_per_query": 5,
                "parallel_workers": 3
            }
        }
```

### BatchQueryResponse
批次查詢回應模型。

```python
class BatchQueryResponse(BaseResponse):
    """批次查詢回應模型"""
    task_id: str = Field(..., description="批次任務 ID")
    total_queries: int = Field(..., description="總查詢數")
    completed_queries: int = Field(..., description="已完成查詢數")
    failed_queries: int = Field(..., description="失敗查詢數")
    results: Optional[List[QueryResponse]] = Field(None, description="查詢結果列表")
    estimated_completion: Optional[datetime] = Field(None, description="預估完成時間")
```

## 配置相關模型

### ConfigUpdateRequest
配置更新請求模型。

```python
class ConfigUpdateRequest(BaseModel):
    """配置更新請求模型"""
    section: Optional[str] = Field(None, description="配置區段")
    key: Optional[str] = Field(None, description="配置鍵")
    value: Any = Field(..., description="配置值")
    validate_only: Optional[bool] = Field(False, description="僅驗證不更新")
    
    class Config:
        schema_extra = {
            "example": {
                "section": "embedding",
                "key": "model_name",
                "value": "text2vec-large-chinese",
                "validate_only": false
            }
        }
```

### ConfigResponse
配置回應模型。

```python
class ConfigResponse(BaseResponse):
    """配置回應模型"""
    config: Dict[str, Any] = Field(..., description="配置資料")
    schema_version: str = Field(..., description="配置結構版本")
    last_modified: datetime = Field(..., description="最後修改時間")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "配置取得成功",
                "timestamp": "2025-08-01T21:57:27+08:00",
                "config": {
                    "embedding": {
                        "model_name": "text2vec-large-chinese",
                        "batch_size": 32
                    },
                    "vector_store": {
                        "type": "lancedb",
                        "path": "./data/vector_store"
                    }
                },
                "schema_version": "1.3.0",
                "last_modified": "2025-08-01T20:30:00+08:00"
            }
        }
```

## 監控相關模型

### SystemMetrics
系統指標模型。

```python
class SystemMetrics(BaseModel):
    """系統指標模型"""
    cpu_usage: float = Field(..., description="CPU 使用率 (%)")
    memory_usage: float = Field(..., description="記憶體使用率 (%)")
    disk_usage: float = Field(..., description="磁碟使用率 (%)")
    network_io: Dict[str, float] = Field(..., description="網路 I/O")
    uptime: float = Field(..., description="運行時間（秒）")
    
    class Config:
        schema_extra = {
            "example": {
                "cpu_usage": 45.2,
                "memory_usage": 68.7,
                "disk_usage": 23.1,
                "network_io": {
                    "bytes_sent": 1024000,
                    "bytes_recv": 2048000
                },
                "uptime": 86400
            }
        }
```

### ApplicationMetrics
應用程式指標模型。

```python
class ApplicationMetrics(BaseModel):
    """應用程式指標模型"""
    total_requests: int = Field(..., description="總請求數")
    successful_requests: int = Field(..., description="成功請求數")
    failed_requests: int = Field(..., description="失敗請求數")
    average_response_time: float = Field(..., description="平均響應時間（毫秒）")
    active_connections: int = Field(..., description="活躍連接數")
    cache_hit_rate: float = Field(..., description="快取命中率 (%)")
    
    class Config:
        schema_extra = {
            "example": {
                "total_requests": 15420,
                "successful_requests": 14892,
                "failed_requests": 528,
                "average_response_time": 234.5,
                "active_connections": 12,
                "cache_hit_rate": 87.3
            }
        }
```

### MonitoringResponse
監控回應模型。

```python
class MonitoringResponse(BaseResponse):
    """監控回應模型"""
    system_metrics: SystemMetrics = Field(..., description="系統指標")
    application_metrics: ApplicationMetrics = Field(..., description="應用程式指標")
    alerts: List[Dict[str, Any]] = Field(..., description="警告列表")
    health_status: str = Field(..., description="健康狀態")
```

## 健康檢查模型

### HealthResponse
健康檢查回應模型。

```python
class HealthResponse(BaseModel):
    """健康檢查回應模型"""
    status: str = Field(..., description="健康狀態")
    timestamp: datetime = Field(..., description="檢查時間")
    uptime: float = Field(..., description="運行時間（秒）")
    version: str = Field(..., description="版本號")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-08-01T21:57:27+08:00",
                "uptime": 86400,
                "version": "1.3.0"
            }
        }
```

### DetailedHealthResponse
詳細健康檢查回應模型。

```python
class DetailedHealthResponse(BaseModel):
    """詳細健康檢查回應模型"""
    status: str = Field(..., description="整體健康狀態")
    timestamp: datetime = Field(..., description="檢查時間")
    components: Dict[str, Dict[str, Any]] = Field(..., description="組件狀態")
    system_info: Dict[str, Any] = Field(..., description="系統資訊")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-08-01T21:57:27+08:00",
                "components": {
                    "database": {
                        "status": "healthy",
                        "response_time": 12.3,
                        "last_check": "2025-08-01T21:57:20+08:00"
                    },
                    "vector_store": {
                        "status": "healthy",
                        "response_time": 8.7,
                        "last_check": "2025-08-01T21:57:25+08:00"
                    }
                },
                "system_info": {
                    "python_version": "3.12.0",
                    "platform": "Linux-5.4.0-x86_64",
                    "memory_total": "16GB",
                    "cpu_count": 8
                }
            }
        }
```

## 任務相關模型

### TaskResponse
任務回應模型。

```python
class TaskResponse(BaseResponse):
    """任務回應模型"""
    task_id: str = Field(..., description="任務 ID")
    task_type: str = Field(..., description="任務類型")
    status: str = Field(..., description="任務狀態")
    progress: float = Field(..., description="進度百分比")
    created_at: datetime = Field(..., description="創建時間")
    started_at: Optional[datetime] = Field(None, description="開始時間")
    completed_at: Optional[datetime] = Field(None, description="完成時間")
    result: Optional[Any] = Field(None, description="任務結果")
    error: Optional[str] = Field(None, description="錯誤訊息")
```

## 驗證規則

### 通用驗證規則

1. **字串長度限制**:
   - 查詢內容: 1-1000 字元
   - 檔案路徑: 1-500 字元
   - 任務 ID: 固定格式

2. **數值範圍限制**:
   - 批次大小: 1-1000
   - 最大結果數: 1-100
   - 進度百分比: 0-100

3. **列舉值驗證**:
   - 查詢類型: auto, semantic, keyword, graph, hybrid
   - 檔案類型: txt, md, pdf, docx, html
   - 狀態值: 預定義的狀態枚舉

### 自定義驗證器

```python
from pydantic import validator
import re

class CustomValidators:
    @validator('email')
    def validate_email(cls, v):
        """驗證電子郵件格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError('無效的電子郵件格式')
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        """驗證電話號碼格式"""
        pattern = r'^\+?1?\d{9,15}$'
        if not re.match(pattern, v):
            raise ValueError('無效的電話號碼格式')
        return v
```

## 序列化配置

### JSON 序列化

```python
from pydantic import BaseModel
from datetime import datetime
import json

class CustomJSONEncoder(json.JSONEncoder):
    """自定義 JSON 編碼器"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# 使用範例
model = QueryResponse(...)
json_str = json.dumps(model.dict(), cls=CustomJSONEncoder, ensure_ascii=False)
```

### 模型配置

```python
class ModelConfig:
    """模型配置類別"""
    # 允許從 ORM 物件創建
    from_attributes = True
    
    # 使用枚舉值而非枚舉名稱
    use_enum_values = True
    
    # 驗證賦值
    validate_assignment = True
    
    # 允許額外欄位
    extra = "forbid"  # 或 "allow", "ignore"
    
    # JSON 編碼器
    json_encoders = {
        datetime: lambda v: v.isoformat()
    }
```

---

*此文件提供完整的 API 資料模型參考，包含所有欄位定義、驗證規則和使用範例。*