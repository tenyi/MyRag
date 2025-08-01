# Chinese GraphRAG API 索引

> 生成時間：2025-08-01 21:57:27 (Asia/Taipei)
> 版本：基於當前程式碼結構

## 概述

Chinese GraphRAG API 是一個針對中文文件優化的知識圖譜檢索增強生成系統的 RESTful API。本文件提供完整的 API 端點索引和使用指南。

## API 基本資訊

- **基礎 URL**: `/api/v1`
- **文件 URL**: `/api/v1/docs` (Swagger UI)
- **ReDoc URL**: `/api/v1/redoc`
- **OpenAPI 規格**: `/api/v1/openapi.json`

## 端點分類

### 1. 健康檢查 (Health Check)

#### 基本健康檢查
- **端點**: `GET /health`
- **描述**: 系統健康檢查
- **回應模型**: `HealthResponse`
- **用途**: 負載均衡器和監控系統使用

#### 詳細健康檢查
- **端點**: `GET /health/detailed`
- **描述**: 詳細健康檢查，包含各組件狀態
- **回應模型**: `DetailedHealthResponse`
- **用途**: 系統診斷和故障排除

#### 存活檢查
- **端點**: `GET /health/live`
- **描述**: 存活檢查，確認服務是否運行
- **用途**: Kubernetes liveness probe

#### 就緒檢查
- **端點**: `GET /health/ready`
- **描述**: 就緒檢查，確認服務是否準備接受請求
- **用途**: Kubernetes readiness probe

### 2. 索引管理 (Indexing)

#### 建立索引
- **端點**: `POST /api/v1/index`
- **描述**: 開始文件索引處理
- **請求模型**: `IndexRequest`
- **回應模型**: `IndexResponse`
- **功能**: 
  - 支援多種文件格式 (PDF, DOCX, TXT, MD)
  - 中文文本處理和分詞
  - 實體提取和關係識別
  - 向量化和儲存

#### 查詢索引狀態
- **端點**: `GET /api/v1/index/{task_id}`
- **描述**: 查詢特定索引任務的進度
- **回應模型**: `IndexResponse`
- **參數**: 
  - `task_id`: 索引任務 ID

#### 列出索引任務
- **端點**: `GET /api/v1/index`
- **描述**: 列出所有索引任務
- **回應模型**: `DataResponse`
- **查詢參數**:
  - `status`: 過濾任務狀態
  - `limit`: 限制返回數量
  - `offset`: 分頁偏移

#### 取消索引任務
- **端點**: `DELETE /api/v1/index/{task_id}`
- **描述**: 取消正在進行的索引任務
- **回應模型**: `DataResponse`
- **參數**: 
  - `task_id`: 要取消的任務 ID

#### 清理索引任務
- **端點**: `DELETE /api/v1/index`
- **描述**: 清理已完成的索引任務記錄
- **回應模型**: `DataResponse`

### 3. 查詢服務 (Query)

#### 單一查詢
- **端點**: `POST /api/v1/query`
- **描述**: 執行單一查詢請求
- **請求模型**: `QueryRequest`
- **回應模型**: `QueryResponse`
- **支援的查詢類型**:
  - `semantic`: 語義搜索
  - `keyword`: 關鍵字搜索
  - `graph`: 圖譜搜索
  - `auto`: 自動選擇最佳搜索策略

#### 批次查詢
- **端點**: `POST /api/v1/query/batch`
- **描述**: 執行批次查詢處理
- **請求模型**: `BatchQueryRequest`
- **回應模型**: `BatchQueryResponse`
- **功能**: 
  - 支援大量查詢的批次處理
  - 異步處理機制
  - 進度追蹤

#### 查詢批次任務狀態
- **端點**: `GET /api/v1/query/batch/{task_id}`
- **描述**: 查詢批次任務的執行狀態
- **回應模型**: `BatchQueryResponse`
- **參數**: 
  - `task_id`: 批次任務 ID

#### 查詢歷史記錄
- **端點**: `GET /api/v1/query/history`
- **描述**: 取得查詢歷史記錄
- **回應模型**: `DataResponse`
- **查詢參數**:
  - `limit`: 限制返回數量
  - `offset`: 分頁偏移
  - `start_date`: 開始日期
  - `end_date`: 結束日期

#### 查詢建議
- **端點**: `GET /api/v1/query/suggestions`
- **描述**: 取得查詢建議和自動完成
- **回應模型**: `DataResponse`
- **查詢參數**:
  - `q`: 部分查詢文本
  - `limit`: 建議數量限制

### 4. 配置管理 (Configuration)

#### 取得系統配置
- **端點**: `GET /api/v1/config`
- **描述**: 取得當前系統配置
- **回應模型**: `ConfigResponse`
- **功能**: 
  - 過濾敏感資訊
  - 支援配置分組查看

#### 更新系統配置
- **端點**: `PUT /api/v1/config`
- **描述**: 更新系統配置
- **請求模型**: `ConfigUpdateRequest`
- **回應模型**: `ConfigResponse`
- **功能**: 
  - 配置驗證
  - 熱更新支援
  - 配置備份

#### 取得配置結構描述
- **端點**: `GET /api/v1/config/schema`
- **描述**: 取得配置結構的 JSON Schema
- **回應模型**: `DataResponse`
- **用途**: 前端配置界面生成

#### 驗證配置
- **端點**: `POST /api/v1/config/validate`
- **描述**: 驗證配置的正確性
- **回應模型**: `DataResponse`
- **功能**: 
  - 語法驗證
  - 語義驗證
  - 依賴檢查

#### 重設配置
- **端點**: `POST /api/v1/config/reset`
- **描述**: 重設配置為預設值
- **回應模型**: `ConfigResponse`
- **功能**: 
  - 安全重設機制
  - 配置備份

### 5. 監控服務 (Monitoring)

#### 取得系統指標
- **端點**: `GET /api/v1/monitoring/metrics`
- **描述**: 取得系統監控指標
- **回應模型**: `MonitoringResponse`
- **包含指標**:
  - CPU 使用率
  - 記憶體使用量
  - 磁碟使用量
  - 網路流量
  - 應用程式指標

#### 取得監控歷史
- **端點**: `GET /api/v1/monitoring/history`
- **描述**: 取得監控歷史資料
- **回應模型**: `DataResponse`
- **查詢參數**:
  - `metric`: 指標類型
  - `start_time`: 開始時間
  - `end_time`: 結束時間
  - `interval`: 時間間隔

#### 取得系統警告
- **端點**: `GET /api/v1/monitoring/alerts`
- **描述**: 取得系統警告資訊
- **回應模型**: `DataResponse`
- **功能**: 
  - 警告分級
  - 警告歷史
  - 警告統計

#### 取得效能統計
- **端點**: `GET /api/v1/monitoring/performance`
- **描述**: 取得效能統計資料
- **回應模型**: `DataResponse`
- **包含統計**:
  - 查詢響應時間
  - 索引處理速度
  - 錯誤率統計
  - 吞吐量指標

#### 執行系統測試
- **端點**: `POST /api/v1/monitoring/test`
- **描述**: 執行系統測試和診斷
- **回應模型**: `DataResponse`
- **測試類型**:
  - `basic`: 基本功能測試
  - `comprehensive`: 全面系統測試
  - `performance`: 效能測試

## 資料模型

### 請求模型

#### IndexRequest
```python
class IndexRequest(BaseModel):
    input_path: str  # 輸入文件路徑
    output_path: Optional[str] = None  # 輸出路徑
    file_types: Optional[List[str]] = None  # 文件類型過濾
    chunk_size: Optional[int] = None  # 分塊大小
    overlap_size: Optional[int] = None  # 重疊大小
    parallel_workers: Optional[int] = None  # 並行工作者數量
```

#### QueryRequest
```python
class QueryRequest(BaseModel):
    query: str  # 查詢文本
    query_type: Optional[str] = "auto"  # 查詢類型
    max_results: Optional[int] = 10  # 最大結果數
    include_sources: Optional[bool] = True  # 是否包含來源
    filters: Optional[Dict[str, Any]] = None  # 查詢過濾器
```

#### BatchQueryRequest
```python
class BatchQueryRequest(BaseModel):
    queries: List[str]  # 查詢列表
    query_type: Optional[str] = "auto"  # 查詢類型
    max_results_per_query: Optional[int] = 10  # 每個查詢的最大結果數
    parallel_workers: Optional[int] = None  # 並行工作者數量
```

### 回應模型

#### BaseResponse
```python
class BaseResponse(BaseModel):
    status: ResponseStatus  # 回應狀態
    message: str  # 回應訊息
    timestamp: datetime  # 時間戳
    request_id: Optional[str] = None  # 請求 ID
```

#### IndexResponse
```python
class IndexResponse(BaseResponse):
    task_id: str  # 任務 ID
    status: IndexStatus  # 索引狀態
    progress: float  # 進度百分比
    processed_files: int  # 已處理文件數
    total_files: int  # 總文件數
    estimated_completion: Optional[datetime] = None  # 預估完成時間
```

#### QueryResponse
```python
class QueryResponse(BaseResponse):
    results: List[QueryResult]  # 查詢結果
    total_results: int  # 總結果數
    query_time: float  # 查詢時間（秒）
    query_type_used: str  # 實際使用的查詢類型
```

## 錯誤處理

### HTTP 狀態碼

- `200 OK`: 請求成功
- `201 Created`: 資源創建成功
- `400 Bad Request`: 請求參數錯誤
- `401 Unauthorized`: 未授權
- `403 Forbidden`: 禁止訪問
- `404 Not Found`: 資源不存在
- `422 Unprocessable Entity`: 請求格式正確但語義錯誤
- `429 Too Many Requests`: 請求頻率過高
- `500 Internal Server Error`: 伺服器內部錯誤
- `503 Service Unavailable`: 服務不可用

### 錯誤回應格式

```python
class ErrorResponse(BaseResponse):
    error_code: str  # 錯誤代碼
    error_details: Optional[Dict[str, Any]] = None  # 錯誤詳情
    suggestions: Optional[List[str]] = None  # 解決建議
```

## 認證與授權

目前 API 支援以下認證方式：
- API Key 認證
- JWT Token 認證
- OAuth 2.0（規劃中）

## 速率限制

- 一般 API：每分鐘 100 次請求
- 索引 API：每分鐘 10 次請求
- 批次查詢：每分鐘 5 次請求

## 使用範例

### 建立索引
```bash
curl -X POST "http://localhost:8000/api/v1/index" \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "./documents",
    "chunk_size": 1000,
    "parallel_workers": 4
  }'
```

### 執行查詢
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什麼是人工智慧？",
    "query_type": "semantic",
    "max_results": 5
  }'
```

### 檢查健康狀態
```bash
curl -X GET "http://localhost:8000/health"
```

## 開發工具

### API 文件生成
```bash
# 生成 API 文件
uv run chinese-graphrag api docs

# 驗證 API
uv run chinese-graphrag api validate

# 效能測試
uv run chinese-graphrag api perf
```

### 部署檢查
```bash
# 部署前檢查
uv run chinese-graphrag api deploy-check
```

## 相關檔案

- **主要應用程式**: `src/chinese_graphrag/api/app.py`
- **路由定義**: `src/chinese_graphrag/api/routes/`
- **資料模型**: `src/chinese_graphrag/api/models.py`
- **API 管理**: `src/chinese_graphrag/api/manage.py`
- **文件生成**: `src/chinese_graphrag/api/docs.py`
- **驗證工具**: `src/chinese_graphrag/api/validation.py`

## 更新日誌

- **v1.0.0**: 初始 API 版本
- **v1.1.0**: 新增批次查詢功能
- **v1.2.0**: 新增監控和配置管理
- **v1.3.0**: 新增健康檢查和系統診斷

---

*此文件由 Serena 自動生成，基於當前程式碼結構分析。如有疑問，請參考原始程式碼或聯繫開發團隊。*
## 
CLI 命令支援

Chinese GraphRAG 提供完整的 CLI 命令來管理 API 服務：

### 啟動 API 伺服器

```bash
# 基本啟動
chinese-graphrag api server

# 開發模式（自動重載）
chinese-graphrag api server --reload

# 指定主機和端口
chinese-graphrag api server --host 127.0.0.1 --port 8080

# 多進程模式
chinese-graphrag api server --workers 4

# 設定日誌級別
chinese-graphrag api server --log-level debug
```

**參數說明**:
- `--host, -h`: 綁定主機地址（預設: 0.0.0.0）
- `--port, -p`: 綁定端口（預設: 8000）
- `--reload`: 啟用自動重載（開發模式）
- `--workers, -w`: 工作進程數（預設: 1）
- `--log-level`: 日誌級別（debug/info/warning/error）

### 生成 API 文件

```bash
# 生成完整 API 文件
chinese-graphrag api docs

# 生成特定格式文件
chinese-graphrag api docs --format openapi
chinese-graphrag api docs --format markdown
```

### API 驗證

```bash
# 驗證 API 配置和端點
chinese-graphrag api validate

# 驗證特定端點
chinese-graphrag api validate --endpoint /api/v1/query
```

### 效能測試

```bash
# 執行 API 效能測試
chinese-graphrag api perf

# 指定測試參數
chinese-graphrag api perf --concurrent 10 --requests 1000
```

### 部署檢查

```bash
# 部署前系統檢查
chinese-graphrag api deploy-check

# 檢查特定組件
chinese-graphrag api deploy-check --component database
chinese-graphrag api deploy-check --component vector-store
```

## 服務管理

### 系統服務配置

#### Systemd 服務檔案範例

```ini
[Unit]
Description=Chinese GraphRAG API Server
After=network.target

[Service]
Type=exec
User=graphrag
Group=graphrag
WorkingDirectory=/opt/chinese-graphrag
Environment=PATH=/opt/chinese-graphrag/.venv/bin
ExecStart=/opt/chinese-graphrag/.venv/bin/chinese-graphrag api server --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Docker 部署

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "chinese-graphrag", "api", "server", "--host", "0.0.0.0"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GRAPHRAG_CONFIG_PATH=/app/config
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    depends_on:
      - vector-db
      
  vector-db:
    image: lancedb/lancedb:latest
    volumes:
      - vector_data:/data
      
volumes:
  vector_data:
```

## 監控與日誌

### 日誌配置

API 服務支援結構化日誌輸出，可配置不同的日誌級別和輸出格式：

```python
# 日誌配置範例
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/api.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

### 效能監控

API 內建效能監控功能，可透過 `/api/v1/monitoring/metrics` 端點取得：

- **請求統計**: 總請求數、成功率、錯誤率
- **響應時間**: 平均響應時間、P95、P99
- **資源使用**: CPU、記憶體、磁碟使用率
- **業務指標**: 索引處理速度、查詢準確率

### 警告系統

系統支援多級警告機制：

- **INFO**: 一般資訊性警告
- **WARNING**: 需要注意的警告
- **ERROR**: 錯誤警告，需要立即處理
- **CRITICAL**: 嚴重錯誤，系統可能無法正常運行

## 安全性

### 認證機制

```python
# API Key 認證範例
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/api/v1/query",
    headers=headers,
    json={"query": "你的查詢"}
)
```

### CORS 配置

```python
# CORS 設定範例
CORS_SETTINGS = {
    "allow_origins": ["http://localhost:3000", "https://yourdomain.com"],
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE"],
    "allow_headers": ["*"],
}
```

### 速率限制

```python
# 速率限制配置
RATE_LIMIT_SETTINGS = {
    "default": "100/minute",
    "indexing": "10/minute", 
    "batch_query": "5/minute",
    "monitoring": "200/minute"
}
```

## 故障排除

### 常見問題

#### 1. 服務啟動失敗
```bash
# 檢查配置
chinese-graphrag doctor

# 檢查端口占用
lsof -i :8000

# 檢查日誌
tail -f logs/api.log
```

#### 2. 查詢響應慢
```bash
# 檢查系統資源
chinese-graphrag api perf

# 檢查向量資料庫狀態
chinese-graphrag status --component vector-store
```

#### 3. 索引處理失敗
```bash
# 檢查索引狀態
chinese-graphrag show-index

# 重新處理失敗的文件
chinese-graphrag index --retry-failed
```

### 除錯模式

```bash
# 啟用除錯模式
chinese-graphrag api server --log-level debug --reload

# 檢查 API 健康狀態
curl http://localhost:8000/health/detailed
```

## 版本兼容性

| API 版本 | 支援的功能 | 狀態 |
|---------|-----------|------|
| v1.0 | 基本索引和查詢 | 穩定 |
| v1.1 | 批次處理 | 穩定 |
| v1.2 | 監控和配置 | 穩定 |
| v1.3 | 健康檢查 | 當前 |
| v2.0 | 進階分析功能 | 規劃中 |

## 社群與支援

- **GitHub**: [chinese-graphrag](https://github.com/your-org/chinese-graphrag)
- **文件**: [完整文件](https://docs.chinese-graphrag.com)
- **問題回報**: [GitHub Issues](https://github.com/your-org/chinese-graphrag/issues)
- **討論區**: [GitHub Discussions](https://github.com/your-org/chinese-graphrag/discussions)

---

*最後更新：2025-08-01*
*文件版本：1.3.0*