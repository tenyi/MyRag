# API 使用文件

本文件詳細介紹 Chinese GraphRAG 系統的 REST API 使用方法，包含所有端點的說明、參數和範例。

## 目錄

- [API 概述](#api-概述)
- [認證和授權](#認證和授權)
- [基本使用](#基本使用)
- [索引 API](#索引-api)
- [查詢 API](#查詢-api)
- [配置管理 API](#配置管理-api)
- [監控 API](#監控-api)
- [錯誤處理](#錯誤處理)
- [SDK 和客戶端](#sdk-和客戶端)

## API 概述

### 基本資訊

- **基礎 URL**: `http://localhost:8000/api/v1`
- **內容類型**: `application/json`
- **字符編碼**: UTF-8
- **API 版本**: v1

### 啟動 API 服務

```bash
# 使用預設配置啟動
uv run chinese-graphrag api start

# 指定埠號和主機
uv run chinese-graphrag api start --host 0.0.0.0 --port 8080

# 使用自訂配置檔案
uv run chinese-graphrag api start --config config/prod.yaml
```

### API 文件

啟動服務後，可以透過以下 URL 查看互動式 API 文件：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## 認證和授權

### API 金鑰認證

```bash
# 在請求標頭中包含 API 金鑰
curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     http://localhost:8000/api/v1/health
```

### Bearer Token 認證

```bash
# 使用 Bearer Token
curl -H "Authorization: Bearer your-token" \
     -H "Content-Type: application/json" \
     http://localhost:8000/api/v1/health
```

## 基本使用

### 健康檢查

```bash
# 檢查 API 服務狀態
curl http://localhost:8000/api/v1/health
```

**回應範例**：

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "database": "connected",
    "embedding_service": "ready",
    "llm_service": "ready"
  }
}
```

### 系統資訊

```bash
# 獲取系統資訊
curl http://localhost:8000/api/v1/info
```

**回應範例**：

```json
{
  "name": "Chinese GraphRAG",
  "version": "0.1.0",
  "description": "中文知識圖譜檢索增強生成系統",
  "models": {
    "chat_model": "gpt-4o-mini",
    "embedding_model": "BAAI/bge-m3"
  },
  "capabilities": [
    "document_indexing",
    "global_search",
    "local_search",
    "chinese_optimization"
  ]
}
```

## 索引 API

### 建立索引任務

```bash
# 建立新的索引任務
curl -X POST http://localhost:8000/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/path/to/documents",
    "output_path": "/path/to/output",
    "config": {
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "embedding_model": "chinese_embedding_model"
    }
  }'
```

**請求參數**：

- `input_path` (string, 必填): 輸入文件路徑
- `output_path` (string, 必填): 輸出資料路徑
- `config` (object, 可選): 索引配置選項

**回應範例**：

```json
{
  "task_id": "idx_20240101_120000_abc123",
  "status": "started",
  "message": "索引任務已開始",
  "estimated_time": "10-15 分鐘",
  "created_at": "2024-01-01T12:00:00Z"
}
```

### 查詢索引狀態

```bash
# 查詢特定索引任務狀態
curl http://localhost:8000/api/v1/index/idx_20240101_120000_abc123/status
```

**回應範例**：

```json
{
  "task_id": "idx_20240101_120000_abc123",
  "status": "running",
  "progress": {
    "current_step": "entity_extraction",
    "completed_steps": 3,
    "total_steps": 8,
    "percentage": 37.5
  },
  "stats": {
    "documents_processed": 45,
    "total_documents": 120,
    "entities_extracted": 1250,
    "relationships_found": 890
  },
  "estimated_remaining": "8 分鐘"
}
```

### 取消索引任務

```bash
# 取消執行中的索引任務
curl -X DELETE http://localhost:8000/api/v1/index/idx_20240101_120000_abc123
```

### 列出所有索引任務

```bash
# 列出所有索引任務
curl http://localhost:8000/api/v1/index/tasks
```

## 查詢 API

### 執行查詢

```bash
# 執行全域搜尋查詢
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什麼是人工智慧？",
    "search_type": "global",
    "config": {
      "max_tokens": 1000,
      "temperature": 0.1
    }
  }'
```

**請求參數**：

- `question` (string, 必填): 查詢問題
- `search_type` (string, 可選): 搜尋類型 (`global`, `local`, `auto`)
- `config` (object, 可選): 查詢配置選項

**回應範例**：

```json
{
  "query_id": "qry_20240101_120000_def456",
  "question": "什麼是人工智慧？",
  "answer": "人工智慧（Artificial Intelligence, AI）是指讓機器模擬人類智慧的技術...",
  "search_type": "global",
  "sources": [
    {
      "document": "ai_introduction.pdf",
      "chunk": "第一章：人工智慧概述",
      "relevance_score": 0.95
    }
  ],
  "metadata": {
    "response_time": "2.3s",
    "tokens_used": 450,
    "model_used": "gpt-4o-mini"
  }
}
```

### 批次查詢

```bash
# 執行批次查詢
curl -X POST http://localhost:8000/api/v1/query/batch \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "什麼是機器學習？",
      "深度學習的應用有哪些？",
      "如何評估 AI 模型的效能？"
    ],
    "search_type": "auto",
    "config": {
      "max_tokens": 500,
      "parallel": true
    }
  }'
```

### 查詢歷史

```bash
# 獲取查詢歷史
curl http://localhost:8000/api/v1/query/history?limit=10&offset=0
```

## 配置管理 API

### 獲取當前配置

```bash
# 獲取系統配置
curl http://localhost:8000/api/v1/config
```

### 更新配置

```bash
# 更新模型配置
curl -X PUT http://localhost:8000/api/v1/config/models \
  -H "Content-Type: application/json" \
  -d '{
    "default_chat_model": {
      "model": "gpt-4o",
      "temperature": 0.2
    }
  }'
```

### 重載配置

```bash
# 重新載入配置檔案
curl -X POST http://localhost:8000/api/v1/config/reload
```

## 監控 API

### 系統指標

```bash
# 獲取系統效能指標
curl http://localhost:8000/api/v1/monitoring/metrics
```

**回應範例**：

```json
{
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 68.5,
    "disk_usage": 23.1
  },
  "api": {
    "requests_per_minute": 12,
    "average_response_time": "1.8s",
    "error_rate": 0.02
  },
  "models": {
    "embedding_requests": 156,
    "llm_requests": 89,
    "cache_hit_rate": 0.73
  }
}
```

### 使用量統計

```bash
# 獲取使用量統計
curl http://localhost:8000/api/v1/monitoring/usage
```

### 錯誤日誌

```bash
# 獲取最近的錯誤日誌
curl http://localhost:8000/api/v1/monitoring/errors?limit=50
```

## 錯誤處理

### 標準錯誤格式

所有 API 錯誤都遵循統一的格式：

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "請求參數無效",
    "details": {
      "field": "question",
      "reason": "不能為空"
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### 常見錯誤碼

| 錯誤碼 | HTTP 狀態 | 說明 |
|--------|-----------|------|
| `INVALID_REQUEST` | 400 | 請求參數無效 |
| `UNAUTHORIZED` | 401 | 未授權訪問 |
| `FORBIDDEN` | 403 | 權限不足 |
| `NOT_FOUND` | 404 | 資源不存在 |
| `RATE_LIMIT_EXCEEDED` | 429 | 請求頻率超限 |
| `INTERNAL_ERROR` | 500 | 內部伺服器錯誤 |
| `SERVICE_UNAVAILABLE` | 503 | 服務暫時不可用 |

### 重試策略

建議使用指數退避重試策略：

```python
import time
import requests
from typing import Optional

def api_request_with_retry(
    url: str,
    method: str = "GET",
    data: Optional[dict] = None,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> requests.Response:
    """帶重試機制的 API 請求"""
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.request(method, url, json=data)
            
            # 成功或客戶端錯誤不重試
            if response.status_code < 500:
                return response
                
        except requests.RequestException:
            pass
        
        # 最後一次嘗試失敗
        if attempt == max_retries:
            raise
        
        # 指數退避
        delay = base_delay * (2 ** attempt)
        time.sleep(delay)
```

## SDK 和客戶端

### Python SDK

```python
from chinese_graphrag.client import GraphRAGClient

# 建立客戶端
client = GraphRAGClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# 執行索引
task = client.index(
    input_path="./documents",
    output_path="./data"
)

# 等待索引完成
task.wait_for_completion()

# 執行查詢
result = client.query("什麼是人工智慧？")
print(result.answer)
```

### JavaScript SDK

```javascript
import { GraphRAGClient } from 'chinese-graphrag-js';

// 建立客戶端
const client = new GraphRAGClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// 執行查詢
const result = await client.query({
  question: '什麼是人工智慧？',
  searchType: 'global'
});

console.log(result.answer);
```

### cURL 腳本範例

```bash
#!/bin/bash

# 設定基本變數
BASE_URL="http://localhost:8000/api/v1"
API_KEY="your-api-key"

# 執行查詢的函數
query() {
  local question="$1"
  curl -X POST "$BASE_URL/query" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$question\"}" \
    | jq -r '.answer'
}

# 使用範例
query "什麼是機器學習？"
```

## 最佳實踐

### 1. 效能優化

- 使用批次查詢處理多個問題
- 啟用快取以提高回應速度
- 合理設定超時時間

### 2. 錯誤處理

- 實作重試機制
- 記錄錯誤詳情以便除錯
- 提供友好的錯誤訊息

### 3. 安全性

- 使用 HTTPS 傳輸敏感資料
- 定期輪換 API 金鑰
- 限制 API 訪問頻率

### 4. 監控

- 監控 API 回應時間
- 追蹤錯誤率和使用量
- 設定告警機制

## 相關文件

- [安裝和配置指南](installation_guide.md)
- [故障排除指南](troubleshooting_guide.md)
- [程式碼範例和教學](examples_and_tutorials.md)
- [架構和設計文件](architecture_design.md)
