# 精簡查詢 API 使用指南

本文檔介紹新增的精簡查詢 API 端點，專為需要簡潔回答的應用場景設計。

## API 端點概覽

### 1. `/api/query/simple` - 精簡查詢
**HTTP 方法**: POST  
**用途**: 執行精簡查詢，僅返回核心答案

#### 請求格式
```json
{
  "query": "小明姓什麼",
  "search_type": "auto",
  "use_llm_segmentation": true
}
```

#### 請求參數
- `query` (string, 必填): 查詢內容
- `search_type` (string, 可選): 搜尋類型，預設 "auto"
  - `"auto"`: 自動選擇最佳搜尋方式
  - `"global"`: 全域搜尋
  - `"local"`: 本地搜尋
- `use_llm_segmentation` (boolean, 可選): 是否使用 LLM 分詞，預設 `true`

#### 回應格式
```json
{
  "success": true,
  "message": "查詢完成",
  "timestamp": "2025-08-04T10:30:00.000Z",
  "answer": "小明的姓是「張」，全名為張小明。",
  "confidence": 0.95,
  "search_type": "local",
  "response_time": 2.156
}
```

#### 回應欄位說明
- `success` (boolean): 查詢是否成功
- `message` (string): 回應訊息
- `timestamp` (string): 回應時間戳
- `answer` (string): 查詢答案
- `confidence` (float): 信心度 (0-1)
- `search_type` (string): 實際使用的搜尋類型
- `response_time` (float): 系統回應時間（秒）

### 2. `/api/query/simple/with-reasoning` - 精簡查詢（含推理）
**HTTP 方法**: POST  
**用途**: 執行精簡查詢，包含推理路徑

#### 請求格式
與 `/api/query/simple` 相同

#### 回應格式
```json
{
  "success": true,
  "message": "查詢完成",
  "timestamp": "2025-08-04T10:30:00.000Z",
  "answer": "小明的姓是「張」，全名為張小明。",
  "confidence": 0.95,
  "search_type": "local",
  "response_time": 2.156,
  "reasoning_path": [
    "識別 10 個目標實體",
    "分析 12 個相關關係",
    "收集 5 個文本證據",
    "基於實體知識圖生成詳細回答"
  ]
}
```

#### 額外回應欄位
- `reasoning_path` (array): 推理路徑步驟列表

## 使用範例

### Python 範例
```python
import requests

# 基本精簡查詢
response = requests.post("http://localhost:8000/api/query/simple", json={
    "query": "小明姓什麼",
    "search_type": "auto",
    "use_llm_segmentation": True
})

result = response.json()
print(f"答案: {result['answer']}")
print(f"信心度: {result['confidence']}")

# 含推理的精簡查詢
response = requests.post("http://localhost:8000/api/query/simple/with-reasoning", json={
    "query": "GraphRAG 是什麼",
    "search_type": "global"
})

result = response.json()
print(f"答案: {result['answer']}")
print("推理過程:")
for i, step in enumerate(result['reasoning_path'], 1):
    print(f"{i}. {step}")
```

### cURL 範例
```bash
# 基本精簡查詢
curl -X POST "http://localhost:8000/api/query/simple" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "小明姓什麼",
    "search_type": "auto",
    "use_llm_segmentation": true
  }'

# 含推理的精簡查詢
curl -X POST "http://localhost:8000/api/query/simple/with-reasoning" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什麼是人工智慧",
    "search_type": "global"
  }'
```

### JavaScript 範例
```javascript
// 使用 fetch API
async function simpleQuery(query) {
  const response = await fetch('http://localhost:8000/api/query/simple', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      search_type: 'auto',
      use_llm_segmentation: true
    })
  });

  const result = await response.json();
  return result;
}

// 使用範例
simpleQuery("小明姓什麼").then(result => {
  console.log('答案:', result.answer);
  console.log('信心度:', result.confidence);
});
```

## 錯誤處理

### 錯誤回應格式
```json
{
  "success": false,
  "message": "精簡查詢執行失敗: 查詢引擎初始化失敗",
  "timestamp": "2025-08-04T10:30:00.000Z"
}
```

### 常見錯誤碼
- `400 Bad Request`: 查詢內容為空或格式錯誤
- `500 Internal Server Error`: 服務器內部錯誤（如查詢引擎初始化失敗）
- `503 Service Unavailable`: 服務暫時不可用

## 效能考量

### 分詞方法選擇
- **LLM 分詞** (`use_llm_segmentation: true`): 
  - 優點: 更準確的中文分詞，特別適合人名處理
  - 缺點: 回應時間較長（通常 +1-2秒）
  - 適用場景: 需要高準確度的查詢

- **jieba 分詞** (`use_llm_segmentation: false`):
  - 優點: 回應速度快
  - 缺點: 中文人名處理可能不如 LLM 精確
  - 適用場景: 需要快速回應的查詢

### 搜尋類型選擇
- **auto**: 系統自動選擇，適合大部分情況
- **local**: 針對特定實體的詳細查詢，適合問答場景
- **global**: 基於整體知識的綜合查詢，適合概念解釋

## 與原有 API 的差異

| 特性 | 精簡查詢 API | 原有查詢 API |
|------|-------------|-------------|
| 回應內容 | 僅核心答案 | 包含詳細結果列表 |
| 回應大小 | 小（~1KB） | 大（~10KB+） |
| 處理速度 | 快 | 較慢 |
| 推理資訊 | 可選 | 包含在 metadata 中 |
| 來源資訊 | 不包含 | 詳細來源資訊 |
| 適用場景 | 聊天機器人、快速問答 | 詳細分析、研究 |

## 測試腳本

項目根目錄包含測試腳本 `test_simple_query_api.py`，可用於測試 API 功能：

```bash
# 安裝依賴
pip install aiohttp

# 執行測試
python test_simple_query_api.py
```

測試腳本會自動測試不同的查詢方法並比較結果。