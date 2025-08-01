# Chinese GraphRAG API 使用指南

> 生成時間：2025-08-01 21:57:27 (Asia/Taipei)
> 適用版本：v1.3.0

## 快速開始

### 1. 啟動 API 服務

```bash
# 基本啟動
chinese-graphrag api server

# 開發模式
chinese-graphrag api server --reload --log-level debug

# 生產模式
chinese-graphrag api server --workers 4 --host 0.0.0.0 --port 8000
```

### 2. 驗證服務狀態

```bash
# 檢查服務健康狀態
curl http://localhost:8000/health

# 詳細健康檢查
curl http://localhost:8000/health/detailed
```

### 3. 查看 API 文件

訪問 `http://localhost:8000/api/v1/docs` 查看互動式 API 文件。

## 完整工作流程範例

### 步驟 1: 建立文件索引

```python
import requests
import json
import time

# API 基礎 URL
BASE_URL = "http://localhost:8000/api/v1"

# 建立索引請求
index_request = {
    "input_path": "./documents",
    "file_types": ["txt", "md", "pdf"],
    "batch_size": 32,
    "force_rebuild": False
}

# 發送索引請求
response = requests.post(
    f"{BASE_URL}/index",
    json=index_request,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    result = response.json()
    task_id = result["task_id"]
    print(f"索引任務已啟動，任務 ID: {task_id}")
else:
    print(f"索引請求失敗: {response.text}")
    exit(1)
```

### 步驟 2: 監控索引進度

```python
# 監控索引進度
def monitor_indexing_progress(task_id):
    while True:
        response = requests.get(f"{BASE_URL}/index/{task_id}")
        
        if response.status_code == 200:
            result = response.json()
            status = result["status"]
            progress = result["progress"]
            
            print(f"索引進度: {progress:.1f}% - 狀態: {status}")
            
            if status == "completed":
                print("索引建立完成！")
                break
            elif status == "failed":
                print("索引建立失敗！")
                break
            
            time.sleep(5)  # 等待 5 秒後再次檢查
        else:
            print(f"查詢索引狀態失敗: {response.text}")
            break

# 執行監控
monitor_indexing_progress(task_id)
```

### 步驟 3: 執行查詢

```python
# 單一查詢
def execute_query(query_text, query_type="auto", max_results=10):
    query_request = {
        "query": query_text,
        "query_type": query_type,
        "max_results": max_results,
        "include_sources": True
    }
    
    response = requests.post(
        f"{BASE_URL}/query",
        json=query_request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"查詢成功，找到 {result['total_results']} 個結果")
        print(f"查詢時間: {result['query_time']:.3f} 秒")
        
        for i, item in enumerate(result['results'], 1):
            print(f"\n結果 {i} (分數: {item['score']:.3f}):")
            print(f"內容: {item['content'][:200]}...")
            if item['source']:
                print(f"來源: {item['source']}")
        
        return result
    else:
        print(f"查詢失敗: {response.text}")
        return None

# 執行查詢範例
queries = [
    "什麼是人工智慧？",
    "機器學習的主要算法有哪些？",
    "深度學習在自然語言處理中的應用"
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"查詢: {query}")
    print('='*50)
    execute_query(query)
```

### 步驟 4: 批次查詢

```python
# 批次查詢
def execute_batch_query(queries, query_type="auto"):
    batch_request = {
        "queries": queries,
        "query_type": query_type,
        "max_results_per_query": 5,
        "parallel_workers": 3
    }
    
    response = requests.post(
        f"{BASE_URL}/query/batch",
        json=batch_request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        batch_task_id = result["task_id"]
        print(f"批次查詢已啟動，任務 ID: {batch_task_id}")
        
        # 監控批次查詢進度
        while True:
            response = requests.get(f"{BASE_URL}/query/batch/{batch_task_id}")
            
            if response.status_code == 200:
                result = response.json()
                completed = result["completed_queries"]
                total = result["total_queries"]
                
                print(f"批次查詢進度: {completed}/{total}")
                
                if result["status"] == "completed":
                    print("批次查詢完成！")
                    # 處理結果
                    if result["results"]:
                        for i, query_result in enumerate(result["results"]):
                            print(f"\n查詢 {i+1} 結果:")
                            print(f"  找到 {query_result['total_results']} 個結果")
                    break
                elif result["status"] == "failed":
                    print("批次查詢失敗！")
                    break
                
                time.sleep(3)
            else:
                print(f"查詢批次狀態失敗: {response.text}")
                break
    else:
        print(f"批次查詢請求失敗: {response.text}")

# 執行批次查詢
batch_queries = [
    "人工智慧的發展歷史",
    "機器學習算法比較",
    "深度學習框架介紹",
    "自然語言處理技術",
    "計算機視覺應用"
]

execute_batch_query(batch_queries)
```

## 進階使用範例

### 1. 使用過濾器查詢

```python
# 帶過濾器的查詢
def query_with_filters(query_text):
    query_request = {
        "query": query_text,
        "query_type": "semantic",
        "max_results": 20,
        "include_sources": True,
        "filters": {
            "document_type": "academic",
            "date_range": {
                "start": "2023-01-01",
                "end": "2024-12-31"
            },
            "language": "zh-TW",
            "min_score": 0.7
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/query",
        json=query_request,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json() if response.status_code == 200 else None

# 使用範例
result = query_with_filters("深度學習在醫療診斷中的應用")
```

### 2. 配置管理

```python
# 取得當前配置
def get_current_config():
    response = requests.get(f"{BASE_URL}/config")
    
    if response.status_code == 200:
        config = response.json()
        print("當前配置:")
        print(json.dumps(config["config"], indent=2, ensure_ascii=False))
        return config
    else:
        print(f"取得配置失敗: {response.text}")
        return None

# 更新配置
def update_config(section, key, value):
    update_request = {
        "section": section,
        "key": key,
        "value": value,
        "validate_only": False
    }
    
    response = requests.put(
        f"{BASE_URL}/config",
        json=update_request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        print(f"配置更新成功: {section}.{key} = {value}")
        return response.json()
    else:
        print(f"配置更新失敗: {response.text}")
        return None

# 使用範例
get_current_config()
update_config("embedding", "batch_size", 64)
```

### 3. 系統監控

```python
# 取得系統指標
def get_system_metrics():
    response = requests.get(f"{BASE_URL}/monitoring/metrics")
    
    if response.status_code == 200:
        metrics = response.json()
        
        print("系統指標:")
        print(f"  CPU 使用率: {metrics['system_metrics']['cpu_usage']:.1f}%")
        print(f"  記憶體使用率: {metrics['system_metrics']['memory_usage']:.1f}%")
        print(f"  磁碟使用率: {metrics['system_metrics']['disk_usage']:.1f}%")
        
        print("\n應用程式指標:")
        app_metrics = metrics['application_metrics']
        print(f"  總請求數: {app_metrics['total_requests']}")
        print(f"  成功率: {app_metrics['successful_requests']/app_metrics['total_requests']*100:.1f}%")
        print(f"  平均響應時間: {app_metrics['average_response_time']:.1f}ms")
        print(f"  快取命中率: {app_metrics['cache_hit_rate']:.1f}%")
        
        return metrics
    else:
        print(f"取得系統指標失敗: {response.text}")
        return None

# 取得警告資訊
def get_system_alerts():
    response = requests.get(f"{BASE_URL}/monitoring/alerts")
    
    if response.status_code == 200:
        alerts = response.json()
        
        if alerts["data"]:
            print("系統警告:")
            for alert in alerts["data"]:
                print(f"  [{alert['level']}] {alert['message']}")
        else:
            print("目前沒有系統警告")
        
        return alerts
    else:
        print(f"取得警告資訊失敗: {response.text}")
        return None

# 使用範例
get_system_metrics()
get_system_alerts()
```

### 4. 錯誤處理

```python
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

class GraphRAGAPIClient:
    def __init__(self, base_url="http://localhost:8000/api/v1", timeout=30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def _make_request(self, method, endpoint, **kwargs):
        """統一的請求處理方法"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method, url, timeout=self.timeout, **kwargs
            )
            
            # 檢查 HTTP 狀態碼
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            elif response.status_code == 400:
                return {"success": False, "error": "請求參數錯誤", "details": response.text}
            elif response.status_code == 404:
                return {"success": False, "error": "資源不存在", "details": response.text}
            elif response.status_code == 422:
                return {"success": False, "error": "資料驗證失敗", "details": response.json()}
            elif response.status_code == 429:
                return {"success": False, "error": "請求頻率過高，請稍後再試"}
            elif response.status_code >= 500:
                return {"success": False, "error": "伺服器內部錯誤", "details": response.text}
            else:
                return {"success": False, "error": f"未知錯誤 (HTTP {response.status_code})"}
                
        except Timeout:
            return {"success": False, "error": "請求超時"}
        except ConnectionError:
            return {"success": False, "error": "連接失敗，請檢查服務是否正常運行"}
        except RequestException as e:
            return {"success": False, "error": f"請求異常: {str(e)}"}
    
    def query(self, query_text, **kwargs):
        """執行查詢"""
        data = {"query": query_text, **kwargs}
        return self._make_request("POST", "/query", json=data)
    
    def create_index(self, input_path, **kwargs):
        """建立索引"""
        data = {"input_path": input_path, **kwargs}
        return self._make_request("POST", "/index", json=data)
    
    def get_index_status(self, task_id):
        """取得索引狀態"""
        return self._make_request("GET", f"/index/{task_id}")

# 使用範例
client = GraphRAGAPIClient()

# 執行查詢
result = client.query("什麼是機器學習？")
if result["success"]:
    print("查詢成功:")
    print(f"找到 {result['data']['total_results']} 個結果")
else:
    print(f"查詢失敗: {result['error']}")
```

## 效能優化建議

### 1. 批次處理

```python
# 優化：使用批次查詢而非多個單獨查詢
queries = ["查詢1", "查詢2", "查詢3", "查詢4", "查詢5"]

# 不推薦：多個單獨請求
# for query in queries:
#     execute_query(query)

# 推薦：批次處理
execute_batch_query(queries)
```

### 2. 連接池

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置連接池和重試策略
session = requests.Session()

# 重試策略
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

# HTTP 適配器
adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=retry_strategy
)

session.mount("http://", adapter)
session.mount("https://", adapter)

# 使用 session 發送請求
response = session.post(f"{BASE_URL}/query", json=query_request)
```

### 3. 異步處理

```python
import asyncio
import aiohttp

async def async_query(session, query_text):
    """異步查詢"""
    query_request = {
        "query": query_text,
        "query_type": "auto",
        "max_results": 10
    }
    
    async with session.post(f"{BASE_URL}/query", json=query_request) as response:
        if response.status == 200:
            return await response.json()
        else:
            return {"error": f"HTTP {response.status}"}

async def batch_async_queries(queries):
    """批次異步查詢"""
    async with aiohttp.ClientSession() as session:
        tasks = [async_query(session, query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results

# 使用範例
queries = ["查詢1", "查詢2", "查詢3"]
results = asyncio.run(batch_async_queries(queries))
```

## 測試與除錯

### 1. API 測試腳本

```python
import unittest
import requests

class TestGraphRAGAPI(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:8000/api/v1"
        self.headers = {"Content-Type": "application/json"}
    
    def test_health_check(self):
        """測試健康檢查"""
        response = requests.get("http://localhost:8000/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
    
    def test_query_api(self):
        """測試查詢 API"""
        query_request = {
            "query": "測試查詢",
            "max_results": 5
        }
        
        response = requests.post(
            f"{self.base_url}/query",
            json=query_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("results", data)
        self.assertIn("total_results", data)
    
    def test_invalid_query(self):
        """測試無效查詢"""
        query_request = {
            "query": "",  # 空查詢
            "max_results": 5
        }
        
        response = requests.post(
            f"{self.base_url}/query",
            json=query_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()
```

### 2. 除錯工具

```python
import logging
import time
from functools import wraps

# 設定日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_api_call(func):
    """API 呼叫除錯裝飾器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"呼叫 {func.__name__} - 參數: {args}, {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"{func.__name__} 成功 - 耗時: {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} 失敗 - 耗時: {end_time - start_time:.3f}s - 錯誤: {e}")
            raise
    
    return wrapper

@debug_api_call
def debug_query(query_text):
    """除錯版本的查詢函數"""
    return execute_query(query_text)

# 使用範例
debug_query("測試查詢")
```

## 最佳實踐

### 1. 錯誤處理
- 總是檢查 HTTP 狀態碼
- 實作重試機制
- 記錄錯誤詳情

### 2. 效能優化
- 使用連接池
- 實作快取機制
- 批次處理大量請求

### 3. 安全性
- 使用 HTTPS
- 實作 API 金鑰認證
- 驗證輸入資料

### 4. 監控
- 監控 API 響應時間
- 追蹤錯誤率
- 設定警告閾值

---

*此指南提供完整的 API 使用範例和最佳實踐，幫助開發者快速上手 Chinese GraphRAG API。*