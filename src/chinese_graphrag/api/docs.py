#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API 文件生成和管理模組

提供自動生成 API 文件、範例程式碼和測試案例的功能。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


class APIDocumentationGenerator:
    """API 文件生成器。

    負責生成完整的 API 文件，包括 OpenAPI 規格、使用範例和測試案例。
    """

    def __init__(self, app: FastAPI, output_dir: str = "docs/api"):
        """初始化文件生成器。

        Args:
            app: FastAPI 應用程式實例
            output_dir: 文件輸出目錄
        """
        self.app = app
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """生成 OpenAPI 規格。

        Returns:
            OpenAPI 規格字典
        """
        openapi_spec = get_openapi(
            title=self.app.title,
            version=self.app.version,
            description=self.app.description,
            routes=self.app.routes,
        )

        # 添加額外的中文描述和範例
        self._enhance_openapi_spec(openapi_spec)

        return openapi_spec

    def _enhance_openapi_spec(self, spec: Dict[str, Any]):
        """增強 OpenAPI 規格，添加中文描述和範例。

        Args:
            spec: OpenAPI 規格字典
        """
        # 添加伺服器資訊
        spec["servers"] = [
            {"url": "http://localhost:8000", "description": "開發環境"},
            {
                "url": "https://api.chinese-graphrag.example.com",
                "description": "生產環境",
            },
        ]

        # 添加標籤描述
        spec["tags"] = [
            {"name": "health", "description": "系統健康檢查相關的端點"},
            {"name": "indexing", "description": "文件索引和處理相關的端點"},
            {"name": "query", "description": "查詢和檢索相關的端點"},
            {"name": "configuration", "description": "系統配置管理相關的端點"},
            {"name": "monitoring", "description": "系統監控和指標相關的端點"},
        ]

        # 添加全域安全定義（雖然目前不需要認證）
        spec["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API 金鑰認證（未來功能）",
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT 權杖認證（未來功能）",
            },
        }

        # 添加範例到每個端點
        self._add_examples_to_paths(spec)

    def _add_examples_to_paths(self, spec: Dict[str, Any]):
        """為每個 API 端點添加範例。

        Args:
            spec: OpenAPI 規格字典
        """
        examples = {
            "/health": {
                "get": {
                    "examples": {
                        "successful_response": {
                            "summary": "成功回應範例",
                            "value": {
                                "success": True,
                                "status": "healthy",
                                "timestamp": 1703123456.789,
                                "data": {
                                    "system": {
                                        "status": "running",
                                        "uptime": 3600.5,
                                        "version": "v1",
                                    }
                                },
                            },
                        }
                    }
                }
            },
            "/api/v1/index": {
                "post": {
                    "examples": {
                        "basic_indexing": {
                            "summary": "基本索引請求",
                            "value": {
                                "input_path": "./documents",
                                "output_path": "./data/output",
                                "file_types": ["txt", "pdf", "docx"],
                                "batch_size": 32,
                                "force_rebuild": False,
                            },
                        }
                    }
                }
            },
            "/api/v1/query": {
                "post": {
                    "examples": {
                        "simple_query": {
                            "summary": "簡單查詢範例",
                            "value": {
                                "query": "什麼是人工智慧？",
                                "query_type": "global_search",
                                "max_tokens": 2000,
                                "temperature": 0.7,
                            },
                        }
                    }
                }
            },
        }

        # 將範例添加到對應的路徑
        if "paths" in spec:
            for path, methods in spec["paths"].items():
                if path in examples:
                    for method, method_spec in methods.items():
                        if method in examples[path]:
                            if "requestBody" in method_spec:
                                method_spec["requestBody"]["content"][
                                    "application/json"
                                ]["examples"] = examples[path][method]["examples"]

    def save_openapi_spec(self, format: str = "json") -> Path:
        """儲存 OpenAPI 規格。

        Args:
            format: 檔案格式 ('json' 或 'yaml')

        Returns:
            儲存的檔案路徑
        """
        spec = self.generate_openapi_spec()

        if format.lower() == "yaml":
            file_path = self.output_dir / "openapi.yaml"
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)
        else:
            file_path = self.output_dir / "openapi.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)

        return file_path

    def generate_client_examples(self) -> Dict[str, str]:
        """生成客戶端程式碼範例。

        Returns:
            不同語言的客戶端範例程式碼
        """
        examples = {
            "python": self._generate_python_examples(),
            "javascript": self._generate_javascript_examples(),
            "curl": self._generate_curl_examples(),
        }

        # 儲存範例到檔案
        for language, code in examples.items():
            file_path = self.output_dir / f"examples.{language}"
            if language == "curl":
                file_path = self.output_dir / "examples.sh"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

        return examples

    def _generate_python_examples(self) -> str:
        """生成 Python 客戶端範例。"""
        return '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chinese GraphRAG API Python 客戶端範例

本範例展示如何使用 Python 呼叫 Chinese GraphRAG API。
"""

import requests
import json
from typing import Dict, Any, Optional


class ChineseGraphRAGClient:
    """Chinese GraphRAG API 客戶端。"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """初始化客戶端。
        
        Args:
            base_url: API 基礎 URL
        """
        self.base_url = base_url.rstrip('/')
        self.api_prefix = "/api/v1"
    
    def check_health(self) -> Dict[str, Any]:
        """檢查系統健康狀態。
        
        Returns:
            健康檢查回應
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def create_index(self, 
                    input_path: str,
                    output_path: Optional[str] = None,
                    file_types: Optional[list] = None,
                    batch_size: int = 32,
                    force_rebuild: bool = False) -> Dict[str, Any]:
        """建立文件索引。
        
        Args:
            input_path: 輸入檔案或目錄路徑
            output_path: 輸出目錄路徑
            file_types: 檔案類型過濾
            batch_size: 批次大小
            force_rebuild: 是否強制重建
            
        Returns:
            索引建立回應
        """
        payload = {
            "input_path": input_path,
            "batch_size": batch_size,
            "force_rebuild": force_rebuild
        }
        
        if output_path:
            payload["output_path"] = output_path
        if file_types:
            payload["file_types"] = file_types
        
        response = requests.post(
            f"{self.base_url}{self.api_prefix}/index",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def query(self, 
              query: str,
              query_type: str = "global_search",
              max_tokens: int = 2000,
              temperature: float = 0.7) -> Dict[str, Any]:
        """執行查詢。
        
        Args:
            query: 查詢文字
            query_type: 查詢類型
            max_tokens: 最大權杖數
            temperature: 溫度參數
            
        Returns:
            查詢結果
        """
        payload = {
            "query": query,
            "query_type": query_type,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{self.base_url}{self.api_prefix}/query",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_config(self) -> Dict[str, Any]:
        """取得系統配置。
        
        Returns:
            系統配置
        """
        response = requests.get(f"{self.base_url}{self.api_prefix}/config")
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """取得系統監控指標。
        
        Returns:
            監控指標
        """
        response = requests.get(f"{self.base_url}{self.api_prefix}/monitoring/metrics")
        response.raise_for_status()
        return response.json()


# 使用範例
def main():
    """主函數範例。"""
    client = ChineseGraphRAGClient()
    
    try:
        # 檢查系統健康狀態
        print("檢查系統健康狀態...")
        health = client.check_health()
        print(f"系統狀態: {health['data']['system']['status']}")
        
        # 建立索引
        print("\\n建立文件索引...")
        index_result = client.create_index(
            input_path="./documents",
            output_path="./data/output",
            file_types=["txt", "pdf", "docx"]
        )
        print(f"索引任務 ID: {index_result.get('data', {}).get('task_id')}")
        
        # 執行查詢
        print("\\n執行查詢...")
        query_result = client.query(
            query="什麼是人工智慧？",
            query_type="global_search"
        )
        print(f"查詢結果: {query_result.get('data', {}).get('answer', '')[:100]}...")
        
        # 取得系統配置
        print("\\n取得系統配置...")
        config = client.get_config()
        print(f"模型配置: {config.get('data', {}).get('llm', {}).get('model')}")
        
        # 取得監控指標
        print("\\n取得監控指標...")
        metrics = client.get_metrics()
        system_metrics = metrics.get('data', {}).get('system', {})
        print(f"CPU 使用率: {system_metrics.get('cpu_percent', 0):.1f}%")
        print(f"記憶體使用率: {system_metrics.get('memory_percent', 0):.1f}%")
        
    except requests.exceptions.RequestException as e:
        print(f"API 請求錯誤: {e}")
    except Exception as e:
        print(f"執行錯誤: {e}")


if __name__ == "__main__":
    main()
'''

    def _generate_javascript_examples(self) -> str:
        """生成 JavaScript 客戶端範例。"""
        return """/**
 * Chinese GraphRAG API JavaScript 客戶端範例
 * 
 * 本範例展示如何使用 JavaScript 呼叫 Chinese GraphRAG API。
 */

class ChineseGraphRAGClient {
    /**
     * 初始化客戶端
     * @param {string} baseUrl - API 基礎 URL
     */
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl.replace(/\\/$/, '');
        this.apiPrefix = '/api/v1';
    }

    /**
     * 發送 HTTP 請求
     * @param {string} endpoint - 端點路徑
     * @param {object} options - 請求選項
     * @returns {Promise<object>} 回應資料
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API 請求錯誤:', error);
            throw error;
        }
    }

    /**
     * 檢查系統健康狀態
     * @returns {Promise<object>} 健康檢查回應
     */
    async checkHealth() {
        return await this.request('/health');
    }

    /**
     * 建立文件索引
     * @param {object} params - 索引參數
     * @returns {Promise<object>} 索引建立回應
     */
    async createIndex({
        inputPath,
        outputPath = null,
        fileTypes = null,
        batchSize = 32,
        forceRebuild = false
    }) {
        const payload = {
            input_path: inputPath,
            batch_size: batchSize,
            force_rebuild: forceRebuild
        };

        if (outputPath) payload.output_path = outputPath;
        if (fileTypes) payload.file_types = fileTypes;

        return await this.request(`${this.apiPrefix}/index`, {
            method: 'POST',
            body: payload
        });
    }

    /**
     * 執行查詢
     * @param {object} params - 查詢參數
     * @returns {Promise<object>} 查詢結果
     */
    async query({
        query,
        queryType = 'global_search',
        maxTokens = 2000,
        temperature = 0.7
    }) {
        const payload = {
            query,
            query_type: queryType,
            max_tokens: maxTokens,
            temperature
        };

        return await this.request(`${this.apiPrefix}/query`, {
            method: 'POST',
            body: payload
        });
    }

    /**
     * 取得系統配置
     * @returns {Promise<object>} 系統配置
     */
    async getConfig() {
        return await this.request(`${this.apiPrefix}/config`);
    }

    /**
     * 取得系統監控指標
     * @returns {Promise<object>} 監控指標
     */
    async getMetrics() {
        return await this.request(`${this.apiPrefix}/monitoring/metrics`);
    }
}

// 使用範例
async function main() {
    const client = new ChineseGraphRAGClient();

    try {
        // 檢查系統健康狀態
        console.log('檢查系統健康狀態...');
        const health = await client.checkHealth();
        console.log(`系統狀態: ${health.data.system.status}`);

        // 建立索引
        console.log('\\n建立文件索引...');
        const indexResult = await client.createIndex({
            inputPath: './documents',
            outputPath: './data/output',
            fileTypes: ['txt', 'pdf', 'docx']
        });
        console.log(`索引任務 ID: ${indexResult.data?.task_id}`);

        // 執行查詢
        console.log('\\n執行查詢...');
        const queryResult = await client.query({
            query: '什麼是人工智慧？',
            queryType: 'global_search'
        });
        console.log(`查詢結果: ${queryResult.data?.answer?.substring(0, 100)}...`);

        // 取得系統配置
        console.log('\\n取得系統配置...');
        const config = await client.getConfig();
        console.log(`模型配置: ${config.data?.llm?.model}`);

        // 取得監控指標
        console.log('\\n取得監控指標...');
        const metrics = await client.getMetrics();
        const systemMetrics = metrics.data?.system || {};
        console.log(`CPU 使用率: ${systemMetrics.cpu_percent?.toFixed(1)}%`);
        console.log(`記憶體使用率: ${systemMetrics.memory_percent?.toFixed(1)}%`);

    } catch (error) {
        console.error('執行錯誤:', error);
    }
}

// 如果在 Node.js 環境中執行
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChineseGraphRAGClient;
    
    // 執行範例
    if (require.main === module) {
        main();
    }
}
"""

    def _generate_curl_examples(self) -> str:
        """生成 cURL 命令範例。"""
        return """#!/bin/bash

# Chinese GraphRAG API cURL 範例
# 
# 本腳本展示如何使用 cURL 呼叫 Chinese GraphRAG API。

BASE_URL="http://localhost:8000"
API_PREFIX="/api/v1"

echo "=== Chinese GraphRAG API 測試 ==="
echo

# 檢查系統健康狀態
echo "1. 檢查系統健康狀態"
curl -X GET \\
  "${BASE_URL}/health" \\
  -H "Accept: application/json" \\
  | jq '.'
echo

# 建立文件索引
echo "2. 建立文件索引"
curl -X POST \\
  "${BASE_URL}${API_PREFIX}/index" \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json" \\
  -d '{
    "input_path": "./documents",
    "output_path": "./data/output",
    "file_types": ["txt", "pdf", "docx"],
    "batch_size": 32,
    "force_rebuild": false
  }' \\
  | jq '.'
echo

# 執行查詢
echo "3. 執行查詢"
curl -X POST \\
  "${BASE_URL}${API_PREFIX}/query" \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json" \\
  -d '{
    "query": "什麼是人工智慧？",
    "query_type": "global_search",
    "max_tokens": 2000,
    "temperature": 0.7
  }' \\
  | jq '.'
echo

# 批次查詢
echo "4. 批次查詢"
curl -X POST \\
  "${BASE_URL}${API_PREFIX}/query/batch" \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json" \\
  -d '{
    "queries": [
      {
        "id": "query1",
        "query": "什麼是機器學習？",
        "query_type": "global_search"
      },
      {
        "id": "query2", 
        "query": "深度學習的應用有哪些？",
        "query_type": "local_search"
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }' \\
  | jq '.'
echo

# 取得系統配置
echo "5. 取得系統配置"
curl -X GET \\
  "${BASE_URL}${API_PREFIX}/config" \\
  -H "Accept: application/json" \\
  | jq '.'
echo

# 更新配置
echo "6. 更新配置"
curl -X PUT \\
  "${BASE_URL}${API_PREFIX}/config" \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json" \\
  -d '{
    "llm": {
      "temperature": 0.8,
      "max_tokens": 3000
    },
    "embedding": {
      "batch_size": 64
    }
  }' \\
  | jq '.'
echo

# 驗證配置
echo "7. 驗證配置"
curl -X POST \\
  "${BASE_URL}${API_PREFIX}/config/validate" \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json" \\
  -d '{
    "llm": {
      "model": "gpt-4",
      "temperature": 0.7,
      "max_tokens": 2000
    },
    "embedding": {
      "model": "BAAI/bge-m3",
      "batch_size": 32
    }
  }' \\
  | jq '.'
echo

# 取得監控指標
echo "8. 取得監控指標"
curl -X GET \\
  "${BASE_URL}${API_PREFIX}/monitoring/metrics" \\
  -H "Accept: application/json" \\
  | jq '.'
echo

# 取得警報
echo "9. 取得警報"
curl -X GET \\
  "${BASE_URL}${API_PREFIX}/monitoring/alerts" \\
  -H "Accept: application/json" \\
  | jq '.'
echo

# 效能測試
echo "10. 效能測試"
curl -X POST \\
  "${BASE_URL}${API_PREFIX}/monitoring/performance-test" \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json" \\
  -d '{
    "test_type": "query",
    "duration": 30,
    "concurrent_requests": 5
  }' \\
  | jq '.'
echo

# 取得詳細健康檢查
echo "11. 取得詳細健康檢查"
curl -X GET \\
  "${BASE_URL}/health/detailed" \\
  -H "Accept: application/json" \\
  | jq '.'
echo

# 就緒檢查
echo "12. 就緒檢查"
curl -X GET \\
  "${BASE_URL}/health/ready" \\
  -H "Accept: application/json" \\
  | jq '.'
echo

# 存活檢查
echo "13. 存活檢查"
curl -X GET \\
  "${BASE_URL}/health/live" \\
  -H "Accept: application/json" \\
  | jq '.'
echo

echo "=== 測試完成 ==="
"""

    def generate_postman_collection(self) -> Dict[str, Any]:
        """生成 Postman 集合。

        Returns:
            Postman 集合定義
        """
        collection = {
            "info": {
                "name": "Chinese GraphRAG API",
                "description": "針對中文文件優化的知識圖譜檢索增強生成系統 API",
                "version": "1.0.0",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            },
            "variable": [
                {"key": "baseUrl", "value": "http://localhost:8000"},
                {"key": "apiPrefix", "value": "/api/v1"},
            ],
            "item": [
                {
                    "name": "Health Check",
                    "item": [
                        {
                            "name": "Basic Health Check",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{baseUrl}}/health",
                                    "host": ["{{baseUrl}}"],
                                    "path": ["health"],
                                },
                            },
                        },
                        {
                            "name": "Detailed Health Check",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{baseUrl}}/health/detailed",
                                    "host": ["{{baseUrl}}"],
                                    "path": ["health", "detailed"],
                                },
                            },
                        },
                    ],
                },
                {
                    "name": "Indexing",
                    "item": [
                        {
                            "name": "Create Index",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {"key": "Content-Type", "value": "application/json"}
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": json.dumps(
                                        {
                                            "input_path": "./documents",
                                            "output_path": "./data/output",
                                            "file_types": ["txt", "pdf", "docx"],
                                            "batch_size": 32,
                                            "force_rebuild": False,
                                        },
                                        indent=2,
                                        ensure_ascii=False,
                                    ),
                                },
                                "url": {
                                    "raw": "{{baseUrl}}{{apiPrefix}}/index",
                                    "host": ["{{baseUrl}}"],
                                    "path": ["{{apiPrefix}}", "index"],
                                },
                            },
                        }
                    ],
                },
                {
                    "name": "Query",
                    "item": [
                        {
                            "name": "Single Query",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {"key": "Content-Type", "value": "application/json"}
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": json.dumps(
                                        {
                                            "query": "什麼是人工智慧？",
                                            "query_type": "global_search",
                                            "max_tokens": 2000,
                                            "temperature": 0.7,
                                        },
                                        indent=2,
                                        ensure_ascii=False,
                                    ),
                                },
                                "url": {
                                    "raw": "{{baseUrl}}{{apiPrefix}}/query",
                                    "host": ["{{baseUrl}}"],
                                    "path": ["{{apiPrefix}}", "query"],
                                },
                            },
                        }
                    ],
                },
            ],
        }

        # 儲存 Postman 集合
        file_path = self.output_dir / "postman-collection.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(collection, f, indent=2, ensure_ascii=False)

        return collection

    def generate_complete_documentation(self):
        """生成完整的 API 文件。"""
        print("正在生成 API 文件...")

        # 生成 OpenAPI 規格
        openapi_json = self.save_openapi_spec("json")
        openapi_yaml = self.save_openapi_spec("yaml")
        print(f"✅ OpenAPI 規格已生成: {openapi_json}, {openapi_yaml}")

        # 生成客戶端範例
        examples = self.generate_client_examples()
        print(f"✅ 客戶端範例已生成: {len(examples)} 種語言")

        # 生成 Postman 集合
        self.generate_postman_collection()
        print("✅ Postman 集合已生成")

        # 生成 README
        self._generate_readme()
        print("✅ API 文件 README 已生成")

        print(f"\\n📁 所有文件已生成到: {self.output_dir}")

    def _generate_readme(self):
        """生成 API 文件的 README。"""
        readme_content = f"""# Chinese GraphRAG API 文件

## 概述

Chinese GraphRAG API 是一個針對中文文件優化的知識圖譜檢索增強生成系統的 REST API 介面。

## 文件結構

```
{self.output_dir}/
├── README.md                 # 本檔案
├── openapi.json             # OpenAPI 3.0 規格 (JSON 格式)
├── openapi.yaml             # OpenAPI 3.0 規格 (YAML 格式)
├── postman-collection.json  # Postman 測試集合
├── examples.py              # Python 客戶端範例
├── examples.js              # JavaScript 客戶端範例
└── examples.sh              # cURL 命令範例
```

## 快速開始

### 1. 啟動 API 服務

```bash
# 開發模式
uv run python -m chinese_graphrag.api.app

# 或使用 uvicorn
uv run uvicorn chinese_graphrag.api.app:app --reload --host 0.0.0.0 --port 8000
```

### 2. 檢查服務狀態

```bash
curl http://localhost:8000/health
```

### 3. 查看 API 文件

開啟瀏覽器訪問：
- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc

## API 端點概覽

### 健康檢查
- `GET /health` - 基本健康檢查
- `GET /health/detailed` - 詳細健康檢查
- `GET /health/ready` - 就緒檢查
- `GET /health/live` - 存活檢查

### 索引管理
- `POST /api/v1/index` - 建立文件索引
- `GET /api/v1/index/status/{{task_id}}` - 查詢索引狀態
- `DELETE /api/v1/index/{{task_id}}` - 取消索引任務

### 查詢服務
- `POST /api/v1/query` - 執行單一查詢
- `POST /api/v1/query/batch` - 執行批次查詢
- `GET /api/v1/query/suggestions` - 取得查詢建議
- `GET /api/v1/query/history` - 查詢歷史記錄

### 配置管理
- `GET /api/v1/config` - 取得系統配置
- `PUT /api/v1/config` - 更新系統配置
- `POST /api/v1/config/validate` - 驗證配置
- `POST /api/v1/config/reset` - 重設配置

### 監控管理
- `GET /api/v1/monitoring/metrics` - 取得系統指標
- `GET /api/v1/monitoring/alerts` - 取得系統警報
- `POST /api/v1/monitoring/performance-test` - 執行效能測試

## 使用範例

### Python 客戶端

```python
from examples import ChineseGraphRAGClient

client = ChineseGraphRAGClient("http://localhost:8000")

# 檢查健康狀態
health = client.check_health()
print(health)

# 建立索引
index_result = client.create_index(
    input_path="./documents",
    file_types=["txt", "pdf", "docx"]
)
print(index_result)

# 執行查詢
query_result = client.query("什麼是人工智慧？")
print(query_result)
```

### JavaScript 客戶端

```javascript
import ChineseGraphRAGClient from './examples.js';

const client = new ChineseGraphRAGClient('http://localhost:8000');

// 執行查詢
const result = await client.query({{
    query: '什麼是人工智慧？',
    queryType: 'global_search'
}});
console.log(result);
```

### cURL 命令

```bash
# 執行查詢
curl -X POST http://localhost:8000/api/v1/query \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "什麼是人工智慧？", "query_type": "global_search"}}'
```

## 錯誤處理

API 使用標準 HTTP 狀態碼：

- `200` - 成功
- `400` - 客戶端錯誤（請求格式錯誤）
- `404` - 資源不存在
- `422` - 驗證錯誤
- `500` - 伺服器內部錯誤

錯誤回應格式：

```json
{{
  "success": false,
  "error": {{
    "code": 400,
    "message": "錯誤描述",
    "type": "validation_error",
    "detail": "詳細錯誤資訊"
  }},
  "timestamp": 1703123456.789
}}
```

## 認證

目前 API 不需要認證，但建議在生產環境中啟用適當的認證機制。

## 效能考量

- 索引建立是異步操作，請使用任務 ID 查詢進度
- 大型查詢可能需要較長時間，建議設定適當的超時時間
- 批次查詢比多個單一查詢更有效率
- 使用適當的批次大小以最佳化效能

## 部署

### Docker 部署

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "chinese_graphrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 環境變數

- `CHINESE_GRAPHRAG_CONFIG_PATH` - 配置檔案路徑
- `CHINESE_GRAPHRAG_DATA_DIR` - 數據目錄路徑
- `CHINESE_GRAPHRAG_LOG_LEVEL` - 日誌級別

## 支援

如果您遇到問題或有任何疑問，請：

1. 查看本文件和 OpenAPI 規格
2. 檢查日誌檔案
3. 使用健康檢查端點診斷問題
4. 聯繫開發團隊

## 更新日誌

### v1.0.0 ({datetime.now().strftime("%Y-%m-%d")})

- 初始版本發布
- 支援完整的索引和查詢功能
- 提供健康檢查和監控端點
- 包含完整的 API 文件和範例程式碼

---

*本文件由 Chinese GraphRAG API 文件生成器自動生成*
"""

        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
