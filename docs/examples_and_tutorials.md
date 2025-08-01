# 程式碼範例和教學

本文件提供 Chinese GraphRAG 系統的詳細使用範例和教學，幫助您快速上手並掌握系統的各種功能。

## 目錄

- [快速開始](#快速開始)
- [基礎教學](#基礎教學)
- [進階範例](#進階範例)
- [API 使用範例](#api-使用範例)
- [自訂擴展](#自訂擴展)
- [效能優化](#效能優化)
- [實際應用案例](#實際應用案例)
- [常見問題解答](#常見問題解答)

## 快速開始

### 1. 最簡單的使用範例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chinese GraphRAG 快速開始範例
"""

from chinese_graphrag import ChineseGraphRAG

def main():
    # 初始化系統
    graphrag = ChineseGraphRAG(config_path="config/settings.yaml")
    
    # 索引文件
    print("開始索引文件...")
    graphrag.index(
        input_path="./documents",
        output_path="./data"
    )
    print("索引完成！")
    
    # 執行查詢
    question = "什麼是人工智慧？"
    result = graphrag.query(question)
    
    print(f"問題：{question}")
    print(f"回答：{result.answer}")
    print(f"來源：{[source.document for source in result.sources]}")

if __name__ == "__main__":
    main()
```

### 2. 命令列快速開始

```bash
# 1. 初始化專案
uv run chinese-graphrag init

# 2. 準備文件
mkdir documents
echo "人工智慧是模擬人類智慧的技術。" > documents/ai.txt
echo "機器學習是人工智慧的一個分支。" > documents/ml.txt

# 3. 索引文件
uv run chinese-graphrag index --input documents --output data

# 4. 執行查詢
uv run chinese-graphrag query "什麼是人工智慧？"
```

## 基礎教學

### 教學 1：文件處理和索引

#### 1.1 準備文件

```python
import os
from pathlib import Path

# 建立測試文件
documents_dir = Path("tutorial_documents")
documents_dir.mkdir(exist_ok=True)

# 建立中文文件
documents = {
    "ai_basics.txt": """
    人工智慧（Artificial Intelligence, AI）是電腦科學的一個分支，
    致力於建立能夠執行通常需要人類智慧的任務的系統。
    AI 的主要目標是讓機器能夠學習、推理、感知和決策。
    """,
    
    "machine_learning.txt": """
    機器學習（Machine Learning, ML）是人工智慧的一個重要分支。
    它使用統計技術讓電腦系統能夠從資料中學習，
    而無需明確程式設計每個任務的執行方式。
    """,
    
    "deep_learning.txt": """
    深度學習（Deep Learning, DL）是機器學習的一個子領域。
    它基於人工神經網路，特別是深層神經網路。
    深度學習在影像識別、自然語言處理等領域取得了重大突破。
    """
}

for filename, content in documents.items():
    with open(documents_dir / filename, 'w', encoding='utf-8') as f:
        f.write(content.strip())

print(f"建立了 {len(documents)} 個測試文件")
```

#### 1.2 配置系統

```python
from chinese_graphrag.config import GraphRAGConfig

# 建立配置
config = GraphRAGConfig(
    # 模型配置
    models={
        "chat_model": {
            "type": "openai_chat",
            "model": "gpt-4o-mini",
            "api_key": "your-api-key"
        },
        "embedding_model": {
            "type": "bge_m3",
            "model": "BAAI/bge-m3",
            "device": "auto"
        }
    },
    
    # 文本處理配置
    text_processing={
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "language": "zh"
    },
    
    # 向量資料庫配置
    vector_store={
        "type": "lancedb",
        "uri": "./tutorial_data/lancedb"
    }
)

# 儲存配置
config.save("tutorial_config.yaml")
```

#### 1.3 執行索引

```python
from chinese_graphrag import ChineseGraphRAG
from chinese_graphrag.monitoring import get_logger

# 設定日誌
logger = get_logger(__name__)

def run_indexing():
    """執行文件索引"""
    
    # 初始化系統
    graphrag = ChineseGraphRAG(config_path="tutorial_config.yaml")
    
    try:
        # 開始索引
        logger.info("開始索引文件...")
        
        result = graphrag.index(
            input_path="tutorial_documents",
            output_path="tutorial_data",
            show_progress=True
        )
        
        logger.info(f"索引完成！處理了 {result.documents_count} 個文件")
        logger.info(f"提取了 {result.entities_count} 個實體")
        logger.info(f"發現了 {result.relationships_count} 個關係")
        
        return result
        
    except Exception as e:
        logger.error(f"索引失敗：{e}")
        raise

if __name__ == "__main__":
    result = run_indexing()
    print("索引結果：", result)
```

### 教學 2：查詢和檢索

#### 2.1 基本查詢

```python
from chinese_graphrag import ChineseGraphRAG

def basic_query_example():
    """基本查詢範例"""
    
    # 初始化系統
    graphrag = ChineseGraphRAG(config_path="tutorial_config.yaml")
    
    # 準備查詢問題
    questions = [
        "什麼是人工智慧？",
        "機器學習和深度學習有什麼區別？",
        "AI 的主要應用領域有哪些？"
    ]
    
    for question in questions:
        print(f"\n問題：{question}")
        print("-" * 50)
        
        # 執行查詢
        result = graphrag.query(question)
        
        print(f"回答：{result.answer}")
        print(f"搜尋類型：{result.search_type}")
        print(f"回應時間：{result.response_time:.2f}秒")
        
        # 顯示來源
        if result.sources:
            print("來源文件：")
            for i, source in enumerate(result.sources[:3], 1):
                print(f"  {i}. {source.document} (相關度: {source.relevance_score:.2f})")

if __name__ == "__main__":
    basic_query_example()
```

#### 2.2 進階查詢選項

```python
from chinese_graphrag import ChineseGraphRAG
from chinese_graphrag.query import QueryConfig

def advanced_query_example():
    """進階查詢範例"""
    
    graphrag = ChineseGraphRAG(config_path="tutorial_config.yaml")
    
    # 自訂查詢配置
    query_config = QueryConfig(
        search_type="global",  # 或 "local", "auto"
        max_tokens=1500,
        temperature=0.1,
        top_k=10,
        similarity_threshold=0.7
    )
    
    question = "深度學習在哪些領域有重要應用？"
    
    # 執行查詢
    result = graphrag.query(
        question=question,
        config=query_config
    )
    
    print(f"問題：{question}")
    print(f"回答：{result.answer}")
    
    # 顯示詳細資訊
    print(f"\n查詢詳情：")
    print(f"- 搜尋類型：{result.search_type}")
    print(f"- 使用的模型：{result.model_used}")
    print(f"- Token 使用量：{result.tokens_used}")
    print(f"- 處理時間：{result.response_time:.2f}秒")
    
    # 顯示中間結果
    if hasattr(result, 'intermediate_results'):
        print(f"\n中間結果：")
        for step, data in result.intermediate_results.items():
            print(f"- {step}: {len(data)} 項")

if __name__ == "__main__":
    advanced_query_example()
```

### 教學 3：批次處理

#### 3.1 批次索引

```python
import asyncio
from pathlib import Path
from chinese_graphrag import ChineseGraphRAG
from chinese_graphrag.indexing import BatchIndexer

async def batch_indexing_example():
    """批次索引範例"""
    
    # 準備多個文件目錄
    directories = [
        "documents/tech",
        "documents/business", 
        "documents/science"
    ]
    
    # 建立批次索引器
    batch_indexer = BatchIndexer(
        config_path="tutorial_config.yaml",
        max_workers=4,
        batch_size=10
    )
    
    # 執行批次索引
    results = []
    for directory in directories:
        if Path(directory).exists():
            print(f"索引目錄：{directory}")
            
            result = await batch_indexer.index_directory(
                input_path=directory,
                output_path=f"data/{Path(directory).name}"
            )
            
            results.append(result)
            print(f"完成：{result.documents_count} 個文件")
    
    # 合併結果
    total_docs = sum(r.documents_count for r in results)
    total_entities = sum(r.entities_count for r in results)
    
    print(f"\n批次索引完成：")
    print(f"- 總文件數：{total_docs}")
    print(f"- 總實體數：{total_entities}")

if __name__ == "__main__":
    asyncio.run(batch_indexing_example())
```

#### 3.2 批次查詢

```python
from chinese_graphrag import ChineseGraphRAG
from concurrent.futures import ThreadPoolExecutor
import time

def batch_query_example():
    """批次查詢範例"""
    
    graphrag = ChineseGraphRAG(config_path="tutorial_config.yaml")
    
    # 準備查詢列表
    questions = [
        "什麼是人工智慧？",
        "機器學習的主要類型有哪些？",
        "深度學習的優勢是什麼？",
        "AI 在醫療領域的應用？",
        "自然語言處理的挑戰？"
    ]
    
    # 單執行緒查詢
    print("單執行緒查詢：")
    start_time = time.time()
    
    single_results = []
    for question in questions:
        result = graphrag.query(question)
        single_results.append(result)
    
    single_time = time.time() - start_time
    print(f"完成時間：{single_time:.2f}秒")
    
    # 多執行緒查詢
    print("\n多執行緒查詢：")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        multi_results = list(executor.map(graphrag.query, questions))
    
    multi_time = time.time() - start_time
    print(f"完成時間：{multi_time:.2f}秒")
    print(f"加速比：{single_time/multi_time:.2f}x")
    
    # 顯示結果
    for i, (question, result) in enumerate(zip(questions, multi_results)):
        print(f"\n{i+1}. {question}")
        print(f"   {result.answer[:100]}...")

if __name__ == "__main__":
    batch_query_example()
```

## 進階範例

### 範例 1：自訂 Embedding 模型

```python
from chinese_graphrag.embeddings import EmbeddingService
import numpy as np
from sentence_transformers import SentenceTransformer

class CustomChineseEmbedding(EmbeddingService):
    """自訂中文 Embedding 服務"""
    
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """編碼文本列表"""
        # 預處理中文文本
        processed_texts = [self._preprocess_chinese(text) for text in texts]
        
        # 生成 embeddings
        embeddings = self.model.encode(
            processed_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        return embeddings
    
    def _preprocess_chinese(self, text: str) -> str:
        """中文文本預處理"""
        import re
        
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 保留中文字符、數字和基本標點
        text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：]', '', text)
        
        return text
    
    def get_dimension(self) -> int:
        """獲取向量維度"""
        return self.model.get_sentence_embedding_dimension()

# 使用自訂 embedding
def use_custom_embedding():
    """使用自訂 embedding 範例"""
    
    from chinese_graphrag.config import GraphRAGConfig
    from chinese_graphrag import ChineseGraphRAG
    
    # 建立自訂 embedding 服務
    custom_embedding = CustomChineseEmbedding()
    
    # 配置系統使用自訂 embedding
    config = GraphRAGConfig(
        models={
            "embedding_model": {
                "type": "custom",
                "service": custom_embedding
            }
        }
    )
    
    # 初始化系統
    graphrag = ChineseGraphRAG(config=config)
    
    # 測試 embedding
    texts = ["這是一個測試", "人工智慧很有趣"]
    embeddings = custom_embedding.encode_texts(texts)
    
    print(f"Embedding 維度：{embeddings.shape}")
    print(f"相似度：{np.dot(embeddings[0], embeddings[1]):.4f}")

if __name__ == "__main__":
    use_custom_embedding()
```

### 範例 2：自訂查詢處理器

```python
from chinese_graphrag.query import QueryProcessor
from chinese_graphrag.models import QueryResult
import jieba
import re

class EnhancedChineseQueryProcessor(QueryProcessor):
    """增強的中文查詢處理器"""
    
    def __init__(self):
        super().__init__()
        # 載入自訂詞典
        jieba.load_userdict("custom_dict.txt")
        
        # 定義查詢類型關鍵詞
        self.query_type_keywords = {
            "definition": ["什麼是", "定義", "含義", "意思"],
            "comparison": ["區別", "差異", "比較", "不同"],
            "application": ["應用", "用途", "使用", "實例"],
            "process": ["如何", "怎樣", "步驟", "流程"]
        }
    
    def process_query(self, question: str) -> dict:
        """處理查詢問題"""
        
        # 基本預處理
        processed_question = self._preprocess_question(question)
        
        # 提取關鍵詞
        keywords = self._extract_keywords(processed_question)
        
        # 判斷查詢類型
        query_type = self._determine_query_type(processed_question)
        
        # 生成搜尋策略
        search_strategy = self._generate_search_strategy(query_type, keywords)
        
        return {
            "original_question": question,
            "processed_question": processed_question,
            "keywords": keywords,
            "query_type": query_type,
            "search_strategy": search_strategy
        }
    
    def _preprocess_question(self, question: str) -> str:
        """預處理問題"""
        # 移除多餘標點
        question = re.sub(r'[？?！!。.]+$', '', question.strip())
        
        # 統一問號
        question = re.sub(r'[？?]', '？', question)
        
        return question
    
    def _extract_keywords(self, question: str) -> list[str]:
        """提取關鍵詞"""
        # 使用 jieba 分詞
        words = jieba.cut(question)
        
        # 過濾停用詞和標點
        stopwords = {"的", "是", "在", "有", "和", "與", "或", "但", "而"}
        keywords = [word for word in words 
                   if len(word) > 1 and word not in stopwords]
        
        return keywords
    
    def _determine_query_type(self, question: str) -> str:
        """判斷查詢類型"""
        for query_type, keywords in self.query_type_keywords.items():
            if any(keyword in question for keyword in keywords):
                return query_type
        
        return "general"
    
    def _generate_search_strategy(self, query_type: str, keywords: list[str]) -> dict:
        """生成搜尋策略"""
        strategies = {
            "definition": {
                "search_type": "local",
                "focus": "entities",
                "boost_keywords": keywords[:3]
            },
            "comparison": {
                "search_type": "global", 
                "focus": "relationships",
                "boost_keywords": keywords
            },
            "application": {
                "search_type": "global",
                "focus": "communities",
                "boost_keywords": keywords
            },
            "general": {
                "search_type": "auto",
                "focus": "mixed",
                "boost_keywords": keywords[:5]
            }
        }
        
        return strategies.get(query_type, strategies["general"])

# 使用自訂查詢處理器
def use_custom_query_processor():
    """使用自訂查詢處理器範例"""
    
    processor = EnhancedChineseQueryProcessor()
    
    test_questions = [
        "什麼是機器學習？",
        "深度學習和機器學習有什麼區別？",
        "人工智慧在醫療領域的應用有哪些？",
        "如何訓練一個神經網路？"
    ]
    
    for question in test_questions:
        result = processor.process_query(question)
        
        print(f"\n問題：{question}")
        print(f"類型：{result['query_type']}")
        print(f"關鍵詞：{result['keywords']}")
        print(f"搜尋策略：{result['search_strategy']}")

if __name__ == "__main__":
    use_custom_query_processor()
```

## API 使用範例

### 範例 1：REST API 客戶端

```python
import requests
import json
from typing import Dict, List, Optional

class ChineseGraphRAGClient:
    """Chinese GraphRAG API 客戶端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
        
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> Dict:
        """檢查 API 健康狀態"""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()
    
    def create_index_task(self, input_path: str, output_path: str, config: Optional[Dict] = None) -> Dict:
        """建立索引任務"""
        data = {
            "input_path": input_path,
            "output_path": output_path
        }
        
        if config:
            data["config"] = config
        
        response = self.session.post(f"{self.base_url}/api/v1/index", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_index_status(self, task_id: str) -> Dict:
        """獲取索引任務狀態"""
        response = self.session.get(f"{self.base_url}/api/v1/index/{task_id}/status")
        response.raise_for_status()
        return response.json()
    
    def query(self, question: str, search_type: str = "auto", config: Optional[Dict] = None) -> Dict:
        """執行查詢"""
        data = {
            "question": question,
            "search_type": search_type
        }
        
        if config:
            data["config"] = config
        
        response = self.session.post(f"{self.base_url}/api/v1/query", json=data)
        response.raise_for_status()
        return response.json()
    
    def batch_query(self, questions: List[str], search_type: str = "auto", config: Optional[Dict] = None) -> Dict:
        """批次查詢"""
        data = {
            "questions": questions,
            "search_type": search_type
        }
        
        if config:
            data["config"] = config
        
        response = self.session.post(f"{self.base_url}/api/v1/query/batch", json=data)
        response.raise_for_status()
        return response.json()

# 使用 API 客戶端
def api_client_example():
    """API 客戶端使用範例"""
    
    # 建立客戶端
    client = ChineseGraphRAGClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"
    )
    
    try:
        # 檢查健康狀態
        health = client.health_check()
        print(f"API 狀態：{health['status']}")
        
        # 建立索引任務
        index_task = client.create_index_task(
            input_path="./documents",
            output_path="./data",
            config={"chunk_size": 1000}
        )
        
        task_id = index_task["task_id"]
        print(f"索引任務 ID：{task_id}")
        
        # 等待索引完成
        import time
        while True:
            status = client.get_index_status(task_id)
            print(f"索引狀態：{status['status']} ({status.get('progress', {}).get('percentage', 0):.1f}%)")
            
            if status["status"] in ["completed", "failed"]:
                break
            
            time.sleep(5)
        
        if status["status"] == "completed":
            # 執行查詢
            result = client.query("什麼是人工智慧？")
            print(f"查詢結果：{result['answer']}")
            
            # 批次查詢
            questions = ["什麼是機器學習？", "深度學習的應用？"]
            batch_results = client.batch_query(questions)
            
            for i, result in enumerate(batch_results["results"]):
                print(f"問題 {i+1}：{questions[i]}")
                print(f"回答：{result['answer'][:100]}...")
        
    except requests.RequestException as e:
        print(f"API 請求錯誤：{e}")

if __name__ == "__main__":
    api_client_example()
```

### 範例 2：異步 API 客戶端

```python
import asyncio
import aiohttp
from typing import Dict, List, Optional

class AsyncChineseGraphRAGClient:
    """異步 Chinese GraphRAG API 客戶端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
        
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    async def query(self, session: aiohttp.ClientSession, question: str, **kwargs) -> Dict:
        """執行查詢"""
        data = {"question": question, **kwargs}
        
        async with session.post(
            f"{self.base_url}/api/v1/query",
            json=data,
            headers=self.headers
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def batch_query_concurrent(self, questions: List[str], max_concurrent: int = 5) -> List[Dict]:
        """並發批次查詢"""
        
        async with aiohttp.ClientSession() as session:
            # 建立信號量限制並發數
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def query_with_semaphore(question: str) -> Dict:
                async with semaphore:
                    return await self.query(session, question)
            
            # 並發執行查詢
            tasks = [query_with_semaphore(q) for q in questions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 處理結果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "question": questions[i],
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results

# 使用異步客戶端
async def async_client_example():
    """異步客戶端使用範例"""
    
    client = AsyncChineseGraphRAGClient()
    
    # 準備大量查詢
    questions = [
        "什麼是人工智慧？",
        "機器學習的類型有哪些？",
        "深度學習的優勢？",
        "自然語言處理的挑戰？",
        "計算機視覺的應用？",
        "強化學習的原理？",
        "神經網路的結構？",
        "AI 的倫理問題？"
    ]
    
    import time
    start_time = time.time()
    
    # 並發查詢
    results = await client.batch_query_concurrent(questions, max_concurrent=3)
    
    end_time = time.time()
    
    print(f"並發查詢完成，耗時：{end_time - start_time:.2f}秒")
    
    # 顯示結果
    for i, result in enumerate(results):
        if "error" in result:
            print(f"{i+1}. 錯誤：{result['error']}")
        else:
            print(f"{i+1}. {questions[i]}")
            print(f"   {result['answer'][:100]}...")

if __name__ == "__main__":
    asyncio.run(async_client_example())
```

## 自訂擴展

### 範例 1：自訂文件處理器

```python
from chinese_graphrag.processors import DocumentProcessor
from chinese_graphrag.models import Document
import json
from pathlib import Path

class JSONDocumentProcessor(DocumentProcessor):
    """JSON 文件處理器"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.json']
    
    def can_process(self, file_path: str) -> bool:
        """檢查是否可以處理該檔案"""
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def process_file(self, file_path: str) -> Document:
        """處理 JSON 檔案"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取文本內容
        content = self._extract_text_from_json(data)
        
        # 建立文件物件
        document = Document(
            id=self._generate_id(file_path),
            title=data.get('title', Path(file_path).stem),
            content=content,
            metadata={
                'file_path': file_path,
                'file_type': 'json',
                'original_data': data
            }
        )
        
        return document
    
    def _extract_text_from_json(self, data: dict) -> str:
        """從 JSON 資料中提取文本"""
        text_parts = []
        
        def extract_text(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and len(value.strip()) > 0:
                        text_parts.append(f"{prefix}{key}: {value
