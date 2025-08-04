#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
精簡查詢 API 測試腳本

測試新增的精簡查詢端點功能
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any


class SimpleQueryAPITester:
    """精簡查詢 API 測試器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_simple_query(self, query: str, use_llm_segmentation: bool = True) -> Dict[str, Any]:
        """測試精簡查詢端點"""
        url = f"{self.base_url}/api/query/simple"
        
        payload = {
            "query": query,
            "search_type": "auto",
            "use_llm_segmentation": use_llm_segmentation
        }
        
        print(f"\n🔍 測試精簡查詢: {query}")
        print(f"   使用 LLM 分詞: {use_llm_segmentation}")
        print(f"   請求 URL: {url}")
        
        start_time = time.time()
        
        try:
            async with self.session.post(url, json=payload) as response:
                result = await response.json()
                elapsed_time = time.time() - start_time
                
                print(f"   回應狀態: {response.status}")
                print(f"   API 回應時間: {elapsed_time:.3f}s")
                
                if response.status == 200 and result.get("success"):
                    print(f"   ✅ 查詢成功")
                    print(f"   📄 答案: {result['answer'][:100]}...")
                    print(f"   🎯 信心度: {result['confidence']:.2f}")
                    print(f"   🔧 搜尋類型: {result['search_type']}")
                    print(f"   ⏱️  系統回應時間: {result['response_time']}s")
                else:
                    print(f"   ❌ 查詢失敗: {result}")
                
                return result
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"   ❌ 請求失敗: {e}")
            print(f"   ⏱️  失敗時間: {elapsed_time:.3f}s")
            return {"error": str(e)}
    
    async def test_simple_query_with_reasoning(self, query: str) -> Dict[str, Any]:
        """測試包含推理的精簡查詢端點"""
        url = f"{self.base_url}/api/query/simple/with-reasoning"
        
        payload = {
            "query": query,
            "search_type": "auto",
            "use_llm_segmentation": True
        }
        
        print(f"\n🧠 測試精簡查詢（含推理）: {query}")
        print(f"   請求 URL: {url}")
        
        start_time = time.time()
        
        try:
            async with self.session.post(url, json=payload) as response:
                result = await response.json()
                elapsed_time = time.time() - start_time
                
                print(f"   回應狀態: {response.status}")
                print(f"   API 回應時間: {elapsed_time:.3f}s")
                
                if response.status == 200 and result.get("success"):
                    print(f"   ✅ 查詢成功")
                    print(f"   📄 答案: {result['answer'][:100]}...")
                    print(f"   🎯 信心度: {result['confidence']:.2f}")
                    print(f"   🔧 搜尋類型: {result['search_type']}")
                    print(f"   ⏱️  系統回應時間: {result['response_time']}s")
                    
                    if result.get("reasoning_path"):
                        print(f"   🤔 推理路徑:")
                        for i, step in enumerate(result["reasoning_path"], 1):
                            print(f"      {i}. {step}")
                else:
                    print(f"   ❌ 查詢失敗: {result}")
                
                return result
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"   ❌ 請求失敗: {e}")
            print(f"   ⏱️  失敗時間: {elapsed_time:.3f}s")
            return {"error": str(e)}
    
    async def compare_query_methods(self, query: str):
        """比較不同查詢方法的結果"""
        print(f"\n📊 比較查詢方法: {query}")
        print("=" * 60)
        
        # 測試精簡查詢（LLM 分詞）
        result_llm = await self.test_simple_query(query, use_llm_segmentation=True)
        
        # 測試精簡查詢（jieba 分詞）
        result_jieba = await self.test_simple_query(query, use_llm_segmentation=False)
        
        # 測試含推理的精簡查詢
        result_reasoning = await self.test_simple_query_with_reasoning(query)
        
        # 比較結果
        print(f"\n📈 結果比較:")
        if all(r.get("success") for r in [result_llm, result_jieba, result_reasoning]):
            print(f"   LLM 分詞信心度: {result_llm.get('confidence', 0):.2f}")
            print(f"   jieba 分詞信心度: {result_jieba.get('confidence', 0):.2f}")
            print(f"   含推理信心度: {result_reasoning.get('confidence', 0):.2f}")
            print(f"   LLM 分詞回應時間: {result_llm.get('response_time', 0):.3f}s")
            print(f"   jieba 分詞回應時間: {result_jieba.get('response_time', 0):.3f}s")
            print(f"   含推理回應時間: {result_reasoning.get('response_time', 0):.3f}s")
        else:
            print("   ⚠️  部分查詢失敗，無法進行完整比較")


async def main():
    """主測試函數"""
    print("🚀 精簡查詢 API 測試開始")
    print("=" * 60)
    
    # 測試查詢列表
    test_queries = [
        "小明姓什麼",
        "什麼是人工智慧",
        "機器學習的應用有哪些",
        "GraphRAG 是什麼"
    ]
    
    async with SimpleQueryAPITester() as tester:
        for query in test_queries:
            await tester.compare_query_methods(query)
            print("\n" + "="*60)
        
        # 單獨測試推理功能
        print("\n🧪 專項測試：推理功能")
        await tester.test_simple_query_with_reasoning("小明的職業是什麼")
    
    print("\n✅ 測試完成")


if __name__ == "__main__":
    asyncio.run(main())