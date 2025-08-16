"""
測試修改後的 GraphRAG 索引引擎
"""

import asyncio
import os
from pathlib import Path

from loguru import logger
from chinese_graphrag.config.loader import load_config
from chinese_graphrag.indexing.engine import GraphRAGIndexer
from chinese_graphrag.models.document import Document


async def test_modified_indexer():
    """測試修改後的索引引擎"""
    
    try:
        # 設置測試環境
        os.environ['GRAPHRAG_API_KEY'] = 'test-key-for-demo'
        
        # 載入配置
        config = load_config("config/settings.yaml")
        logger.info(f"配置載入成功: {type(config)}")
        
        # 建立索引引擎
        indexer = GraphRAGIndexer(config)
        logger.info("索引引擎建立成功")
        
        # 準備測試文檔
        test_documents = [
            Document(
                id="doc1",
                title="人工智慧基礎",
                content="人工智慧（AI）是一門研究如何讓機器模擬人類智慧的學科。深度學習是AI的重要分支，它通過多層神經網路來模擬人腦的工作方式。",
                file_path="test/doc1.txt",
                metadata={"source": "test", "category": "ai"}
            ),
            Document(
                id="doc2",
                title="機器學習發展",
                content="機器學習是人工智慧的核心技術。神經網路和深度學習推動了現代AI的發展。圖靈（Alan Turing）被稱為人工智慧之父。",
                file_path="test/doc2.txt",
                metadata={"source": "test", "category": "ml"}
            )
        ]
        
        # 測試 GraphRAG workflow
        logger.info("開始測試 GraphRAG text units workflow...")
        text_units = await indexer._create_text_units_with_graphrag(test_documents)
        
        logger.info(f"✅ GraphRAG workflow 成功建立了 {len(text_units)} 個文本單元")
        
        # 顯示結果
        for i, unit in enumerate(text_units[:3]):  # 只顯示前3個
            logger.info(f"文本單元 {i+1}: {unit.text[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_modified_indexer())
    if success:
        print("✅ 修改後的索引引擎測試成功")
    else:
        print("❌ 修改後的索引引擎測試失敗")