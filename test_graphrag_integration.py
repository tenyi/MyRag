"""
簡單的 GraphRAG 測試，展示正確的工作流程調用
"""

import asyncio
import os
from pathlib import Path

import pandas as pd
from loguru import logger

from graphrag.config.load_config import load_config
from graphrag.index.workflows.create_base_text_units import run_workflow as create_base_text_units_workflow
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.stats import PipelineRunStats
from graphrag.storage.file_pipeline_storage import FilePipelineStorage
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.cache.noop_pipeline_cache import NoopPipelineCache


async def test_graphrag_integration():
    """測試 GraphRAG 整合的基本功能"""
    
    # 設置測試環境
    os.environ['GRAPHRAG_API_KEY'] = 'test-key-for-demo'
    
    try:
        # 使用正確的 GraphRAG 配置目錄
        config_dir = Path("examples/graphrag_test")
        
        logger.info(f"載入 GraphRAG 配置從: {config_dir}")
        config = load_config(config_dir)
        logger.info("GraphRAG 配置載入成功")
        
        # 建立執行上下文
        stats = PipelineRunStats()
        callbacks = NoopWorkflowCallbacks()
        cache = NoopPipelineCache()
        
        # 建立 storage
        input_storage = FilePipelineStorage(config_dir / "input")
        output_storage = FilePipelineStorage(config_dir / "output")
        cache_storage = FilePipelineStorage(config_dir / "cache")
        
        # 準備測試數據
        test_documents = [
            {
                "id": "doc1",
                "title": "AI 測試文檔",
                "text": "人工智慧（AI）是一門研究如何讓機器模擬人類智慧的學科。深度學習技術使得AI在各個領域都取得了顯著的進步。圖靈（Alan Turing）是人工智慧之父。",
                "metadata": {"source": "test", "category": "ai"}
            }
        ]
        
        documents_df = pd.DataFrame(test_documents)
        
        context = PipelineRunContext(
            stats=stats,
            input_storage=input_storage,
            output_storage=output_storage,
            previous_storage=cache_storage,
            cache=cache,
            callbacks=callbacks,
            state={'input': documents_df}
        )
        
        logger.info("GraphRAG 執行上下文建立成功")
        
        # 測試 create_base_text_units workflow
        logger.info("測試 create_base_text_units workflow...")
        
        try:
            result = await create_base_text_units_workflow(
                config=config,
                context=context
            )
            
            logger.info(f"create_base_text_units workflow 成功: {result}")
            
            # 檢查輸出
            if hasattr(result, 'result') and result.result is not None:
                logger.info(f"輸出數據類型: {type(result.result)}")
                if hasattr(result.result, 'shape'):
                    logger.info(f"輸出數據形狀: {result.result.shape}")
                
        except Exception as e:
            logger.error(f"create_base_text_units workflow 失敗: {e}")
            raise
        
        logger.info("GraphRAG 整合測試完成")
        return True
        
    except Exception as e:
        logger.error(f"GraphRAG 整合測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_graphrag_integration())
    if success:
        print("✅ GraphRAG 整合測試成功")
    else:
        print("❌ GraphRAG 整合測試失敗")