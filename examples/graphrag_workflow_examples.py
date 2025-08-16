"""
GraphRAG Workflow 調用範例

展示如何正確調用 Microsoft GraphRAG 的工作流程
"""

import asyncio
import os
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

# GraphRAG 相關導入
from graphrag.config.load_config import load_config
from graphrag.index.workflows.create_base_text_units import run_workflow as create_base_text_units_workflow
from graphrag.index.workflows.extract_graph import run_workflow as extract_graph_workflow
from graphrag.index.workflows.create_communities import run_workflow as create_communities_workflow
from graphrag.index.workflows.create_community_reports import run_workflow as create_community_reports_workflow

# Context 和 utilities
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.stats import PipelineRunStats
from graphrag.storage.file_pipeline_storage import FilePipelineStorage
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.cache.noop_pipeline_cache import NoopPipelineCache


class GraphRAGWorkflowExample:
    """
    GraphRAG 工作流程調用範例
    
    展示如何建立正確的配置和上下文來調用 Microsoft GraphRAG workflows
    """
    
    def __init__(self, root_dir: Path):
        """
        初始化 GraphRAG 工作流程範例
        
        Args:
            root_dir: GraphRAG 項目根目錄，包含 settings.yaml
        """
        self.root_dir = root_dir
        self.config = None
        self.context = None
    
    async def setup(self):
        """設置 GraphRAG 配置和執行上下文"""
        try:
            # 載入 GraphRAG 配置
            logger.info(f"載入 GraphRAG 配置從: {self.root_dir}")
            self.config = load_config(self.root_dir)
            logger.info("GraphRAG 配置載入成功")
            
            # 建立執行上下文
            stats = PipelineRunStats()
            callbacks = NoopWorkflowCallbacks()
            cache = NoopPipelineCache()
            
            # 建立 storage
            input_storage = FilePipelineStorage(self.root_dir / "input")
            output_storage = FilePipelineStorage(self.root_dir / "output")
            cache_storage = FilePipelineStorage(self.root_dir / "cache")
            
            self.context = PipelineRunContext(
                stats=stats,
                input_storage=input_storage,
                output_storage=output_storage,
                previous_storage=cache_storage,
                cache=cache,
                callbacks=callbacks,
                state={}
            )
            
            logger.info("GraphRAG 執行上下文建立成功")
            
        except Exception as e:
            logger.error(f"GraphRAG 設置失敗: {e}")
            raise
    
    async def create_text_units_example(self, documents_data: List[dict]):
        """
        範例：使用 GraphRAG create_base_text_units workflow
        
        Args:
            documents_data: 文檔數據列表，每個項目包含 id, title, text, metadata
        """
        try:
            logger.info("開始執行 create_base_text_units workflow")
            
            # 準備輸入數據
            documents_df = pd.DataFrame(documents_data)
            
            # 更新 context state
            self.context.state['input'] = documents_df
            
            # 調用 workflow
            result = await create_base_text_units_workflow(
                config=self.config,
                context=self.context
            )
            
            logger.info(f"create_base_text_units workflow 完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"create_base_text_units workflow 失敗: {e}")
            raise
    
    async def extract_graph_example(self):
        """
        範例：使用 GraphRAG extract_graph workflow
        需要先有 text_units 數據
        """
        try:
            logger.info("開始執行 extract_graph workflow")
            
            # 調用 workflow
            result = await extract_graph_workflow(
                config=self.config,
                context=self.context
            )
            
            logger.info(f"extract_graph workflow 完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"extract_graph workflow 失敗: {e}")
            raise
    
    async def create_communities_example(self):
        """
        範例：使用 GraphRAG create_communities workflow
        需要先有 entities 和 relationships 數據
        """
        try:
            logger.info("開始執行 create_communities workflow")
            
            # 調用 workflow
            result = await create_communities_workflow(
                config=self.config,
                context=self.context
            )
            
            logger.info(f"create_communities workflow 完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"create_communities workflow 失敗: {e}")
            raise
    
    async def create_community_reports_example(self):
        """
        範例：使用 GraphRAG create_community_reports workflow
        需要先有 communities 數據
        """
        try:
            logger.info("開始執行 create_community_reports workflow")
            
            # 調用 workflow
            result = await create_community_reports_workflow(
                config=self.config,
                context=self.context
            )
            
            logger.info(f"create_community_reports workflow 完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"create_community_reports workflow 失敗: {e}")
            raise
    
    async def run_full_pipeline_example(self, documents_data: List[dict]):
        """
        範例：執行完整的 GraphRAG pipeline
        
        Args:
            documents_data: 輸入文檔數據
        """
        try:
            logger.info("開始執行完整的 GraphRAG pipeline")
            
            # 1. 建立文本單元
            text_units_result = await self.create_text_units_example(documents_data)
            
            # 2. 提取圖結構（實體和關係）
            graph_result = await self.extract_graph_example()
            
            # 3. 檢測社群
            communities_result = await self.create_communities_example()
            
            # 4. 生成社群報告
            reports_result = await self.create_community_reports_example()
            
            logger.info("完整的 GraphRAG pipeline 執行完成")
            
            return {
                "text_units": text_units_result,
                "graph": graph_result,
                "communities": communities_result,
                "reports": reports_result
            }
            
        except Exception as e:
            logger.error(f"GraphRAG pipeline 執行失敗: {e}")
            raise


async def main():
    """主函數 - 示範如何使用 GraphRAG workflows"""
    
    # 設置測試環境
    os.environ['GRAPHRAG_API_KEY'] = 'test-key-for-demo'
    
    # 準備測試數據
    test_documents = [
        {
            "id": "doc1",
            "title": "人工智慧基礎",
            "text": "人工智慧（AI）是一門研究如何讓機器模擬人類智慧的學科。深度學習是AI的重要分支。",
            "metadata": {"source": "test", "category": "ai"}
        },
        {
            "id": "doc2", 
            "title": "機器學習發展",
            "text": "機器學習是人工智慧的核心技術。神經網路和深度學習推動了現代AI的發展。",
            "metadata": {"source": "test", "category": "ml"}
        }
    ]
    
    try:
        # 假設我們有一個正確的 GraphRAG 配置目錄
        # 在實際使用中，這應該指向包含 settings.yaml 的目錄
        root_dir = Path("./examples/graphrag_test")
        
        # 建立 workflow 範例
        example = GraphRAGWorkflowExample(root_dir)
        
        # 設置
        await example.setup()
        
        # 執行單個 workflow 測試
        logger.info("測試 create_base_text_units workflow...")
        text_units_result = await example.create_text_units_example(test_documents)
        
        logger.info("GraphRAG workflow 範例執行完成")
        
    except Exception as e:
        logger.error(f"範例執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())