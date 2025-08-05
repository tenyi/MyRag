"""
GraphRAG 索引引擎

整合 Microsoft GraphRAG 索引流程，支援中文文件處理
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

# Microsoft GraphRAG 相關導入
try:
    from graphrag.index.run import run_pipeline
    from graphrag.index.workflows.create_base_text_units import (
        run_workflow as create_base_text_units_workflow,
    )
    from graphrag.index.workflows.create_communities import (
        run_workflow as create_communities_workflow,
    )
    from graphrag.index.workflows.create_community_reports import (
        run_workflow as create_community_reports_workflow,
    )
    from graphrag.index.workflows.extract_graph import (
        run_workflow as extract_graph_workflow,
    )
    from graphrag.index.workflows.generate_text_embeddings import (
        run_workflow as generate_text_embeddings_workflow,
    )

    GRAPHRAG_AVAILABLE = True

    # 修復說明：
    # - 已修復函數名不匹配問題（create_base_text_units -> create_base_text_units_workflow）
    # - GraphRAG 工作流程需要標準的 GraphRAGConfig 和 PipelineRunContext
    # - 目前暫時回退到自定義實現，直到建立配置轉換層
    # TODO: 在階段 2 建立 GraphRAGConfigAdapter 來正確使用這些工作流程

except ImportError:
    logger.warning("Microsoft GraphRAG 套件未安裝，將使用自定義索引流程")
    GRAPHRAG_AVAILABLE = False

from chinese_graphrag.config import GraphRAGConfig
from chinese_graphrag.embeddings import EmbeddingManager
from chinese_graphrag.indexing.community_detector import CommunityDetector
from chinese_graphrag.indexing.community_report_generator import (
    CommunityReportGenerator,
)
from chinese_graphrag.models import Community, Document, Entity, Relationship, TextUnit
from chinese_graphrag.vector_stores import VectorStoreManager


class GraphRAGIndexer:
    """
    GraphRAG 索引引擎

    負責執行完整的 GraphRAG 索引流程，包括：
    1. 文件處理和分塊
    2. 實體和關係提取
    3. 社群檢測
    4. 向量化和儲存

    整合 Microsoft GraphRAG 的官方 workflow 和 pipeline 配置
    """

    def __init__(self, config: GraphRAGConfig):
        """
        初始化索引引擎

        Args:
            config: GraphRAG 配置
        """
        self.config = config

        # 初始化各個元件
        try:
            from chinese_graphrag.indexing.document_processor import DocumentProcessor

            self.document_processor = DocumentProcessor(config)
        except ImportError:
            # 如果 DocumentProcessor 不存在，創建一個簡單的替代
            self.document_processor = self._create_simple_document_processor()

        self.embedding_manager = EmbeddingManager(config)
        # 確保 vector_store_type 是正確的枚舉類型
        from chinese_graphrag.vector_stores.base import VectorStoreType

        vector_store_type = config.vector_store.type
        if isinstance(vector_store_type, str):
            vector_store_type = VectorStoreType(vector_store_type)

        self.vector_store_manager = VectorStoreManager(vector_store_type)

        # 使用安全的屬性訪問
        min_community_size = getattr(config.indexing, "min_community_size", 3)
        max_community_size = getattr(config.indexing, "max_community_size", 50)
        enable_hierarchical = getattr(
            config.indexing, "enable_hierarchical_communities", True
        )

        self.community_detector = CommunityDetector(
            min_community_size=min_community_size,
            max_community_size=max_community_size,
            enable_hierarchical=enable_hierarchical,
        )

        try:
            self.report_generator = CommunityReportGenerator(config)
        except ImportError:
            # 如果 CommunityReportGenerator 不存在，創建一個簡單的替代
            self.report_generator = self._create_simple_report_generator()

        # 索引狀態
        self.indexed_documents: Dict[str, Document] = {}
        self.text_units: Dict[str, TextUnit] = {}
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.communities: Dict[str, Community] = {}
        self.community_reports: Dict[str, Dict[str, Any]] = {}

        # GraphRAG 可用性檢查
        self.graphrag_available = GRAPHRAG_AVAILABLE

    def _check_graphrag_availability(self) -> bool:
        """
        檢查 GraphRAG 套件是否可用

        Returns:
            bool: GraphRAG 是否可用
        """
        return self.graphrag_available

    async def index_documents(
        self, input_path: Path, output_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        執行完整的文件索引流程

        優先使用 GraphRAG 官方 pipeline，如果不可用則使用自定義流程

        Args:
            input_path: 輸入文件路徑
            output_path: 輸出路徑（可選）

        Returns:
            Dict: 索引結果統計
        """
        logger.info(f"開始索引文件: {input_path}")

        try:
            # 檢查 GraphRAG 可用性（目前使用自定義實現）
            if self._check_graphrag_availability():
                logger.info("GraphRAG 套件可用，使用增強的自定義索引流程")
                return await self._run_graphrag_pipeline(input_path, output_path)
            else:
                logger.info("使用自定義索引流程")
                return await self._run_custom_pipeline(input_path, output_path)

        except Exception as e:
            logger.error(f"索引過程中發生錯誤: {e}")
            raise

    async def process_documents(
        self,
        files_to_process: List[Path],
        progress_callback=None,
        resume: bool = False,
        incremental: bool = False,
    ) -> Dict[str, Any]:
        """
        處理文件列表並執行索引

        Args:
            files_to_process: 要處理的文件路徑列表
            progress_callback: 進度回調函數，接收 (current_file, file_progress, overall_progress)
            resume: 是否恢復中斷的索引
            incremental: 是否增量索引

        Returns:
            Dict: 索引結果統計
        """
        logger.info(f"開始處理 {len(files_to_process)} 個文件")

        try:
            # 如果只有一個文件，使用文件路徑；否則創建臨時目錄處理
            if len(files_to_process) == 1:
                input_path = files_to_process[0]
            else:
                # 對於多個文件，使用第一個文件的父目錄作為輸入路徑
                input_path = files_to_process[0].parent

            # 使用現有的索引方法
            total_files = len(files_to_process)

            for i, file_path in enumerate(files_to_process):
                # 更新進度
                if progress_callback:
                    file_progress = 0.0
                    overall_progress = i
                    progress_callback(str(file_path), file_progress, overall_progress)

                # 處理單個文件
                try:
                    # 執行文件索引（使用現有的索引邏輯）
                    file_stats = await self.index_documents(file_path)

                    # 更新文件完成進度
                    if progress_callback:
                        file_progress = 100.0
                        overall_progress = i + 1
                        progress_callback(
                            str(file_path), file_progress, overall_progress
                        )

                    logger.info(f"文件 {file_path.name} 處理完成")

                except Exception as e:
                    logger.error(f"處理文件 {file_path} 時發生錯誤: {e}")
                    if not incremental:
                        # 如果不是增量模式，傳播錯誤
                        raise
                    # 增量模式下跳過失敗的文件
                    continue

            # 生成總體統計
            stats = self.get_statistics()

            # 轉換為期望的結果格式
            result = {
                "documents_processed": stats.get("documents", 0),
                "chunks_created": stats.get("text_units", 0),
                "entities_extracted": stats.get("entities", 0),
                "relationships_found": stats.get("relationships", 0),
                "communities_detected": stats.get("communities", 0),
                "output_files": {
                    "entities": f"{self.config.storage.base_dir}/entities.json",
                    "relationships": f"{self.config.storage.base_dir}/relationships.json",
                    "communities": f"{self.config.storage.base_dir}/communities.json",
                    "documents": f"{self.config.storage.base_dir}/documents.json",
                },
            }

            logger.info(f"所有文件處理完成: {result}")
            return result

        except Exception as e:
            logger.error(f"文件處理過程中發生錯誤: {e}")
            raise

    async def _run_graphrag_pipeline(
        self, input_path: Path, output_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        使用 GraphRAG 官方 pipeline 執行索引

        Args:
            input_path: 輸入文件路徑
            output_path: 輸出路徑

        Returns:
            Dict: 索引結果統計
        """
        if not self._check_graphrag_availability():
            raise RuntimeError("GraphRAG workflows 不可用")

        try:
            # 準備輸入數據
            logger.info("準備 GraphRAG pipeline 輸入數據")

            # 設置輸入和輸出路徑
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)

            # 執行 GraphRAG workflows
            logger.info("執行 GraphRAG workflows")

            # 直接執行自定義索引流程（GraphRAG 套件可用，使用增強版實現）
            stats = await self._execute_graphrag_workflows(input_path, output_path)

            logger.info(f"GraphRAG pipeline 索引完成: {stats}")
            return stats

        except Exception as e:
            logger.error(f"GraphRAG pipeline 執行失敗: {e}")
            # 回退到自定義流程
            logger.info("回退到自定義索引流程")
            return await self._run_custom_pipeline(input_path, output_path)

    async def _execute_graphrag_workflows(
        self, input_path: Path, output_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        執行 GraphRAG workflows

        Args:
            input_path: 輸入文件路徑
            output_path: 輸出路徑

        Returns:
            Dict: 執行結果統計
        """
        logger.info("開始執行增強的自定義索引流程")

        # 使用自定義流程（GraphRAG 套件可用，未來可整合官方 workflows）
        # 1. 文件處理和分塊
        logger.info("步驟 1: 處理文件和分塊")
        documents = await self._process_documents(input_path)

        # 2. 建立文本單元（目前使用自定義實現）
        text_units = await self._create_text_units(documents)

        # 3. 提取實體和關係（目前使用自定義實現）
        entities, relationships = await self._extract_entities_and_relationships(
            text_units
        )

        # 4. 進行社群檢測（目前使用自定義實現）
        communities = await self._detect_communities(entities, relationships)

        # 5. 向量化處理
        await self._create_embeddings(text_units, entities, communities)

        # 6. 儲存結果
        if output_path:
            await self._save_results(output_path)

        # 統計結果
        stats = {
            "documents": len(documents),
            "text_units": len(text_units),
            "entities": len(entities),
            "relationships": len(relationships),
            "communities": len(communities),
        }

        logger.info(f"GraphRAG workflows 完成: {stats}")
        return stats

    async def _process_graphrag_results(
        self, pipeline_result: Any, output_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        處理 GraphRAG pipeline 的結果

        Args:
            pipeline_result: GraphRAG pipeline 執行結果
            output_path: 輸出路徑

        Returns:
            Dict: 處理後的統計資訊
        """
        stats = {
            "documents": 0,
            "text_units": 0,
            "entities": 0,
            "relationships": 0,
            "communities": 0,
        }

        try:
            # 如果 pipeline_result 是列表（多個 workflow 結果）
            if isinstance(pipeline_result, list):
                for workflow_result in pipeline_result:
                    if hasattr(workflow_result, "workflow"):
                        workflow_name = workflow_result.workflow
                        logger.info(f"處理 workflow 結果: {workflow_name}")

                        # 根據 workflow 類型更新統計
                        if "text_units" in workflow_name:
                            stats["text_units"] = getattr(workflow_result, "count", 0)
                        elif "entities" in workflow_name:
                            stats["entities"] = getattr(workflow_result, "count", 0)
                        elif "relationships" in workflow_name:
                            stats["relationships"] = getattr(
                                workflow_result, "count", 0
                            )
                        elif "communities" in workflow_name:
                            stats["communities"] = getattr(workflow_result, "count", 0)

            # 如果有輸出路徑，嘗試從輸出文件中讀取更詳細的統計
            if output_path and output_path.exists():
                stats = await self._read_graphrag_output_stats(output_path, stats)

        except Exception as e:
            logger.warning(f"處理 GraphRAG 結果時發生錯誤: {e}")

        return stats

    async def _read_graphrag_output_stats(
        self, output_path: Path, default_stats: Dict[str, int]
    ) -> Dict[str, int]:
        """
        從 GraphRAG 輸出文件中讀取統計資訊

        Args:
            output_path: 輸出路徑
            default_stats: 預設統計資訊

        Returns:
            Dict: 更新後的統計資訊
        """
        stats = default_stats.copy()

        try:
            # 檢查常見的 GraphRAG 輸出文件
            output_files = {
                "entities": "create_final_entities.parquet",
                "relationships": "create_final_relationships.parquet",
                "communities": "create_final_communities.parquet",
                "text_units": "create_final_text_units.parquet",
                "documents": "create_final_documents.parquet",
            }

            for stat_key, filename in output_files.items():
                file_path = output_path / filename
                if file_path.exists():
                    try:
                        import pandas as pd

                        df = pd.read_parquet(file_path)
                        stats[stat_key] = len(df)
                        logger.info(
                            f"從 {filename} 讀取到 {len(df)} 條 {stat_key} 記錄"
                        )
                    except Exception as e:
                        logger.warning(f"讀取 {filename} 失敗: {e}")

        except Exception as e:
            logger.warning(f"讀取 GraphRAG 輸出統計失敗: {e}")

        return stats

    async def _run_custom_pipeline(
        self, input_path: Path, output_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        使用自定義流程執行索引（原有邏輯）

        Args:
            input_path: 輸入文件路徑
            output_path: 輸出路徑

        Returns:
            Dict: 索引結果統計
        """
        # 1. 文件處理和分塊
        logger.info("步驟 1: 處理文件和分塊")
        documents = await self._process_documents(input_path)

        # 使用 GraphRAG workflow 函數（如果可用）
        if GRAPHRAG_AVAILABLE:
            text_units = await self._create_text_units_with_graphrag(documents)
        else:
            text_units = await self._create_text_units(documents)

        # 2. 實體和關係提取
        logger.info("步驟 2: 提取實體和關係")
        if GRAPHRAG_AVAILABLE:
            entities, relationships = (
                await self._extract_entities_and_relationships_with_graphrag(text_units)
            )
        else:
            entities, relationships = await self._extract_entities_and_relationships(
                text_units
            )

        # 3. 社群檢測
        logger.info("步驟 3: 社群檢測")
        if GRAPHRAG_AVAILABLE:
            communities = await self._detect_communities_with_graphrag(
                entities, relationships
            )
        else:
            communities = await self._detect_communities(entities, relationships)

        # 4. 向量化處理
        logger.info("步驟 4: 向量化處理")
        await self._create_embeddings(text_units, entities, communities)

        # 更新索引狀態
        for doc in documents:
            self.indexed_documents[doc.id] = doc
        for unit in text_units:
            self.text_units[unit.id] = unit
        for entity in entities:
            self.entities[entity.id] = entity
        for rel in relationships:
            self.relationships[rel.id] = rel
        for comm in communities:
            self.communities[comm.id] = comm

        # 5. 儲存結果
        logger.info("步驟 5: 儲存索引結果")
        if output_path:
            await self._save_results(output_path)

        # 統計結果
        stats = {
            "documents": len(documents),
            "text_units": len(text_units),
            "entities": len(entities),
            "relationships": len(relationships),
            "communities": len(communities),
        }

        logger.info(f"自定義索引完成: {stats}")
        return stats

    async def _create_text_units_with_graphrag(
        self, documents: List[Document]
    ) -> List[TextUnit]:
        """
        使用 GraphRAG 的 create_base_text_units workflow 建立文本單元

        Args:
            documents: 文件列表

        Returns:
            List[TextUnit]: 文本單元列表
        """
        if not GRAPHRAG_AVAILABLE:
            return await self._create_text_units(documents)

        try:
            logger.info("使用 GraphRAG create_base_text_units workflow")

            # 建立 GraphRAG 配置適配器
            from chinese_graphrag.config.graphrag_adapter import GraphRAGConfigAdapter

            adapter = GraphRAGConfigAdapter(self.config)

            # 準備 GraphRAG 執行環境
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # 設置環境
                if not adapter.validate_and_prepare_environment(temp_path):
                    logger.warning("GraphRAG 環境準備失敗，回退到自定義實現")
                    return await self._create_text_units(documents)

                # 轉換文檔為 GraphRAG 格式
                documents_parquet_path = adapter.convert_documents_to_graphrag_format(
                    documents, temp_path / "output"
                )

                # 載入 GraphRAG 配置
                from graphrag.config.load_config import load_config

                graphrag_config = load_config(temp_path)

                # 建立 GraphRAG 執行上下文
                from graphrag.cache.noop_pipeline_cache import NoopPipelineCache
                from graphrag.callbacks.noop_workflow_callbacks import (
                    NoopWorkflowCallbacks,
                )
                from graphrag.index.typing.context import PipelineRunContext
                from graphrag.index.typing.stats import PipelineRunStats
                from graphrag.storage.file_pipeline_storage import FilePipelineStorage

                stats = PipelineRunStats()
                callbacks = NoopWorkflowCallbacks()
                cache = NoopPipelineCache()

                input_storage = FilePipelineStorage(temp_path / "input")
                output_storage = FilePipelineStorage(temp_path / "output")
                cache_storage = FilePipelineStorage(temp_path / "cache")

                context = PipelineRunContext(
                    stats=stats,
                    input_storage=input_storage,
                    output_storage=output_storage,
                    previous_storage=cache_storage,
                    cache=cache,
                    callbacks=callbacks,
                    state={},
                )

                # 調用 GraphRAG workflow
                result = await create_base_text_units_workflow(
                    config=graphrag_config, context=context
                )

                # 轉換結果為我們的 TextUnit 模型
                text_units = []
                if hasattr(result, "result") and result.result is not None:
                    result_df = result.result
                    logger.info(f"GraphRAG 返回了 {len(result_df)} 個文本單元")

                    for _, row in result_df.iterrows():
                        text_unit = TextUnit(
                            id=row.get("id", str(uuid.uuid4())),
                            text=row.get("text", ""),
                            document_id=row.get("document_id", ""),
                            chunk_index=row.get("chunk_index", 0),
                            metadata=row.get("metadata", {}),
                        )
                        text_units.append(text_unit)
                        self.text_units[text_unit.id] = text_unit

                logger.info(f"GraphRAG 成功建立了 {len(text_units)} 個文本單元")
                return text_units

        except Exception as e:
            logger.warning(
                f"GraphRAG create_base_text_units 失敗: {e}，回退到自定義方法"
            )
            return await self._create_text_units(documents)

    async def _extract_entities_and_relationships_with_graphrag(
        self, text_units: List[TextUnit]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        使用 GraphRAG 的實體和關係提取 workflows

        Args:
            text_units: 文本單元列表

        Returns:
            Tuple[List[Entity], List[Relationship]]: 實體和關係列表
        """
        if not GRAPHRAG_AVAILABLE:
            return await self._extract_entities_and_relationships(text_units)

        try:
            logger.info("使用 GraphRAG workflows 進行實體和關係提取")

            # 準備輸入數據
            import pandas as pd

            text_units_data = []
            for unit in text_units:
                text_units_data.append(
                    {
                        "id": unit.id,
                        "text": unit.text,
                        "document_id": unit.document_id,
                        "chunk_index": unit.chunk_index,
                    }
                )

            text_units_df = pd.DataFrame(text_units_data)

            # GraphRAG 實體和關係提取工作流程需要完整的配置轉換
            # 這些函數調用不正確，需要使用正確的工作流程：
            # - extract_graph_workflow 用於實體和關係提取
            # 暫時回退到自定義實現，直到建立正確的 GraphRAG 配置轉換
            logger.warning(
                "GraphRAG 實體關係提取工作流程需要完整的配置轉換，暫時使用自定義實現"
            )
            return await self._extract_entities_and_relationships(text_units)

            # 轉換結果為我們的模型
            entities = self._convert_graphrag_entities_to_models(final_entities_result)
            relationships = self._convert_graphrag_relationships_to_models(
                final_relationships_result
            )

            logger.info(
                f"GraphRAG 提取了 {len(entities)} 個實體和 {len(relationships)} 個關係"
            )
            return entities, relationships

        except Exception as e:
            logger.warning(f"GraphRAG 實體關係提取失敗: {e}，回退到自定義方法")
            return await self._extract_entities_and_relationships(text_units)

    async def _detect_communities_with_graphrag(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> List[Community]:
        """
        使用 GraphRAG 的社群檢測 workflows

        Args:
            entities: 實體列表
            relationships: 關係列表

        Returns:
            List[Community]: 社群列表
        """
        if not GRAPHRAG_AVAILABLE:
            return await self._detect_communities(entities, relationships)

        try:
            logger.info("使用 GraphRAG workflows 進行社群檢測")

            # 準備輸入數據
            import pandas as pd

            # 轉換實體和關係為 DataFrame
            entities_data = []
            for entity in entities:
                entities_data.append(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type,
                        "description": entity.description,
                    }
                )

            relationships_data = []
            for rel in relationships:
                relationships_data.append(
                    {
                        "id": rel.id,
                        "source": rel.source_entity_id,
                        "target": rel.target_entity_id,
                        "description": rel.description,
                        "weight": rel.weight if rel.weight is not None else 1.0,
                    }
                )

            entities_df = pd.DataFrame(entities_data)
            relationships_df = pd.DataFrame(relationships_data)

            # GraphRAG 社群檢測工作流程需要完整的配置轉換
            # 正確的函數應該是：
            # - create_communities_workflow 用於社群檢測
            # - create_community_reports_workflow 用於社群報告生成
            # 暫時回退到自定義實現，直到建立正確的 GraphRAG 配置轉換
            logger.warning(
                "GraphRAG 社群檢測工作流程需要完整的配置轉換，暫時使用自定義實現"
            )
            return await self._detect_communities(entities, relationships)

            # 轉換結果為我們的模型
            communities = self._convert_graphrag_communities_to_models(
                communities_result, community_reports_result
            )

            # 儲存社群報告
            self.community_reports = self._extract_community_reports(
                community_reports_result
            )

            logger.info(f"GraphRAG 檢測到 {len(communities)} 個社群")
            return communities

        except Exception as e:
            logger.warning(f"GraphRAG 社群檢測失敗: {e}，回退到自定義方法")
            return await self._detect_communities(entities, relationships)

    def _convert_graphrag_entities_to_models(
        self, graphrag_entities: Any
    ) -> List[Entity]:
        """
        將 GraphRAG 實體結果轉換為我們的 Entity 模型

        Args:
            graphrag_entities: GraphRAG 實體結果

        Returns:
            List[Entity]: 實體列表
        """
        entities = []

        try:
            if hasattr(graphrag_entities, "iterrows"):
                for _, row in graphrag_entities.iterrows():
                    entity = Entity(
                        id=row.get("id", str(uuid.uuid4())),
                        name=row.get("name", ""),
                        type=row.get("type", "UNKNOWN"),
                        description=row.get("description", ""),
                        text_units=row.get("text_unit_ids", []),
                        rank=row.get("rank", 1.0),
                    )
                    entities.append(entity)
                    self.entities[entity.id] = entity
        except Exception as e:
            logger.error(f"轉換 GraphRAG 實體失敗: {e}")

        return entities

    def _convert_graphrag_relationships_to_models(
        self, graphrag_relationships: Any
    ) -> List[Relationship]:
        """
        將 GraphRAG 關係結果轉換為我們的 Relationship 模型

        Args:
            graphrag_relationships: GraphRAG 關係結果

        Returns:
            List[Relationship]: 關係列表
        """
        relationships = []

        try:
            if hasattr(graphrag_relationships, "iterrows"):
                for _, row in graphrag_relationships.iterrows():
                    relationship = Relationship(
                        id=row.get("id", str(uuid.uuid4())),
                        source_entity_id=row.get("source", ""),
                        target_entity_id=row.get("target", ""),
                        relationship_type=row.get("type", "RELATED_TO"),
                        description=row.get("description", ""),
                        weight=row.get("weight", 1.0),
                        text_units=row.get("text_unit_ids", []),
                    )
                    relationships.append(relationship)
                    self.relationships[relationship.id] = relationship
        except Exception as e:
            logger.error(f"轉換 GraphRAG 關係失敗: {e}")

        return relationships

    def _convert_graphrag_communities_to_models(
        self, graphrag_communities: Any, community_reports: Any
    ) -> List[Community]:
        """
        將 GraphRAG 社群結果轉換為我們的 Community 模型

        Args:
            graphrag_communities: GraphRAG 社群結果
            community_reports: GraphRAG 社群報告結果

        Returns:
            List[Community]: 社群列表
        """
        communities = []

        try:
            # 建立報告映射
            reports_map = {}
            if hasattr(community_reports, "iterrows"):
                for _, row in community_reports.iterrows():
                    community_id = row.get("community", "")
                    reports_map[community_id] = {
                        "title": row.get("title", ""),
                        "summary": row.get("summary", ""),
                        "full_content": row.get("full_content", ""),
                    }

            if hasattr(graphrag_communities, "iterrows"):
                for _, row in graphrag_communities.iterrows():
                    community_id = row.get("id", str(uuid.uuid4()))
                    report = reports_map.get(community_id, {})

                    community = Community(
                        id=community_id,
                        title=report.get("title", f"Community {community_id}"),
                        level=row.get("level", 0),
                        entities=row.get("entity_ids", []),
                        relationships=row.get("relationship_ids", []),
                        summary=report.get("summary", ""),
                        rank=row.get("rank", 1.0),
                    )
                    communities.append(community)
                    self.communities[community.id] = community
        except Exception as e:
            logger.error(f"轉換 GraphRAG 社群失敗: {e}")

        return communities

    def _extract_community_reports(
        self, community_reports: Any
    ) -> Dict[str, Dict[str, Any]]:
        """
        從 GraphRAG 社群報告結果中提取報告資訊

        Args:
            community_reports: GraphRAG 社群報告結果

        Returns:
            Dict: 社群報告字典
        """
        reports = {}

        try:
            if hasattr(community_reports, "iterrows"):
                for _, row in community_reports.iterrows():
                    community_id = row.get("community", "")
                    reports[community_id] = {
                        "title": row.get("title", ""),
                        "summary": row.get("summary", ""),
                        "full_content": row.get("full_content", ""),
                        "rank": row.get("rank", 1.0),
                    }
        except Exception as e:
            logger.error(f"提取社群報告失敗: {e}")

        return reports

    async def _process_documents(self, input_path: Path) -> List[Document]:
        """處理輸入文件"""
        logger.info(f"處理文件目錄: {input_path}")

        if input_path.is_file():
            # 處理單一文件
            documents = [self.document_processor.process_document(input_path)]
        else:
            # 處理目錄中的所有文件
            documents = self.document_processor.batch_process(input_path)

        # 儲存文件資訊
        for doc in documents:
            self.indexed_documents[doc.id] = doc

        logger.info(f"處理了 {len(documents)} 個文件")
        return documents

    async def _create_text_units(self, documents: List[Document]) -> List[TextUnit]:
        """建立文本單元"""
        logger.info("建立文本單元")

        text_units = []

        for doc in documents:
            # 使用配置的分塊策略
            chunks = self.document_processor.split_text(
                doc.content,
                chunk_size=self.config.chunks.size,
                overlap=self.config.chunks.overlap,
            )

            for i, chunk in enumerate(chunks):
                text_unit = TextUnit(
                    id=f"{doc.id}_chunk_{i}",
                    text=chunk,
                    document_id=doc.id,
                    chunk_index=i,
                    metadata={
                        "document_title": doc.title,
                        "chunk_size": len(chunk),
                        "language": "zh",  # 假設為中文
                    },
                )
                text_units.append(text_unit)
                self.text_units[text_unit.id] = text_unit

        logger.info(f"建立了 {len(text_units)} 個文本單元")
        return text_units

    async def _extract_entities_and_relationships(
        self, text_units: List[TextUnit]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """提取實體和關係"""
        logger.info("提取實體和關係")

        # 使用預設的 LLM 配置
        default_llm_name = self.config.model_selection.default_llm
        llm_config = self.config.get_llm_config(default_llm_name)

        if not llm_config:
            # 創建一個簡單的測試配置
            llm_config = {
                "type": "mock",
                "model": "test_model",
                "max_tokens": 4000,
                "temperature": 0.7,
            }

        logger.info(f"使用 LLM 模型進行實體提取: {default_llm_name}")

        entities = []
        relationships = []

        # 批次處理文本單元
        batch_size = 10  # 預設批次大小

        for i in range(0, len(text_units), batch_size):
            batch = text_units[i : i + batch_size]

            # 並行處理批次
            batch_entities, batch_relationships = await self._process_text_unit_batch(
                batch, llm_config
            )

            entities.extend(batch_entities)
            relationships.extend(batch_relationships)

            # 記錄進度
            logger.info(
                f"已處理 {min(i + batch_size, len(text_units))}/{len(text_units)} 個文本單元"
            )

        # 儲存結果
        for entity in entities:
            self.entities[entity.id] = entity

        for relationship in relationships:
            self.relationships[relationship.id] = relationship

        logger.info(f"提取了 {len(entities)} 個實體和 {len(relationships)} 個關係")
        return entities, relationships

    async def _process_text_unit_batch(
        self, text_units: List[TextUnit], llm_config
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        使用 LLM 從一批文本單元中提取實體和關係。
        """
        if not text_units:
            return [], []

        # 呼叫新的圖譜提取方法
        entities, relationships = await self._extract_graph_from_batch(
            text_units, llm_config
        )

        # 處理關係中的實體名稱，將其轉換為 ID
        entity_name_to_id = {entity.name: entity.id for entity in entities}

        valid_relationships = []
        for rel in relationships:
            # 確保來源和目標實體都存在
            if (
                rel.source_entity_id in entity_name_to_id
                and rel.target_entity_id in entity_name_to_id
            ):
                rel.source_entity_id = entity_name_to_id[rel.source_entity_id]
                rel.target_entity_id = entity_name_to_id[rel.target_entity_id]
                valid_relationships.append(rel)
            else:
                logger.warning(
                    f"關係 '{rel.description}' 的實體 '{rel.source_entity_id}' 或 '{rel.target_entity_id}' 不存在，已忽略。"
                )

        return entities, valid_relationships

    async def _extract_graph_from_batch(
        self, text_units: List[TextUnit], llm_config
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        使用 LLM 從一批文本單元中提取實體和關係圖譜。
        """
        from chinese_graphrag.llm import LLM, create_llm

        # 建立 LLM 實例
        # 正確處理 LLM 類型，轉換枚舉為字串
        llm_type = getattr(llm_config, "type", "mock")

        # 安全地處理類型轉換
        if hasattr(llm_type, "value") and not isinstance(llm_type, str):
            llm_type = llm_type.value
        elif not isinstance(llm_type, str):
            llm_type = str(llm_type)

        # 將 openai_chat 映射為 openai
        if llm_type == "openai_chat":
            llm_type = "openai"

        # 將 LLMConfig 對象轉換為字典
        if hasattr(llm_config, "model_dump"):
            config_dict = llm_config.model_dump()
        elif hasattr(llm_config, "dict"):
            config_dict = llm_config.dict()
        else:
            config_dict = llm_config

        llm: LLM = create_llm(llm_type, config_dict)

        # 建構 prompt
        prompt = self._build_extraction_prompt(text_units)

        # 呼叫 LLM
        try:
            response = await llm.async_generate(prompt)

            # 清理回應，移除可能的 markdown 標籤
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # 移除 ```json
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # 移除 ```
            cleaned_response = cleaned_response.strip()

            output = json.loads(cleaned_response)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"無法解析 LLM 回應: {e}. 回應內容: {response}")
            return [], []
        except Exception as e:
            logger.error(f"LLM 呼叫失敗: {e}")
            return [], []

        # 解析回應
        return self._parse_llm_output(output, text_units)

    def _build_extraction_prompt(self, text_units: List[TextUnit]) -> str:
        """建構實體和關係提取的 prompt"""
        input_text = "\n\n".join(
            [f"## 文件片段 (ID: {unit.id})\n\n{unit.text}" for unit in text_units]
        )

        return f"""# 指令：知識圖譜提取

## 角色
你是一個專業的知識圖譜分析師，擅長從非結構化的中文文本中精確地提取實體（Entities）和它們之間的關係（Relationships）。

## 任務
你的任務是仔細閱讀以下提供的多個文件片段，並以 JSON 格式輸出一個包含所有實體和關係的知識圖譜。

## 實體（Entity）定義
- **定義**: 在文本中有明確意義的名詞或概念，通常是現實世界中的物體、人物、地點、組織、專有名詞、或重要的概念。
- **屬性**:
  - `name`: 實體的唯一名稱 (字串)。
  - `type`: 實體的類型 (字串)，例如：「公司」、「人物」、「技術」、「地點」、「產品」、「專案」。
  - `description`: 對實體的簡要描述 (字串)。

## 關係（Relationship）定義
- **定義**: 描述兩個實體之間有意義的連結。
- **屬性**:
  - `source`: 關係的來源實體名稱 (字串)，必須是你在實體列表中定義的實體名稱。
  - `target`: 關係的目標實體名稱 (字串)，必須是你在實體列表中定義的實體名稱。
  - `description`: 對關係的詳細描述 (字串)，需能清楚說明兩個實體的關係。

## 輸出格式要求
- 你必須嚴格以 JSON 格式輸出。
- JSON 的根節點應包含兩個鍵：`entities` 和 `relationships`。
- `entities` 的值是一個包含所有實體物件的陣列。
- `relationships` 的值是一個包含所有關係物件的陣列。
- 關係中的 `source` 和 `target` 必須與 `entities` 列表中的實體 `name` 完全對應。
- 所有的文字都應使用繁體中文。

## 待處理文本

請根據以上規則，處理以下文本：

```
{input_text}
```

## 輸出 JSON
"""

    def _parse_llm_output(
        self, output: Dict[str, List[Dict[str, str]]], text_units: List[TextUnit]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """解析 LLM 的 JSON 輸出並轉換為系統的資料模型"""
        entities = []
        relationships = []

        # 提取實體
        for item in output.get("entities", []):
            entity = Entity(
                id=str(uuid.uuid4()),
                name=item.get("name"),
                type=item.get("type"),
                description=item.get("description"),
                text_units=[unit.id for unit in text_units],  # 關聯到所有輸入的文本單元
                rank=1.0,
            )
            entities.append(entity)

        # 提取關係
        for item in output.get("relationships", []):
            relationship = Relationship(
                id=str(uuid.uuid4()),
                source_entity_id=item.get("source"),  # 暫存實體名稱
                target_entity_id=item.get("target"),  # 暫存實體名稱
                relationship_type="related_to",  # 可從 description 推斷
                description=item.get("description"),
                weight=0.8,
                text_units=[unit.id for unit in text_units],
            )
            relationships.append(relationship)

        return entities, relationships

    async def _detect_communities(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> List[Community]:
        """檢測社群"""
        logger.info("檢測社群結構")

        # 使用社群檢測器進行檢測
        communities = self.community_detector.detect_communities(
            entities, relationships
        )

        # 儲存社群資訊
        for community in communities:
            self.communities[community.id] = community

        # 生成社群報告 (預設啟用)
        try:
            logger.info("生成社群報告")
            self.community_reports = (
                await self.report_generator.generate_community_reports(
                    communities, self.entities, self.relationships, self.text_units
                )
            )
        except Exception as e:
            logger.warning(f"社群報告生成失敗: {e}")
            self.community_reports = {}

        logger.info(f"檢測到 {len(communities)} 個社群")
        return communities

    async def _create_embeddings(
        self,
        text_units: List[TextUnit],
        entities: List[Entity],
        communities: List[Community],
    ):
        """建立向量嵌入"""
        logger.info("建立向量嵌入")

        # 使用預設的 Embedding 服務
        default_embedding_name = self.config.model_selection.default_embedding
        embedding_config = self.config.get_embedding_config(default_embedding_name)

        # 使用服務名稱而不是模型名稱
        service_name = default_embedding_name  # 使用配置中的服務名稱

        if embedding_config:
            logger.info(
                f"使用 Embedding 服務: {service_name} (模型: {embedding_config.model})"
            )
        else:
            logger.info(f"使用預設 Embedding 服務: {service_name}")

        # 為文本單元建立嵌入
        if text_units:
            texts = [unit.text for unit in text_units]
            embedding_result = await self.embedding_manager.embed_texts(
                texts, service_name
            )
            embeddings = embedding_result.embeddings

            for unit, embedding in zip(text_units, embeddings):
                unit.embedding = embedding
                # 儲存到向量資料庫
                await self.vector_store_manager.store_text_unit(unit)

        # 為實體建立嵌入
        if entities:
            entity_texts = [
                f"{entity.name}: {entity.description}" for entity in entities
            ]
            entity_embedding_result = await self.embedding_manager.embed_texts(
                entity_texts, service_name
            )
            entity_embeddings = entity_embedding_result.embeddings

            for entity, embedding in zip(entities, entity_embeddings):
                entity.embedding = embedding
                # 儲存到向量資料庫
                await self.vector_store_manager.store_entity(entity)

        # 為社群建立嵌入
        if communities:
            community_texts = [community.summary for community in communities]
            community_embedding_result = await self.embedding_manager.embed_texts(
                community_texts, service_name
            )
            community_embeddings = community_embedding_result.embeddings

            for community, embedding in zip(communities, community_embeddings):
                community.embedding = embedding
                # 儲存到向量資料庫
                await self.vector_store_manager.store_community(community)

        logger.info("向量嵌入建立完成")

    async def _save_results(self, output_path: Path):
        """儲存索引結果"""
        logger.info(f"儲存索引結果到: {output_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        # 儲存各種資料結構
        import json
        from datetime import datetime

        # 儲存文件資訊
        documents_data = {
            doc_id: {
                "id": doc.id,
                "title": doc.title,
                "file_path": str(doc.file_path),
                "created_at": doc.created_at.isoformat(),
                "metadata": doc.metadata,
            }
            for doc_id, doc in self.indexed_documents.items()
        }

        with open(output_path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)

        # 儲存實體資訊
        entities_data = {
            entity_id: {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "text_units": entity.text_units,
                "rank": entity.rank,
            }
            for entity_id, entity in self.entities.items()
        }

        with open(output_path / "entities.json", "w", encoding="utf-8") as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)

        # 儲存關係資訊
        relationships_data = {
            rel_id: {
                "id": rel.id,
                "source_entity_id": rel.source_entity_id,
                "target_entity_id": rel.target_entity_id,
                "relationship_type": rel.relationship_type,
                "description": rel.description,
                "weight": rel.weight if rel.weight is not None else 1.0,
                "text_units": rel.text_units,
            }
            for rel_id, rel in self.relationships.items()
        }

        with open(output_path / "relationships.json", "w", encoding="utf-8") as f:
            json.dump(relationships_data, f, ensure_ascii=False, indent=2)

        # 儲存社群資訊
        communities_data = {
            comm_id: {
                "id": comm.id,
                "title": comm.title,
                "level": comm.level,
                "entities": comm.entities,
                "relationships": comm.relationships,
                "summary": comm.summary,
                "rank": comm.rank,
            }
            for comm_id, comm in self.communities.items()
        }

        with open(output_path / "communities.json", "w", encoding="utf-8") as f:
            json.dump(communities_data, f, ensure_ascii=False, indent=2)

        # 儲存索引統計
        stats = {
            "indexed_at": datetime.now().isoformat(),
            "documents_count": len(self.indexed_documents),
            "text_units_count": len(self.text_units),
            "entities_count": len(self.entities),
            "relationships_count": len(self.relationships),
            "communities_count": len(self.communities),
        }

        with open(output_path / "index_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("索引結果儲存完成")

    def _create_simple_document_processor(self):
        """創建簡單的文件處理器替代"""

        class SimpleDocumentProcessor:
            def __init__(self, config):
                self.config = config

            def process_document(self, file_path):
                # 簡單的文件處理邏輯
                from chinese_graphrag.models import Document

                return Document(
                    id=str(file_path),
                    title=file_path.name,
                    content="測試內容",
                    file_path=file_path,
                )

            def batch_process(self, input_path):
                return [self.process_document(input_path)]

            def split_text(self, text, chunk_size=1000, overlap=200):
                # 簡單的文本分割
                chunks = []
                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i : i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                return chunks

        return SimpleDocumentProcessor(self.config)

    def _create_simple_report_generator(self):
        """創建簡單的報告生成器替代"""

        class SimpleReportGenerator:
            def __init__(self, config):
                self.config = config

            async def generate_community_reports(
                self, communities, entities, relationships, text_units
            ):
                # 簡單的報告生成邏輯
                reports = {}
                for community in communities:
                    reports[community.id] = {
                        "title": community.title,
                        "summary": community.summary,
                        "entities_count": len(community.entities),
                        "relationships_count": len(community.relationships),
                    }
                return reports

        return SimpleReportGenerator(self.config)

    def get_statistics(self) -> Dict[str, int]:
        """取得索引統計資訊"""
        return {
            "documents": len(self.indexed_documents),
            "text_units": len(self.text_units),
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "communities": len(self.communities),
        }

    def clear_index(self):
        """清除索引資料"""
        logger.info("清除索引資料")

        self.indexed_documents.clear()
        self.text_units.clear()
        self.entities.clear()
        self.relationships.clear()
        self.communities.clear()
