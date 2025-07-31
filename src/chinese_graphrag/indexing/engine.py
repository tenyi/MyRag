"""
GraphRAG 索引引擎

整合 Microsoft GraphRAG 索引流程，支援中文文件處理
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from loguru import logger

# Microsoft GraphRAG 相關導入
try:
    from graphrag.index import create_pipeline_config
    from graphrag.index.run import run_pipeline_with_config
    from graphrag.index.config import PipelineConfig
    from graphrag.index.workflows.v1.create_base_text_units import (
        create_base_text_units,
    )
    from graphrag.index.workflows.v1.create_base_extracted_entities import (
        create_base_extracted_entities,
    )
    from graphrag.index.workflows.v1.create_summarized_entities import (
        create_summarized_entities,
    )
    from graphrag.index.workflows.v1.create_base_entity_graph import (
        create_base_entity_graph,
    )
    from graphrag.index.workflows.v1.create_final_entities import (
        create_final_entities,
    )
    from graphrag.index.workflows.v1.create_final_relationships import (
        create_final_relationships,
    )
    from graphrag.index.workflows.v1.create_final_communities import (
        create_final_communities,
    )
    from graphrag.index.workflows.v1.create_final_community_reports import (
        create_final_community_reports,
    )
    GRAPHRAG_AVAILABLE = True
except ImportError:
    logger.warning("Microsoft GraphRAG 套件未安裝，將使用簡化的索引流程")
    GRAPHRAG_AVAILABLE = False

from chinese_graphrag.config import GraphRAGConfig
from chinese_graphrag.config.strategy import ModelSelector, TaskType
from chinese_graphrag.embeddings import EmbeddingManager
from chinese_graphrag.models import Community, Document, Entity, Relationship, TextUnit
from chinese_graphrag.vector_stores import VectorStoreManager
from chinese_graphrag.indexing.community_detector import CommunityDetector
from chinese_graphrag.indexing.community_report_generator import CommunityReportGenerator


class GraphRAGIndexer:
    """
    GraphRAG 索引引擎
    
    負責執行完整的 GraphRAG 索引流程，包括：
    1. 文件處理和分塊
    2. 實體和關係提取
    3. 社群檢測
    4. 向量化和儲存
    """

    def __init__(self, config: GraphRAGConfig):
        """
        初始化索引引擎
        
        Args:
            config: GraphRAG 配置
        """
        self.config = config
        self.model_selector = ModelSelector(config)
        
        # 初始化各個元件
        from chinese_graphrag.indexing.document_processor import DocumentProcessor
        self.document_processor = DocumentProcessor(config)
        self.embedding_manager = EmbeddingManager(config)
        self.vector_store_manager = VectorStoreManager(config)
        self.community_detector = CommunityDetector(
            min_community_size=config.indexing.min_community_size,
            max_community_size=config.indexing.max_community_size,
            enable_hierarchical=config.indexing.enable_hierarchical_communities
        )
        self.report_generator = CommunityReportGenerator(config)
        
        # 索引狀態
        self.indexed_documents: Dict[str, Document] = {}
        self.text_units: Dict[str, TextUnit] = {}
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.communities: Dict[str, Community] = {}
        self.community_reports: Dict[str, Dict[str, Any]] = {}

    async def index_documents(
        self, 
        input_path: Path,
        output_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        執行完整的文件索引流程
        
        Args:
            input_path: 輸入文件路徑
            output_path: 輸出路徑（可選）
            
        Returns:
            Dict: 索引結果統計
        """
        logger.info(f"開始索引文件: {input_path}")
        
        try:
            # 1. 文件處理和分塊
            logger.info("步驟 1: 處理文件和分塊")
            documents = await self._process_documents(input_path)
            text_units = await self._create_text_units(documents)
            
            # 2. 實體和關係提取
            logger.info("步驟 2: 提取實體和關係")
            entities, relationships = await self._extract_entities_and_relationships(text_units)
            
            # 3. 社群檢測
            logger.info("步驟 3: 社群檢測")
            communities = await self._detect_communities(entities, relationships)
            
            # 4. 向量化處理
            logger.info("步驟 4: 向量化處理")
            await self._create_embeddings(text_units, entities, communities)
            
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
                "communities": len(communities)
            }
            
            logger.info(f"索引完成: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"索引過程中發生錯誤: {e}")
            raise

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
                overlap=self.config.chunks.overlap
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
                        "language": "zh"  # 假設為中文
                    }
                )
                text_units.append(text_unit)
                self.text_units[text_unit.id] = text_unit
        
        logger.info(f"建立了 {len(text_units)} 個文本單元")
        return text_units

    async def _extract_entities_and_relationships(
        self, 
        text_units: List[TextUnit]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """提取實體和關係"""
        logger.info("提取實體和關係")
        
        if not self.config.indexing.enable_entity_extraction:
            logger.info("實體提取已停用")
            return [], []
        
        # 選擇 LLM 模型進行實體提取
        llm_name, llm_config = self.model_selector.select_llm_model(
            TaskType.ENTITY_EXTRACTION,
            context={"language": "zh"}
        )
        
        logger.info(f"使用 LLM 模型進行實體提取: {llm_name}")
        
        entities = []
        relationships = []
        
        # 批次處理文本單元
        batch_size = self.config.parallelization.batch_size
        
        for i in range(0, len(text_units), batch_size):
            batch = text_units[i:i + batch_size]
            
            # 並行處理批次
            batch_entities, batch_relationships = await self._process_text_unit_batch(
                batch, llm_config
            )
            
            entities.extend(batch_entities)
            relationships.extend(batch_relationships)
            
            # 記錄進度
            logger.info(f"已處理 {min(i + batch_size, len(text_units))}/{len(text_units)} 個文本單元")
        
        # 儲存結果
        for entity in entities:
            self.entities[entity.id] = entity
        
        for relationship in relationships:
            self.relationships[relationship.id] = relationship
        
        logger.info(f"提取了 {len(entities)} 個實體和 {len(relationships)} 個關係")
        return entities, relationships

    async def _process_text_unit_batch(
        self, 
        text_units: List[TextUnit],
        llm_config
    ) -> Tuple[List[Entity], List[Relationship]]:
        """處理文本單元批次"""
        # 這裡應該整合實際的 GraphRAG 實體提取邏輯
        # 目前提供一個簡化的實作
        
        entities = []
        relationships = []
        
        for text_unit in text_units:
            # 模擬實體提取（實際應該使用 LLM）
            extracted_entities = await self._extract_entities_from_text(
                text_unit.text, text_unit.id, llm_config
            )
            entities.extend(extracted_entities)
            
            # 模擬關係提取
            if self.config.indexing.enable_relationship_extraction:
                extracted_relationships = await self._extract_relationships_from_text(
                    text_unit.text, text_unit.id, extracted_entities, llm_config
                )
                relationships.extend(extracted_relationships)
        
        return entities, relationships

    async def _extract_entities_from_text(
        self, 
        text: str, 
        text_unit_id: str,
        llm_config
    ) -> List[Entity]:
        """從文本中提取實體"""
        # 簡化的實體提取邏輯
        # 實際實作應該使用 GraphRAG 的 LLM 提取流程
        
        entities = []
        
        # 模擬提取一些實體
        import re
        import uuid
        
        # 簡單的中文實體識別（實際應該使用 LLM）
        patterns = {
            "person": r'[\u4e00-\u9fff]{2,4}(?:先生|女士|教授|博士|主任|經理|總監)',
            "organization": r'[\u4e00-\u9fff]{2,10}(?:公司|企業|機構|組織|部門|學校|大學)',
            "location": r'[\u4e00-\u9fff]{2,8}(?:市|縣|區|省|國|地區|城市)',
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entity = Entity(
                    id=str(uuid.uuid4()),
                    name=match,
                    type=entity_type,
                    description=f"從文本中提取的{entity_type}實體",
                    text_units=[text_unit_id],
                    rank=1.0
                )
                entities.append(entity)
        
        return entities

    async def _extract_relationships_from_text(
        self, 
        text: str, 
        text_unit_id: str,
        entities: List[Entity],
        llm_config
    ) -> List[Relationship]:
        """從文本中提取關係"""
        # 簡化的關係提取邏輯
        relationships = []
        
        # 如果有多個實體，建立簡單的關係
        if len(entities) >= 2:
            import uuid
            
            for i in range(len(entities) - 1):
                relationship = Relationship(
                    id=str(uuid.uuid4()),
                    source_entity_id=entities[i].id,
                    target_entity_id=entities[i + 1].id,
                    relationship_type="related_to",
                    description="在同一文本單元中出現",
                    weight=0.5,
                    text_units=[text_unit_id]
                )
                relationships.append(relationship)
        
        return relationships

    async def _detect_communities(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship]
    ) -> List[Community]:
        """檢測社群"""
        logger.info("檢測社群結構")
        
        if not self.config.indexing.enable_community_detection:
            logger.info("社群檢測已停用")
            return []
        
        # 使用社群檢測器進行檢測
        communities = self.community_detector.detect_communities(entities, relationships)
        
        # 儲存社群資訊
        for community in communities:
            self.communities[community.id] = community
        
        # 生成社群報告
        if self.config.indexing.enable_community_reports:
            logger.info("生成社群報告")
            self.community_reports = await self.report_generator.generate_community_reports(
                communities,
                self.entities,
                self.relationships,
                self.text_units
            )
        
        logger.info(f"檢測到 {len(communities)} 個社群")
        return communities

    async def _create_embeddings(
        self, 
        text_units: List[TextUnit],
        entities: List[Entity],
        communities: List[Community]
    ):
        """建立向量嵌入"""
        logger.info("建立向量嵌入")
        
        # 選擇 Embedding 模型
        embedding_name, embedding_config = self.model_selector.select_embedding_model(
            TaskType.TEXT_EMBEDDING,
            context={"language": "zh"}
        )
        
        logger.info(f"使用 Embedding 模型: {embedding_name}")
        
        # 為文本單元建立嵌入
        if text_units:
            texts = [unit.text for unit in text_units]
            embeddings = await self.embedding_manager.embed_texts(texts, embedding_name)
            
            for unit, embedding in zip(text_units, embeddings):
                unit.embedding = embedding
                # 儲存到向量資料庫
                await self.vector_store_manager.store_text_unit(unit)
        
        # 為實體建立嵌入
        if entities:
            entity_texts = [f"{entity.name}: {entity.description}" for entity in entities]
            entity_embeddings = await self.embedding_manager.embed_texts(entity_texts, embedding_name)
            
            for entity, embedding in zip(entities, entity_embeddings):
                entity.embedding = embedding
                # 儲存到向量資料庫
                await self.vector_store_manager.store_entity(entity)
        
        # 為社群建立嵌入
        if communities:
            community_texts = [community.summary for community in communities]
            community_embeddings = await self.embedding_manager.embed_texts(community_texts, embedding_name)
            
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
                "metadata": doc.metadata
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
                "rank": entity.rank
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
                "weight": rel.weight,
                "text_units": rel.text_units
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
                "rank": comm.rank
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
            "communities_count": len(self.communities)
        }
        
        with open(output_path / "index_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("索引結果儲存完成")

    def get_statistics(self) -> Dict[str, int]:
        """取得索引統計資訊"""
        return {
            "documents": len(self.indexed_documents),
            "text_units": len(self.text_units),
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "communities": len(self.communities)
        }

    def clear_index(self):
        """清除索引資料"""
        logger.info("清除索引資料")
        
        self.indexed_documents.clear()
        self.text_units.clear()
        self.entities.clear()
        self.relationships.clear()
        self.communities.clear()