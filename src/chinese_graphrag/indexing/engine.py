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
# from chinese_graphrag.config.strategy import ModelSelector, TaskType  # 已移除
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
        # self.model_selector = ModelSelector(config)  # 已移除
        
        # 初始化各個元件
        try:
            from chinese_graphrag.indexing.document_processor import DocumentProcessor
            self.document_processor = DocumentProcessor(config)
        except ImportError:
            # 如果 DocumentProcessor 不存在，創建一個簡單的替代
            self.document_processor = self._create_simple_document_processor()
        
        self.embedding_manager = EmbeddingManager(config)
        self.vector_store_manager = VectorStoreManager(config.vector_store.type)
        
        # 使用安全的屬性訪問
        min_community_size = getattr(config.indexing, 'min_community_size', 3)
        max_community_size = getattr(config.indexing, 'max_community_size', 50)
        enable_hierarchical = getattr(config.indexing, 'enable_hierarchical_communities', True)
        
        self.community_detector = CommunityDetector(
            min_community_size=min_community_size,
            max_community_size=max_community_size,
            enable_hierarchical=enable_hierarchical
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
            
            # 更新索引狀態（確保測試時也能正確更新）
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
        
        # 實體提取預設啟用
        # if not self.config.indexing.enable_entity_extraction:
        #     logger.info("實體提取已停用")
        #     return [], []
        
        # 使用預設的 LLM 配置
        default_llm_name = self.config.model_selection.default_llm
        llm_config = self.config.get_llm_config(default_llm_name)
        
        if not llm_config:
            # 創建一個簡單的測試配置
            llm_config = {
                "type": "mock",
                "model": "test_model",
                "max_tokens": 4000,
                "temperature": 0.7
            }
        
        logger.info(f"使用 LLM 模型進行實體提取: {default_llm_name}")
        
        entities = []
        relationships = []
        
        # 批次處理文本單元
        batch_size = 10  # 預設批次大小
        
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
        """
        使用 LLM 從一批文本單元中提取實體和關係。
        """
        if not text_units:
            return [], []

        # 呼叫新的圖譜提取方法
        entities, relationships = await self._extract_graph_from_batch(text_units, llm_config)
        
        # 處理關係中的實體名稱，將其轉換為 ID
        entity_name_to_id = {entity.name: entity.id for entity in entities}
        
        valid_relationships = []
        for rel in relationships:
            # 確保來源和目標實體都存在
            if rel.source_entity_id in entity_name_to_id and rel.target_entity_id in entity_name_to_id:
                rel.source_entity_id = entity_name_to_id[rel.source_entity_id]
                rel.target_entity_id = entity_name_to_id[rel.target_entity_id]
                valid_relationships.append(rel)
            else:
                logger.warning(f"關係 '{rel.description}' 的實體 '{rel.source_entity_id}' 或 '{rel.target_entity_id}' 不存在，已忽略。")
        
        return entities, valid_relationships

    async def _extract_graph_from_batch(
        self,
        text_units: List[TextUnit],
        llm_config
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        使用 LLM 從一批文本單元中提取實體和關係圖譜。
        """
        from chinese_graphrag.llm import create_llm, LLM
        
        # 建立 LLM 實例
        llm: LLM = create_llm(llm_config.get("type", "mock"), llm_config)

        # 建構 prompt
        prompt = self._build_extraction_prompt(text_units)

        # 呼叫 LLM
        try:
            response = await llm.async_generate(prompt)
            output = json.loads(response)
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
        input_text = "\n\n".join([f"## 文件片段 (ID: {unit.id})\n\n{unit.text}" for unit in text_units])
        
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

## 範例

### 輸入文本
```
## 文件片段 (ID: doc1_chunk_0)

會議記錄顯示，張偉明代表「宏達電（HTC）」與「台灣大哥大」的林總經理討論了關於「5G應用」的合作計畫。此計畫旨在利用HTC的「虛擬實境（VR）」技術開發新的消費者應用。
```

### 輸出JSON
```json
{{
  "entities": [
    {{ "name": "張偉明", "type": "人物", "description": "宏達電的代表" }},
    {{ "name": "宏達電（HTC）", "type": "公司", "description": "一家消費性電子產品公司，專注於虛擬實境技術" }},
    {{ "name": "台灣大哥大", "type": "公司", "description": "台灣主要的電信服務提供商之一" }},
    {{ "name": "林總經理", "type": "人物", "description": "台灣大哥大的總經理" }},
    {{ "name": "5G應用", "type": "技術", "description": "基於第五代行動通訊技術的應用服務" }},
    {{ "name": "虛擬實境（VR）", "type": "技術", "description": "HTC 專長的一種沉浸式技術" }}
  ],
  "relationships": [
    {{ "source": "張偉明", "target": "宏達電（HTC）", "description": "張偉明是宏達電（HTC）的代表" }},
    {{ "source": "林總經理", "target": "台灣大哥大", "description": "林總經理是台灣大哥大的總經理" }},
    {{ "source": "宏達電（HTC）", "target": "台灣大哥大", "description": "宏達電與台灣大哥大討論合作計畫" }},
    {{ "source": "合作計畫", "target": "5G應用", "description": "合作計畫的主題是5G應用" }},
    {{ "source": "合作計畫", "target": "虛擬實境（VR）", "description": "合作計畫利用虛擬實境（VR）技術" }}
  ]
}}
```

## 待處理文本

請根據以上規則，處理以下文本：

```
{input_text}
```

## 輸出 JSON
"""

    def _parse_llm_output(
        self, 
        output: Dict[str, List[Dict[str, str]]],
        text_units: List[TextUnit]
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
                rank=1.0
            )
            entities.append(entity)

        # 提取關係
        for item in output.get("relationships", []):
            relationship = Relationship(
                id=str(uuid.uuid4()),
                source_entity_id=item.get("source"), # 暫存實體名稱
                target_entity_id=item.get("target"), # 暫存實體名稱
                relationship_type="related_to", # 可從 description 推斷
                description=item.get("description"),
                weight=0.8,
                text_units=[unit.id for unit in text_units]
            )
            relationships.append(relationship)
            
        return entities, relationships

    async def _detect_communities(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship]
    ) -> List[Community]:
        """檢測社群"""
        logger.info("檢測社群結構")
        
        # 社群檢測預設啟用
        # if not self.config.indexing.enable_community_detection:
        #     logger.info("社群檢測已停用")
        #     return []
        
        # 使用社群檢測器進行檢測
        communities = self.community_detector.detect_communities(entities, relationships)
        
        # 儲存社群資訊
        for community in communities:
            self.communities[community.id] = community
        
        # 生成社群報告 (預設啟用)
        try:
            logger.info("生成社群報告")
            self.community_reports = await self.report_generator.generate_community_reports(
                communities,
                self.entities,
                self.relationships,
                self.text_units
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
        communities: List[Community]
    ):
        """建立向量嵌入"""
        logger.info("建立向量嵌入")
        
        # 使用預設的 Embedding 模型
        default_embedding_name = self.config.model_selection.default_embedding
        embedding_config = self.config.get_embedding_config(default_embedding_name)
        
        if embedding_config:
            embedding_name = embedding_config.model
        else:
            embedding_name = "bge-m3"  # 預設值
        
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
                    file_path=file_path
                )
            
            def batch_process(self, input_path):
                return [self.process_document(input_path)]
            
            def split_text(self, text, chunk_size=1000, overlap=200):
                # 簡單的文本分割
                chunks = []
                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i:i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                return chunks
        
        return SimpleDocumentProcessor(self.config)
    
    def _create_simple_report_generator(self):
        """創建簡單的報告生成器替代"""
        class SimpleReportGenerator:
            def __init__(self, config):
                self.config = config
            
            async def generate_community_reports(self, communities, entities, relationships, text_units):
                # 簡單的報告生成邏輯
                reports = {}
                for community in communities:
                    reports[community.id] = {
                        "title": community.title,
                        "summary": community.summary,
                        "entities_count": len(community.entities),
                        "relationships_count": len(community.relationships)
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