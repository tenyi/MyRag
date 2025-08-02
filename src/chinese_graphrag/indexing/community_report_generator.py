"""
社群報告生成器

為檢測到的社群生成詳細的報告和摘要
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from chinese_graphrag.models import Community, Entity, Relationship, TextUnit
from chinese_graphrag.config import GraphRAGConfig

logger = logging.getLogger(__name__)


class CommunityReportGenerator:
    """社群報告生成器"""
    
    def __init__(self, config: GraphRAGConfig):
        """
        初始化社群報告生成器
        
        Args:
            config: GraphRAG 配置
        """
        self.config = config
        
    async def generate_community_reports(
        self,
        communities: List[Community],
        entities: Dict[str, Entity],
        relationships: Dict[str, Relationship],
        text_units: Dict[str, TextUnit]
    ) -> Dict[str, Dict[str, Any]]:
        """
        為所有社群生成報告
        
        Args:
            communities: 社群列表
            entities: 實體字典
            relationships: 關係字典
            text_units: 文本單元字典
            
        Returns:
            Dict[str, Dict[str, Any]]: 社群報告字典
        """
        logger.info(f"開始生成 {len(communities)} 個社群的報告")
        
        reports = {}
        
        for community in communities:
            try:
                report = await self._generate_single_community_report(
                    community, entities, relationships, text_units
                )
                reports[community.id] = report
                
            except Exception as e:
                logger.error(f"生成社群 {community.id} 報告失敗: {e}")
                # 生成基本報告
                reports[community.id] = self._generate_basic_report(community)
        
        logger.info(f"社群報告生成完成，共生成 {len(reports)} 個報告")
        return reports
    
    async def _generate_single_community_report(
        self,
        community: Community,
        entities: Dict[str, Entity],
        relationships: Dict[str, Relationship],
        text_units: Dict[str, TextUnit]
    ) -> Dict[str, Any]:
        """
        生成單個社群的詳細報告
        
        Args:
            community: 社群對象
            entities: 實體字典
            relationships: 關係字典
            text_units: 文本單元字典
            
        Returns:
            Dict[str, Any]: 社群報告
        """
        # 收集社群中的實體
        community_entities = []
        for entity_id in community.entities:
            if entity_id in entities:
                community_entities.append(entities[entity_id])
        
        # 收集社群中的關係
        community_relationships = []
        for rel_id in community.relationships:
            if rel_id in relationships:
                community_relationships.append(relationships[rel_id])
        
        # 收集相關的文本單元
        related_text_units = self._collect_related_text_units(
            community_entities, text_units
        )
        
        # 分析實體類型分佈
        entity_type_distribution = self._analyze_entity_types(community_entities)
        
        # 分析關係類型分佈
        relationship_type_distribution = self._analyze_relationship_types(community_relationships)
        
        # 生成關鍵詞
        keywords = self._extract_community_keywords(
            community_entities, community_relationships, related_text_units
        )
        
        # 計算重要性分數
        importance_score = self._calculate_importance_score(
            community, community_entities, community_relationships
        )
        
        # 生成詳細摘要
        detailed_summary = await self._generate_detailed_summary(
            community, community_entities, community_relationships, related_text_units
        )
        
        # 構建報告
        report = {
            "community_id": community.id,
            "title": community.title,
            "level": community.level,
            "summary": community.summary,
            "detailed_summary": detailed_summary,
            "entities_count": len(community_entities),
            "relationships_count": len(community_relationships),
            "entity_type_distribution": entity_type_distribution,
            "relationship_type_distribution": relationship_type_distribution,
            "keywords": keywords,
            "importance_score": importance_score,
            "rank": community.rank,
            "generated_at": datetime.now().isoformat(),
            "entities": [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description
                }
                for entity in community_entities
            ],
            "relationships": [
                {
                    "id": rel.id,
                    "source_entity_id": rel.source_entity_id,
                    "target_entity_id": rel.target_entity_id,
                    "description": rel.description,
                    "relationship_type": getattr(rel, 'relationship_type', 'unknown')
                }
                for rel in community_relationships
            ],
            "related_text_units": len(related_text_units)
        }
        
        return report
    
    def _generate_basic_report(self, community: Community) -> Dict[str, Any]:
        """
        生成基本報告（當詳細報告生成失敗時使用）
        
        Args:
            community: 社群對象
            
        Returns:
            Dict[str, Any]: 基本報告
        """
        return {
            "community_id": community.id,
            "title": community.title,
            "level": community.level,
            "summary": community.summary,
            "entities_count": len(community.entities),
            "relationships_count": len(community.relationships),
            "rank": community.rank,
            "generated_at": datetime.now().isoformat(),
            "status": "basic_report_only"
        }
    
    def _collect_related_text_units(
        self,
        entities: List[Entity],
        text_units: Dict[str, TextUnit]
    ) -> List[TextUnit]:
        """
        收集與實體相關的文本單元
        
        Args:
            entities: 實體列表
            text_units: 文本單元字典
            
        Returns:
            List[TextUnit]: 相關的文本單元
        """
        related_units = []
        collected_ids = set()
        
        for entity in entities:
            for unit_id in getattr(entity, 'text_units', []):
                if unit_id in text_units and unit_id not in collected_ids:
                    related_units.append(text_units[unit_id])
                    collected_ids.add(unit_id)
        
        return related_units
    
    def _analyze_entity_types(self, entities: List[Entity]) -> Dict[str, int]:
        """
        分析實體類型分佈
        
        Args:
            entities: 實體列表
            
        Returns:
            Dict[str, int]: 類型分佈統計
        """
        type_counts = {}
        for entity in entities:
            entity_type = entity.type or "未知"
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        return type_counts
    
    def _analyze_relationship_types(self, relationships: List[Relationship]) -> Dict[str, int]:
        """
        分析關係類型分佈
        
        Args:
            relationships: 關係列表
            
        Returns:
            Dict[str, int]: 關係類型分佈統計
        """
        type_counts = {}
        for relationship in relationships:
            rel_type = getattr(relationship, 'relationship_type', '未知')
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        
        return type_counts
    
    def _extract_community_keywords(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        text_units: List[TextUnit]
    ) -> List[str]:
        """
        提取社群關鍵詞
        
        Args:
            entities: 實體列表
            relationships: 關係列表
            text_units: 文本單元列表
            
        Returns:
            List[str]: 關鍵詞列表
        """
        keywords = set()
        
        # 從實體名稱中提取關鍵詞
        for entity in entities:
            if entity.name:
                keywords.add(entity.name)
        
        # 從關係描述中提取關鍵詞
        for relationship in relationships:
            if relationship.description:
                # 簡單的關鍵詞提取（可以改進）
                words = relationship.description.split()
                for word in words:
                    if len(word) > 2:  # 過濾短詞
                        keywords.add(word)
        
        # 限制關鍵詞數量
        return list(keywords)[:20]
    
    def _calculate_importance_score(
        self,
        community: Community,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> float:
        """
        計算社群重要性分數
        
        Args:
            community: 社群對象
            entities: 實體列表
            relationships: 關係列表
            
        Returns:
            float: 重要性分數 (0-1)
        """
        # 基於多個因素計算重要性分數
        
        # 1. 實體數量權重
        entity_score = min(len(entities) / 20, 1.0)  # 最多20個實體得滿分
        
        # 2. 關係數量權重
        relationship_score = min(len(relationships) / 30, 1.0)  # 最多30個關係得滿分
        
        # 3. 實體類型多樣性權重
        entity_types = set(entity.type for entity in entities if entity.type)
        diversity_score = min(len(entity_types) / 5, 1.0)  # 最多5種類型得滿分
        
        # 4. 社群排名權重
        rank_score = community.rank
        
        # 綜合計算
        importance_score = (
            entity_score * 0.3 +
            relationship_score * 0.3 +
            diversity_score * 0.2 +
            rank_score * 0.2
        )
        
        return round(importance_score, 3)
    
    async def _generate_detailed_summary(
        self,
        community: Community,
        entities: List[Entity],
        relationships: List[Relationship],
        text_units: List[TextUnit]
    ) -> str:
        """
        生成詳細摘要
        
        Args:
            community: 社群對象
            entities: 實體列表
            relationships: 關係列表
            text_units: 文本單元列表
            
        Returns:
            str: 詳細摘要
        """
        # 這裡可以使用 LLM 生成更詳細的摘要
        # 目前使用基於規則的方法
        
        summary_parts = []
        
        # 基本資訊
        summary_parts.append(f"這是一個名為「{community.title}」的社群")
        summary_parts.append(f"包含 {len(entities)} 個實體和 {len(relationships)} 個關係")
        
        # 主要實體
        if entities:
            main_entities = sorted(entities, key=lambda x: getattr(x, 'rank', 0), reverse=True)[:5]
            entity_names = [entity.name for entity in main_entities]
            summary_parts.append(f"主要實體包括：{', '.join(entity_names)}")
        
        # 實體類型分佈
        entity_types = {}
        for entity in entities:
            entity_type = entity.type or "未知"
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        if entity_types:
            type_desc = []
            for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                type_desc.append(f"{count}個{entity_type}")
            summary_parts.append(f"實體類型分佈：{', '.join(type_desc)}")
        
        # 主要關係
        if relationships:
            rel_descriptions = [rel.description for rel in relationships[:3] if rel.description]
            if rel_descriptions:
                summary_parts.append(f"主要關係：{'; '.join(rel_descriptions)}")
        
        return "。".join(summary_parts) + "。"