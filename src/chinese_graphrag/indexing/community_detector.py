"""
社群檢測器

使用圖算法檢測實體和關係中的社群結構
"""

import logging
import uuid
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from chinese_graphrag.models import Entity, Relationship, Community

logger = logging.getLogger(__name__)


class CommunityDetector:
    """社群檢測器"""
    
    def __init__(
        self,
        min_community_size: int = 3,
        max_community_size: int = 50,
        enable_hierarchical: bool = True
    ):
        """
        初始化社群檢測器
        
        Args:
            min_community_size: 最小社群大小
            max_community_size: 最大社群大小
            enable_hierarchical: 是否啟用層次化檢測
        """
        self.min_community_size = min_community_size
        self.max_community_size = max_community_size
        self.enable_hierarchical = enable_hierarchical
        
    def detect_communities(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship]
    ) -> List[Community]:
        """
        檢測社群
        
        Args:
            entities: 實體列表
            relationships: 關係列表
            
        Returns:
            List[Community]: 檢測到的社群列表
        """
        logger.info(f"開始社群檢測，實體數: {len(entities)}, 關係數: {len(relationships)}")
        
        # 建立圖結構
        graph = self._build_graph(entities, relationships)
        
        # 檢測社群
        communities = self._detect_communities_simple(graph, entities, relationships)
        
        # 過濾和優化社群
        communities = self._filter_communities(communities)
        
        # 生成社群摘要
        for community in communities:
            community.summary = self._generate_community_summary(community, entities, relationships)
        
        logger.info(f"社群檢測完成，檢測到 {len(communities)} 個社群")
        return communities
    
    def _build_graph(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship]
    ) -> Dict[str, Set[str]]:
        """
        建立圖結構
        
        Args:
            entities: 實體列表
            relationships: 關係列表
            
        Returns:
            Dict[str, Set[str]]: 鄰接表表示的圖
        """
        graph = defaultdict(set)
        
        # 初始化所有實體節點
        for entity in entities:
            graph[entity.id] = set()
        
        # 添加邊
        for relationship in relationships:
            source_id = relationship.source_entity_id
            target_id = relationship.target_entity_id
            
            if source_id in graph and target_id in graph:
                graph[source_id].add(target_id)
                graph[target_id].add(source_id)  # 無向圖
        
        return dict(graph)
    
    def _detect_communities_simple(
        self,
        graph: Dict[str, Set[str]],
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> List[Community]:
        """
        簡單的社群檢測算法（基於連通分量）
        
        Args:
            graph: 圖結構
            entities: 實體列表
            relationships: 關係列表
            
        Returns:
            List[Community]: 檢測到的社群
        """
        visited = set()
        communities = []
        
        for entity_id in graph:
            if entity_id not in visited:
                # 使用 DFS 找到連通分量
                component = self._dfs_component(graph, entity_id, visited)
                
                if len(component) >= self.min_community_size:
                    # 找到相關的關係
                    community_relationships = self._find_community_relationships(
                        component, relationships
                    )
                    
                    # 創建社群
                    community = Community(
                        id=str(uuid.uuid4()),
                        title=self._generate_community_title(component, entities),
                        level=1,
                        entities=list(component),
                        relationships=[rel.id for rel in community_relationships],
                        summary="",  # 稍後生成
                        rank=len(component) / len(entities)  # 基於大小的排名
                    )
                    
                    communities.append(community)
        
        return communities
    
    def _dfs_component(
        self,
        graph: Dict[str, Set[str]],
        start_node: str,
        visited: Set[str]
    ) -> Set[str]:
        """
        使用深度優先搜索找到連通分量
        
        Args:
            graph: 圖結構
            start_node: 起始節點
            visited: 已訪問節點集合
            
        Returns:
            Set[str]: 連通分量中的節點
        """
        component = set()
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.add(node)
                
                # 添加鄰居節點
                for neighbor in graph.get(node, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return component
    
    def _find_community_relationships(
        self,
        entity_ids: Set[str],
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """
        找到社群內的關係
        
        Args:
            entity_ids: 社群中的實體 ID 集合
            relationships: 所有關係列表
            
        Returns:
            List[Relationship]: 社群內的關係
        """
        community_relationships = []
        
        for relationship in relationships:
            if (relationship.source_entity_id in entity_ids and 
                relationship.target_entity_id in entity_ids):
                community_relationships.append(relationship)
        
        return community_relationships
    
    def _generate_community_title(
        self,
        entity_ids: Set[str],
        entities: List[Entity]
    ) -> str:
        """
        生成社群標題
        
        Args:
            entity_ids: 社群中的實體 ID 集合
            entities: 所有實體列表
            
        Returns:
            str: 社群標題
        """
        # 創建實體 ID 到實體的映射
        entity_map = {entity.id: entity for entity in entities}
        
        # 統計實體類型
        type_counts = defaultdict(int)
        for entity_id in entity_ids:
            if entity_id in entity_map:
                entity_type = entity_map[entity_id].type
                type_counts[entity_type] += 1
        
        # 找到最常見的類型
        if type_counts:
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            return f"{most_common_type}社群"
        else:
            return "未知社群"
    
    def _generate_community_summary(
        self,
        community: Community,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> str:
        """
        生成社群摘要
        
        Args:
            community: 社群對象
            entities: 所有實體列表
            relationships: 所有關係列表
            
        Returns:
            str: 社群摘要
        """
        # 創建映射
        entity_map = {entity.id: entity for entity in entities}
        relationship_map = {rel.id: rel for rel in relationships}
        
        # 收集社群中的實體名稱
        entity_names = []
        for entity_id in community.entities:
            if entity_id in entity_map:
                entity_names.append(entity_map[entity_id].name)
        
        # 收集關係描述
        relationship_descriptions = []
        for rel_id in community.relationships:
            if rel_id in relationship_map:
                relationship_descriptions.append(relationship_map[rel_id].description)
        
        # 生成摘要
        summary_parts = []
        
        if entity_names:
            summary_parts.append(f"這個社群包含 {len(entity_names)} 個實體：{', '.join(entity_names[:5])}")
            if len(entity_names) > 5:
                summary_parts[-1] += f" 等 {len(entity_names)} 個實體"
        
        if relationship_descriptions:
            summary_parts.append(f"主要關係包括：{'; '.join(relationship_descriptions[:3])}")
            if len(relationship_descriptions) > 3:
                summary_parts[-1] += f" 等 {len(relationship_descriptions)} 個關係"
        
        return "。".join(summary_parts) + "。"
    
    def _filter_communities(self, communities: List[Community]) -> List[Community]:
        """
        過濾和優化社群
        
        Args:
            communities: 原始社群列表
            
        Returns:
            List[Community]: 過濾後的社群列表
        """
        filtered_communities = []
        
        for community in communities:
            # 檢查大小限制
            if (self.min_community_size <= len(community.entities) <= self.max_community_size):
                filtered_communities.append(community)
            elif len(community.entities) > self.max_community_size:
                # 如果社群太大，可以考慮分割（這裡簡化處理）
                logger.warning(f"社群 {community.id} 太大 ({len(community.entities)} 個實體)，已跳過")
        
        # 按排名排序
        filtered_communities.sort(key=lambda x: x.rank, reverse=True)
        
        return filtered_communities