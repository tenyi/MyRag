"""
社群檢測器

實作知識圖譜中的社群檢測和層次結構建立
"""

import uuid
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import numpy as np
from loguru import logger

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    NETWORKX_AVAILABLE = True
except ImportError:
    logger.warning("NetworkX 未安裝，將使用簡化的社群檢測演算法")
    NETWORKX_AVAILABLE = False

from chinese_graphrag.models import Community, Entity, Relationship


class CommunityDetector:
    """
    社群檢測器
    
    使用圖形演算法檢測實體和關係中的社群結構，
    並建立層次化的社群組織
    """
    
    def __init__(
        self,
        min_community_size: int = 3,
        max_community_size: int = 50,
        resolution: float = 1.0,
        enable_hierarchical: bool = True
    ):
        """
        初始化社群檢測器
        
        Args:
            min_community_size: 最小社群大小
            max_community_size: 最大社群大小
            resolution: 社群檢測解析度（越高越多小社群）
            enable_hierarchical: 是否啟用層次化社群檢測
        """
        self.min_community_size = min_community_size
        self.max_community_size = max_community_size
        self.resolution = resolution
        self.enable_hierarchical = enable_hierarchical
        
        logger.info(f"初始化社群檢測器: min_size={min_community_size}, max_size={max_community_size}")
    
    def detect_communities(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> List[Community]:
        """
        檢測社群結構
        
        Args:
            entities: 實體列表
            relationships: 關係列表
            
        Returns:
            List[Community]: 檢測到的社群列表
        """
        logger.info(f"開始社群檢測: {len(entities)} 個實體, {len(relationships)} 個關係")
        
        if not entities:
            logger.warning("沒有實體資料，無法進行社群檢測")
            return []
        
        # 建立圖形結構
        graph = self._build_graph(entities, relationships)
        
        if NETWORKX_AVAILABLE:
            # 使用 NetworkX 進行社群檢測
            communities = self._detect_communities_networkx(graph, entities, relationships)
        else:
            # 使用簡化的社群檢測演算法
            communities = self._detect_communities_simple(entities, relationships)
        
        # 建立層次結構
        if self.enable_hierarchical and len(communities) > 1:
            communities = self._build_hierarchical_communities(communities, entities, relationships)
        
        logger.info(f"檢測到 {len(communities)} 個社群")
        return communities
    
    def _build_graph(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> Optional[Any]:
        """建立圖形結構"""
        if not NETWORKX_AVAILABLE:
            return None
        
        # 建立無向圖
        G = nx.Graph()
        
        # 添加節點（實體）
        for entity in entities:
            G.add_node(
                entity.id,
                name=entity.name,
                type=entity.type,
                rank=entity.rank,
                entity=entity
            )
        
        # 添加邊（關係）
        for relationship in relationships:
            if (relationship.source_entity_id in [e.id for e in entities] and
                relationship.target_entity_id in [e.id for e in entities]):
                G.add_edge(
                    relationship.source_entity_id,
                    relationship.target_entity_id,
                    weight=relationship.weight,
                    type=relationship.relationship_type,
                    relationship=relationship
                )
        
        logger.info(f"建立圖形: {G.number_of_nodes()} 個節點, {G.number_of_edges()} 條邊")
        return G
    
    def _detect_communities_networkx(
        self,
        graph: Any,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> List[Community]:
        """使用 NetworkX 進行社群檢測"""
        communities = []
        
        try:
            # 使用 Louvain 演算法進行社群檢測
            community_partition = nx_community.louvain_communities(
                graph,
                resolution=self.resolution,
                seed=42
            )
            
            logger.info(f"Louvain 演算法檢測到 {len(community_partition)} 個社群")
            
            # 轉換為 Community 物件
            for i, community_nodes in enumerate(community_partition):
                if len(community_nodes) < self.min_community_size:
                    continue
                
                if len(community_nodes) > self.max_community_size:
                    # 對大社群進行進一步分割
                    sub_communities = self._split_large_community(
                        graph, community_nodes, entities, relationships
                    )
                    communities.extend(sub_communities)
                else:
                    community = self._create_community_from_nodes(
                        community_nodes, entities, relationships, level=0
                    )
                    if community:
                        communities.append(community)
            
        except Exception as e:
            logger.error(f"NetworkX 社群檢測失敗: {e}")
            # 回退到簡化演算法
            communities = self._detect_communities_simple(entities, relationships)
        
        return communities
    
    def _detect_communities_simple(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> List[Community]:
        """簡化的社群檢測演算法"""
        logger.info("使用簡化的社群檢測演算法")
        
        communities = []
        
        # 按實體類型分組作為基本社群
        entity_groups = defaultdict(list)
        for entity in entities:
            entity_groups[entity.type].append(entity)
        
        # 為每個類型建立社群
        for entity_type, group_entities in entity_groups.items():
            if len(group_entities) >= self.min_community_size:
                # 找到相關的關係
                entity_ids = {e.id for e in group_entities}
                related_relationships = [
                    rel for rel in relationships
                    if rel.source_entity_id in entity_ids or rel.target_entity_id in entity_ids
                ]
                
                community = Community(
                    id=str(uuid.uuid4()),
                    title=f"{entity_type}社群",
                    level=0,
                    entities=[e.id for e in group_entities],
                    relationships=[r.id for r in related_relationships],
                    summary=f"包含 {len(group_entities)} 個{entity_type}實體的社群",
                    full_content=self._generate_community_content(group_entities, related_relationships),
                    rank=self._calculate_community_rank(group_entities, related_relationships)
                )
                communities.append(community)
        
        # 處理混合類型的社群（基於關係密度）
        mixed_communities = self._detect_mixed_type_communities(entities, relationships)
        communities.extend(mixed_communities)
        
        return communities
    
    def _split_large_community(
        self,
        graph: Any,
        community_nodes: Set[str],
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> List[Community]:
        """分割大型社群"""
        logger.info(f"分割大型社群: {len(community_nodes)} 個節點")
        
        sub_communities = []
        
        try:
            # 建立子圖
            subgraph = graph.subgraph(community_nodes)
            
            # 使用更高的解析度進行二次檢測
            sub_partitions = nx_community.louvain_communities(
                subgraph,
                resolution=self.resolution * 2.0,
                seed=42
            )
            
            for sub_nodes in sub_partitions:
                if len(sub_nodes) >= self.min_community_size:
                    community = self._create_community_from_nodes(
                        sub_nodes, entities, relationships, level=0
                    )
                    if community:
                        sub_communities.append(community)
            
        except Exception as e:
            logger.error(f"分割大型社群失敗: {e}")
            # 回退到簡單分割
            node_list = list(community_nodes)
            chunk_size = self.max_community_size // 2
            
            for i in range(0, len(node_list), chunk_size):
                chunk_nodes = set(node_list[i:i + chunk_size])
                if len(chunk_nodes) >= self.min_community_size:
                    community = self._create_community_from_nodes(
                        chunk_nodes, entities, relationships, level=0
                    )
                    if community:
                        sub_communities.append(community)
        
        logger.info(f"分割結果: {len(sub_communities)} 個子社群")
        return sub_communities
    
    def _create_community_from_nodes(
        self,
        nodes: Set[str],
        entities: List[Entity],
        relationships: List[Relationship],
        level: int = 0
    ) -> Optional[Community]:
        """從節點集合建立社群"""
        # 找到對應的實體
        community_entities = [e for e in entities if e.id in nodes]
        if not community_entities:
            return None
        
        # 找到相關的關係
        community_relationships = [
            rel for rel in relationships
            if rel.source_entity_id in nodes and rel.target_entity_id in nodes
        ]
        
        # 生成社群標題
        entity_types = list(set(e.type for e in community_entities))
        if len(entity_types) == 1:
            title = f"{entity_types[0]}社群"
        else:
            title = f"混合社群({', '.join(entity_types[:3])})"
        
        # 生成摘要
        summary = self._generate_community_summary(community_entities, community_relationships)
        
        # 生成完整內容
        full_content = self._generate_community_content(community_entities, community_relationships)
        
        # 計算排名
        rank = self._calculate_community_rank(community_entities, community_relationships)
        
        community = Community(
            id=str(uuid.uuid4()),
            title=title,
            level=level,
            entities=[e.id for e in community_entities],
            relationships=[r.id for r in community_relationships],
            summary=summary,
            full_content=full_content,
            rank=rank
        )
        
        return community
    
    def _detect_mixed_type_communities(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> List[Community]:
        """檢測混合類型社群"""
        mixed_communities = []
        
        # 建立實體連接圖
        entity_connections = defaultdict(set)
        for rel in relationships:
            entity_connections[rel.source_entity_id].add(rel.target_entity_id)
            entity_connections[rel.target_entity_id].add(rel.source_entity_id)
        
        # 使用連通分量檢測密集連接的實體群組
        visited = set()
        entity_dict = {e.id: e for e in entities}
        
        for entity in entities:
            if entity.id in visited:
                continue
            
            # 深度優先搜尋找到連通分量
            component = self._dfs_component(
                entity.id, entity_connections, visited, min_size=self.min_community_size
            )
            
            if len(component) >= self.min_community_size:
                component_entities = [entity_dict[eid] for eid in component if eid in entity_dict]
                
                # 檢查是否為混合類型
                entity_types = set(e.type for e in component_entities)
                if len(entity_types) > 1:
                    # 找到相關關係
                    component_relationships = [
                        rel for rel in relationships
                        if rel.source_entity_id in component and rel.target_entity_id in component
                    ]
                    
                    community = Community(
                        id=str(uuid.uuid4()),
                        title=f"混合社群({', '.join(list(entity_types)[:3])})",
                        level=0,
                        entities=list(component),
                        relationships=[r.id for r in component_relationships],
                        summary=self._generate_community_summary(component_entities, component_relationships),
                        full_content=self._generate_community_content(component_entities, component_relationships),
                        rank=self._calculate_community_rank(component_entities, component_relationships)
                    )
                    mixed_communities.append(community)
        
        return mixed_communities
    
    def _dfs_component(
        self,
        start_node: str,
        connections: Dict[str, Set[str]],
        visited: Set[str],
        min_size: int
    ) -> Set[str]:
        """深度優先搜尋連通分量"""
        component = set()
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            
            visited.add(node)
            component.add(node)
            
            # 添加鄰居節點
            for neighbor in connections.get(node, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return component if len(component) >= min_size else set()
    
    def _build_hierarchical_communities(
        self,
        base_communities: List[Community],
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> List[Community]:
        """建立層次化社群結構"""
        logger.info(f"建立層次化社群結構: {len(base_communities)} 個基礎社群")
        
        if len(base_communities) <= 2:
            return base_communities
        
        all_communities = base_communities.copy()
        
        # 計算社群間的相似度
        community_similarities = self._calculate_community_similarities(base_communities, entities)
        
        # 使用層次聚類建立上層社群
        higher_level_communities = self._hierarchical_clustering(
            base_communities, community_similarities, entities, relationships
        )
        
        # 設定父子關係
        for higher_community in higher_level_communities:
            for child_id in higher_community.child_communities:
                for base_community in base_communities:
                    if base_community.id == child_id:
                        base_community.parent_community_id = higher_community.id
                        break
        
        all_communities.extend(higher_level_communities)
        
        logger.info(f"層次化結果: {len(all_communities)} 個社群（包含 {len(higher_level_communities)} 個上層社群）")
        return all_communities
    
    def _calculate_community_similarities(
        self,
        communities: List[Community],
        entities: List[Entity]
    ) -> Dict[Tuple[str, str], float]:
        """計算社群間的相似度"""
        similarities = {}
        entity_dict = {e.id: e for e in entities}
        
        for i, comm1 in enumerate(communities):
            for j, comm2 in enumerate(communities[i+1:], i+1):
                # 計算實體類型重疊度
                entities1 = [entity_dict[eid] for eid in comm1.entities if eid in entity_dict]
                entities2 = [entity_dict[eid] for eid in comm2.entities if eid in entity_dict]
                
                types1 = set(e.type for e in entities1)
                types2 = set(e.type for e in entities2)
                
                # Jaccard 相似度
                intersection = len(types1 & types2)
                union = len(types1 | types2)
                
                if union > 0:
                    similarity = intersection / union
                else:
                    similarity = 0.0
                
                similarities[(comm1.id, comm2.id)] = similarity
        
        return similarities
    
    def _hierarchical_clustering(
        self,
        communities: List[Community],
        similarities: Dict[Tuple[str, str], float],
        entities: List[Entity],
        relationships: List[Relationship],
        threshold: float = 0.3
    ) -> List[Community]:
        """層次聚類建立上層社群"""
        higher_communities = []
        
        # 找到相似度高於閾值的社群對
        similar_pairs = [
            (comm1_id, comm2_id) for (comm1_id, comm2_id), sim in similarities.items()
            if sim >= threshold
        ]
        
        if not similar_pairs:
            return higher_communities
        
        # 建立社群群組
        community_groups = []
        used_communities = set()
        
        for comm1_id, comm2_id in similar_pairs:
            if comm1_id in used_communities or comm2_id in used_communities:
                continue
            
            # 找到所有相關的社群
            group = {comm1_id, comm2_id}
            used_communities.update(group)
            
            # 擴展群組
            expanded = True
            while expanded:
                expanded = False
                for other_comm1, other_comm2 in similar_pairs:
                    if other_comm1 in group and other_comm2 not in used_communities:
                        group.add(other_comm2)
                        used_communities.add(other_comm2)
                        expanded = True
                    elif other_comm2 in group and other_comm1 not in used_communities:
                        group.add(other_comm1)
                        used_communities.add(other_comm1)
                        expanded = True
            
            if len(group) >= 2:
                community_groups.append(group)
        
        # 為每個群組建立上層社群
        community_dict = {c.id: c for c in communities}
        
        for group in community_groups:
            child_communities = [community_dict[cid] for cid in group if cid in community_dict]
            
            if len(child_communities) >= 2:
                higher_community = self._create_higher_level_community(
                    child_communities, entities, relationships
                )
                if higher_community:
                    higher_communities.append(higher_community)
        
        return higher_communities
    
    def _create_higher_level_community(
        self,
        child_communities: List[Community],
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> Optional[Community]:
        """建立上層社群"""
        if not child_communities:
            return None
        
        # 合併所有子社群的實體和關係
        all_entities = set()
        all_relationships = set()
        
        for child in child_communities:
            all_entities.update(child.entities)
            all_relationships.update(child.relationships)
        
        # 生成標題
        child_titles = [child.title for child in child_communities]
        title = f"上層社群({len(child_communities)}個子社群)"
        
        # 生成摘要
        summary = f"包含 {len(child_communities)} 個子社群的上層社群，共有 {len(all_entities)} 個實體和 {len(all_relationships)} 個關係"
        
        # 生成完整內容
        full_content = f"上層社群包含以下子社群：\n"
        for child in child_communities:
            full_content += f"- {child.title}: {child.summary}\n"
        
        # 計算排名（基於子社群的平均排名）
        avg_rank = sum(child.rank for child in child_communities) / len(child_communities)
        
        higher_community = Community(
            id=str(uuid.uuid4()),
            title=title,
            level=1,  # 上層社群
            entities=list(all_entities),
            relationships=list(all_relationships),
            summary=summary,
            full_content=full_content,
            rank=avg_rank,
            child_communities=[child.id for child in child_communities]
        )
        
        return higher_community
    
    def _generate_community_summary(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> str:
        """生成社群摘要"""
        if not entities:
            return "空社群"
        
        # 統計實體類型
        entity_types = {}
        for entity in entities:
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        # 統計關係類型
        relationship_types = {}
        for rel in relationships:
            relationship_types[rel.relationship_type] = relationship_types.get(rel.relationship_type, 0) + 1
        
        # 生成摘要文本
        summary_parts = []
        
        # 實體摘要
        if entity_types:
            entity_summary = ", ".join([f"{count}個{etype}" for etype, count in entity_types.items()])
            summary_parts.append(f"包含{entity_summary}")
        
        # 關係摘要
        if relationship_types:
            rel_summary = ", ".join([f"{count}個{rtype}關係" for rtype, count in relationship_types.items()])
            summary_parts.append(f"具有{rel_summary}")
        
        # 重要實體
        top_entities = sorted(entities, key=lambda x: x.rank, reverse=True)[:3]
        if top_entities:
            entity_names = [e.name for e in top_entities]
            summary_parts.append(f"主要實體包括：{', '.join(entity_names)}")
        
        return "；".join(summary_parts) if summary_parts else "社群摘要"
    
    def _generate_community_content(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> str:
        """生成社群完整內容"""
        content_parts = []
        
        # 實體部分
        if entities:
            content_parts.append("## 實體列表")
            for entity in sorted(entities, key=lambda x: x.rank, reverse=True):
                content_parts.append(f"- **{entity.name}** ({entity.type}): {entity.description}")
        
        # 關係部分
        if relationships:
            content_parts.append("\n## 關係列表")
            entity_dict = {e.id: e for e in entities}
            
            for rel in sorted(relationships, key=lambda x: x.weight, reverse=True):
                source_name = entity_dict.get(rel.source_entity_id, {}).name if rel.source_entity_id in entity_dict else "未知實體"
                target_name = entity_dict.get(rel.target_entity_id, {}).name if rel.target_entity_id in entity_dict else "未知實體"
                
                content_parts.append(
                    f"- {source_name} --[{rel.relationship_type}]--> {target_name}: {rel.description}"
                )
        
        return "\n".join(content_parts) if content_parts else "無內容"
    
    def _calculate_community_rank(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> float:
        """計算社群排名"""
        if not entities:
            return 0.0
        
        # 基於實體排名的平均值
        entity_rank_avg = sum(e.rank for e in entities) / len(entities)
        
        # 基於關係權重的平均值
        if relationships:
            rel_weight_avg = sum(r.weight for r in relationships) / len(relationships)
        else:
            rel_weight_avg = 0.0
        
        # 基於社群大小的加權
        size_factor = min(len(entities) / 10.0, 1.0)  # 最大為 1.0
        
        # 綜合排名
        final_rank = (entity_rank_avg * 0.5 + rel_weight_avg * 0.3 + size_factor * 0.2)
        
        return min(final_rank, 1.0)  # 確保不超過 1.0