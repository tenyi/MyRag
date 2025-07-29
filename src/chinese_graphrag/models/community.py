"""
社群資料模型

定義社群的結構和驗證規則
"""

from typing import List, Optional

import numpy as np
from pydantic import Field, validator

from .base import BaseModel


class Community(BaseModel):
    """社群模型
    
    代表實體和關係的聚集，形成有意義的主題群組
    """
    
    title: str = Field(..., description="社群標題", min_length=1, max_length=200)
    level: int = Field(..., description="社群層級", ge=0)
    entities: List[str] = Field(default_factory=list, description="實體 ID 列表")
    relationships: List[str] = Field(default_factory=list, description="關係 ID 列表")
    summary: str = Field(..., description="社群摘要", min_length=1)
    full_content: str = Field(..., description="社群完整內容", min_length=1)
    embedding: Optional[np.ndarray] = Field(default=None, description="社群向量表示")
    rank: float = Field(default=0.0, description="社群重要性排名", ge=0.0, le=1.0)
    size: Optional[int] = Field(default=None, description="社群大小", ge=0)
    density: Optional[float] = Field(default=None, description="社群密度", ge=0.0, le=1.0)
    parent_community_id: Optional[str] = Field(default=None, description="父社群 ID")
    child_communities: List[str] = Field(default_factory=list, description="子社群 ID 列表")
    
    @validator("title")
    def validate_title(cls, v: str) -> str:
        """驗證社群標題"""
        if not v.strip():
            raise ValueError("社群標題不能為空")
        return v.strip()
    
    @validator("summary")
    def validate_summary(cls, v: str) -> str:
        """驗證社群摘要"""
        if not v.strip():
            raise ValueError("社群摘要不能為空")
        return v.strip()
    
    @validator("full_content")
    def validate_full_content(cls, v: str) -> str:
        """驗證社群完整內容"""
        if not v.strip():
            raise ValueError("社群完整內容不能為空")
        return v.strip()
    
    @validator("entities")
    def validate_entities(cls, v: List[str]) -> List[str]:
        """驗證實體 ID 列表"""
        # 移除空字串和重複項目
        cleaned = list(set(entity.strip() for entity in v if entity.strip()))
        return cleaned
    
    @validator("relationships")
    def validate_relationships(cls, v: List[str]) -> List[str]:
        """驗證關係 ID 列表"""
        # 移除空字串和重複項目
        cleaned = list(set(rel.strip() for rel in v if rel.strip()))
        return cleaned
    
    @validator("child_communities")
    def validate_child_communities(cls, v: List[str]) -> List[str]:
        """驗證子社群 ID 列表"""
        # 移除空字串和重複項目
        cleaned = list(set(child.strip() for child in v if child.strip()))
        return cleaned
    
    @validator("embedding")
    def validate_embedding(cls, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """驗證 embedding 向量"""
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("embedding 必須是 numpy array")
            if v.ndim != 1:
                raise ValueError("embedding 必須是一維向量")
            if len(v) == 0:
                raise ValueError("embedding 不能為空向量")
        return v
    
    @property
    def entity_count(self) -> int:
        """取得實體數量"""
        return len(self.entities)
    
    @property
    def relationship_count(self) -> int:
        """取得關係數量"""
        return len(self.relationships)
    
    @property
    def child_community_count(self) -> int:
        """取得子社群數量"""
        return len(self.child_communities)
    
    @property
    def has_embedding(self) -> bool:
        """檢查是否有 embedding"""
        return self.embedding is not None
    
    @property
    def embedding_dimension(self) -> Optional[int]:
        """取得 embedding 維度"""
        if self.embedding is not None:
            return len(self.embedding)
        return None
    
    @property
    def has_parent(self) -> bool:
        """檢查是否有父社群"""
        return self.parent_community_id is not None
    
    @property
    def has_children(self) -> bool:
        """檢查是否有子社群"""
        return len(self.child_communities) > 0
    
    @property
    def is_leaf_community(self) -> bool:
        """檢查是否為葉子社群（沒有子社群）"""
        return not self.has_children
    
    @property
    def is_root_community(self) -> bool:
        """檢查是否為根社群（沒有父社群）"""
        return not self.has_parent
    
    def add_entity(self, entity_id: str) -> None:
        """新增實體到社群"""
        if entity_id.strip() and entity_id not in self.entities:
            self.entities.append(entity_id.strip())
            self.update_timestamp()
    
    def remove_entity(self, entity_id: str) -> bool:
        """從社群移除實體"""
        if entity_id in self.entities:
            self.entities.remove(entity_id)
            self.update_timestamp()
            return True
        return False
    
    def add_relationship(self, relationship_id: str) -> None:
        """新增關係到社群"""
        if relationship_id.strip() and relationship_id not in self.relationships:
            self.relationships.append(relationship_id.strip())
            self.update_timestamp()
    
    def remove_relationship(self, relationship_id: str) -> bool:
        """從社群移除關係"""
        if relationship_id in self.relationships:
            self.relationships.remove(relationship_id)
            self.update_timestamp()
            return True
        return False
    
    def add_child_community(self, child_id: str) -> None:
        """新增子社群"""
        if child_id.strip() and child_id not in self.child_communities:
            self.child_communities.append(child_id.strip())
            self.update_timestamp()
    
    def remove_child_community(self, child_id: str) -> bool:
        """移除子社群"""
        if child_id in self.child_communities:
            self.child_communities.remove(child_id)
            self.update_timestamp()
            return True
        return False
    
    def update_rank(self, new_rank: float) -> None:
        """更新社群排名"""
        if 0.0 <= new_rank <= 1.0:
            self.rank = new_rank
            self.update_timestamp()
        else:
            raise ValueError("排名必須在 0.0 到 1.0 之間")
    
    def calculate_size(self) -> int:
        """計算社群大小（實體數量 + 關係數量）"""
        calculated_size = self.entity_count + self.relationship_count
        self.size = calculated_size
        self.update_timestamp()
        return calculated_size
    
    def get_summary_preview(self, max_length: int = 200) -> str:
        """取得摘要預覽"""
        if len(self.summary) <= max_length:
            return self.summary
        return self.summary[:max_length] + "..."