"""
實體資料模型

定義實體的結構和驗證規則
"""

from typing import List, Optional

import numpy as np
from pydantic import Field, validator

from .base import BaseModel


class Entity(BaseModel):
    """實體模型
    
    代表從文件中提取的實體，包含實體的基本資訊和關聯資料
    """
    
    name: str = Field(..., description="實體名稱", min_length=1, max_length=200)
    type: str = Field(..., description="實體類型", min_length=1, max_length=50)
    description: str = Field(..., description="實體描述", min_length=1)
    text_units: List[str] = Field(default_factory=list, description="關聯的文本單元 ID 列表")
    embedding: Optional[np.ndarray] = Field(default=None, description="實體向量表示")
    community_id: Optional[str] = Field(default=None, description="所屬社群 ID")
    rank: float = Field(default=0.0, description="實體重要性排名", ge=0.0, le=1.0)
    frequency: int = Field(default=1, description="在文件中出現的頻率", ge=1)
    confidence: float = Field(default=1.0, description="提取置信度", ge=0.0, le=1.0)
    
    @validator("name")
    def validate_name(cls, v: str) -> str:
        """驗證實體名稱"""
        if not v.strip():
            raise ValueError("實體名稱不能為空")
        return v.strip()
    
    @validator("type")
    def validate_type(cls, v: str) -> str:
        """驗證實體類型"""
        if not v.strip():
            raise ValueError("實體類型不能為空")
        return v.strip().upper()
    
    @validator("description")
    def validate_description(cls, v: str) -> str:
        """驗證實體描述"""
        if not v.strip():
            raise ValueError("實體描述不能為空")
        return v.strip()
    
    @validator("text_units")
    def validate_text_units(cls, v: List[str]) -> List[str]:
        """驗證文本單元 ID 列表"""
        # 移除空字串和重複項目
        cleaned = list(set(unit.strip() for unit in v if unit.strip()))
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
    def text_unit_count(self) -> int:
        """取得關聯文本單元數量"""
        return len(self.text_units)
    
    @property
    def has_community(self) -> bool:
        """檢查是否屬於某個社群"""
        return self.community_id is not None
    
    def add_text_unit(self, text_unit_id: str) -> None:
        """新增關聯的文本單元"""
        if text_unit_id.strip() and text_unit_id not in self.text_units:
            self.text_units.append(text_unit_id.strip())
            self.update_timestamp()
    
    def remove_text_unit(self, text_unit_id: str) -> bool:
        """移除關聯的文本單元"""
        if text_unit_id in self.text_units:
            self.text_units.remove(text_unit_id)
            self.update_timestamp()
            return True
        return False
    
    def update_rank(self, new_rank: float) -> None:
        """更新實體排名"""
        if 0.0 <= new_rank <= 1.0:
            self.rank = new_rank
            self.update_timestamp()
        else:
            raise ValueError("排名必須在 0.0 到 1.0 之間")