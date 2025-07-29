"""
關係資料模型

定義實體間關係的結構和驗證規則
"""

from typing import List, Optional

from pydantic import Field, validator

from .base import BaseModel


class Relationship(BaseModel):
    """關係模型
    
    代表兩個實體之間的關係，包含關係的類型、描述和權重等資訊
    """
    
    source_entity_id: str = Field(..., description="來源實體 ID")
    target_entity_id: str = Field(..., description="目標實體 ID")
    relationship_type: str = Field(..., description="關係類型", min_length=1, max_length=100)
    description: str = Field(..., description="關係描述", min_length=1)
    weight: float = Field(default=1.0, description="關係權重", ge=0.0, le=1.0)
    text_units: List[str] = Field(default_factory=list, description="支持此關係的文本單元 ID 列表")
    confidence: float = Field(default=1.0, description="關係提取置信度", ge=0.0, le=1.0)
    frequency: int = Field(default=1, description="關係在文件中出現的頻率", ge=1)
    bidirectional: bool = Field(default=False, description="是否為雙向關係")
    
    @validator("source_entity_id")
    def validate_source_entity_id(cls, v: str) -> str:
        """驗證來源實體 ID"""
        if not v.strip():
            raise ValueError("來源實體 ID 不能為空")
        return v.strip()
    
    @validator("target_entity_id")
    def validate_target_entity_id(cls, v: str, values: dict) -> str:
        """驗證目標實體 ID"""
        if not v.strip():
            raise ValueError("目標實體 ID 不能為空")
        
        # 檢查是否與來源實體相同
        source_id = values.get("source_entity_id")
        if source_id and v.strip() == source_id:
            raise ValueError("目標實體不能與來源實體相同")
        
        return v.strip()
    
    @validator("relationship_type")
    def validate_relationship_type(cls, v: str) -> str:
        """驗證關係類型"""
        if not v.strip():
            raise ValueError("關係類型不能為空")
        return v.strip().upper()
    
    @validator("description")
    def validate_description(cls, v: str) -> str:
        """驗證關係描述"""
        if not v.strip():
            raise ValueError("關係描述不能為空")
        return v.strip()
    
    @validator("text_units")
    def validate_text_units(cls, v: List[str]) -> List[str]:
        """驗證文本單元 ID 列表"""
        # 移除空字串和重複項目
        cleaned = list(set(unit.strip() for unit in v if unit.strip()))
        return cleaned
    
    @property
    def text_unit_count(self) -> int:
        """取得支持此關係的文本單元數量"""
        return len(self.text_units)
    
    @property
    def entity_pair(self) -> tuple[str, str]:
        """取得實體對"""
        return (self.source_entity_id, self.target_entity_id)
    
    @property
    def is_self_relationship(self) -> bool:
        """檢查是否為自我關係（理論上不應該存在）"""
        return self.source_entity_id == self.target_entity_id
    
    def add_text_unit(self, text_unit_id: str) -> None:
        """新增支持此關係的文本單元"""
        if text_unit_id.strip() and text_unit_id not in self.text_units:
            self.text_units.append(text_unit_id.strip())
            self.update_timestamp()
    
    def remove_text_unit(self, text_unit_id: str) -> bool:
        """移除支持此關係的文本單元"""
        if text_unit_id in self.text_units:
            self.text_units.remove(text_unit_id)
            self.update_timestamp()
            return True
        return False
    
    def update_weight(self, new_weight: float) -> None:
        """更新關係權重"""
        if 0.0 <= new_weight <= 1.0:
            self.weight = new_weight
            self.update_timestamp()
        else:
            raise ValueError("權重必須在 0.0 到 1.0 之間")
    
    def get_reverse_relationship(self) -> "Relationship":
        """取得反向關係（如果是雙向關係）"""
        if not self.bidirectional:
            raise ValueError("此關係不是雙向關係")
        
        return Relationship(
            source_entity_id=self.target_entity_id,
            target_entity_id=self.source_entity_id,
            relationship_type=self.relationship_type,
            description=f"反向關係: {self.description}",
            weight=self.weight,
            text_units=self.text_units.copy(),
            confidence=self.confidence,
            frequency=self.frequency,
            bidirectional=True,
            metadata=self.metadata.copy()
        )