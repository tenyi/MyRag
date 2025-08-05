"""
基礎資料模型類別

提供所有資料模型的共同基礎功能，包括：
- 資料驗證
- 序列化/反序列化
- 通用方法
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field


class BaseModel(PydanticBaseModel):
    """所有資料模型的基礎類別"""

    model_config = ConfigDict(
        # 允許任意類型（如 numpy arrays）
        arbitrary_types_allowed=True,
        # 驗證賦值
        validate_assignment=True,
        # 使用 enum 值而不是名稱
        use_enum_values=True,
        # 序列化時排除 None 值
        exclude_none=True,
    )

    id: str = Field(default_factory=lambda: str(uuid4()), description="唯一識別碼")
    created_at: datetime = Field(default_factory=datetime.now, description="建立時間")
    updated_at: Optional[datetime] = Field(default=None, description="更新時間")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="額外元資料")

    def update_timestamp(self) -> None:
        """更新時間戳記"""
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """從字典建立實例"""
        return cls(**data)

    def to_json(self) -> str:
        """轉換為 JSON 字串"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "BaseModel":
        """從 JSON 字串建立實例"""
        return cls.model_validate_json(json_str)
