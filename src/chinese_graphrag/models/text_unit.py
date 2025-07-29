"""
文本單元資料模型

定義文本單元的結構和驗證規則
"""

from typing import Optional

import numpy as np
from pydantic import Field, validator

from .base import BaseModel


class TextUnit(BaseModel):
    """文本單元模型
    
    代表文件中的一個文本片段，是索引和檢索的基本單位
    """
    
    text: str = Field(..., description="文本內容", min_length=1)
    document_id: str = Field(..., description="所屬文件 ID")
    chunk_index: int = Field(..., description="在文件中的塊索引", ge=0)
    start_position: Optional[int] = Field(default=None, description="在原文件中的起始位置", ge=0)
    end_position: Optional[int] = Field(default=None, description="在原文件中的結束位置", ge=0)
    embedding: Optional[np.ndarray] = Field(default=None, description="文本向量表示")
    token_count: Optional[int] = Field(default=None, description="Token 數量", ge=0)
    
    @validator("text")
    def validate_text(cls, v: str) -> str:
        """驗證文本內容"""
        if not v.strip():
            raise ValueError("文本內容不能為空")
        return v.strip()
    
    @validator("document_id")
    def validate_document_id(cls, v: str) -> str:
        """驗證文件 ID"""
        if not v.strip():
            raise ValueError("文件 ID 不能為空")
        return v.strip()
    
    @validator("end_position")
    def validate_positions(cls, v: Optional[int], values: dict) -> Optional[int]:
        """驗證位置資訊的一致性"""
        if v is not None and "start_position" in values:
            start_pos = values.get("start_position")
            if start_pos is not None and v <= start_pos:
                raise ValueError("結束位置必須大於起始位置")
        return v
    
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
    def text_length(self) -> int:
        """取得文本長度"""
        return len(self.text)
    
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
    
    def get_text_preview(self, max_length: int = 100) -> str:
        """取得文本預覽"""
        if len(self.text) <= max_length:
            return self.text
        return self.text[:max_length] + "..."