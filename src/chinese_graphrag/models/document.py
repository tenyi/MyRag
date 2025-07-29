"""
文件資料模型

定義文件的結構和驗證規則
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, validator

from .base import BaseModel


class Document(BaseModel):
    """文件模型
    
    代表系統中的一個文件，包含文件的基本資訊和內容
    """
    
    title: str = Field(..., description="文件標題", min_length=1, max_length=500)
    content: str = Field(..., description="文件內容", min_length=1)
    file_path: str = Field(..., description="檔案路徑")
    language: str = Field(default="zh", description="文件語言", pattern=r"^[a-z]{2}$")
    file_type: Optional[str] = Field(default=None, description="檔案類型")
    file_size: Optional[int] = Field(default=None, description="檔案大小（位元組）", ge=0)
    encoding: str = Field(default="utf-8", description="檔案編碼")
    
    @validator("file_path")
    def validate_file_path(cls, v: str) -> str:
        """驗證檔案路徑格式"""
        if not v.strip():
            raise ValueError("檔案路徑不能為空")
        return v.strip()
    
    @validator("content")
    def validate_content(cls, v: str) -> str:
        """驗證文件內容"""
        if not v.strip():
            raise ValueError("文件內容不能為空")
        return v.strip()
    
    @validator("title")
    def validate_title(cls, v: str) -> str:
        """驗證文件標題"""
        if not v.strip():
            raise ValueError("文件標題不能為空")
        return v.strip()
    
    @property
    def file_name(self) -> str:
        """取得檔案名稱"""
        return Path(self.file_path).name
    
    @property
    def file_extension(self) -> str:
        """取得檔案副檔名"""
        return Path(self.file_path).suffix.lower()
    
    @property
    def content_length(self) -> int:
        """取得內容長度"""
        return len(self.content)
    
    def get_summary(self, max_length: int = 200) -> str:
        """取得文件摘要"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."