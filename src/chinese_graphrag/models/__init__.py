"""
中文 GraphRAG 系統資料模型

此模組包含系統中使用的所有核心資料模型，包括：
- Document: 文件模型
- TextUnit: 文本單元模型  
- Entity: 實體模型
- Relationship: 關係模型
- Community: 社群模型
"""

from .base import BaseModel
from .document import Document
from .text_unit import TextUnit
from .entity import Entity
from .relationship import Relationship
from .community import Community

__all__ = [
    "BaseModel",
    "Document",
    "TextUnit", 
    "Entity",
    "Relationship",
    "Community",
]