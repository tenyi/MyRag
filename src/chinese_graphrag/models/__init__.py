from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    """文件模型"""

    id: str = Field(default_factory=lambda: str(uuid4()), description="文件ID")
    title: str = Field(description="文件標題")
    content: str = Field(description="文件內容")
    file_path: Path = Field(description="文件路徑")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文件元數據")


class TextUnit(BaseModel):
    """文本單元模型"""

    id: str = Field(default_factory=lambda: str(uuid4()), description="文本單元ID")
    text: str = Field(description="文本內容")
    document_id: str = Field(description="所屬文件ID")
    chunk_index: int = Field(description="文本塊索引")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文本單元元數據")
    embedding: Optional[List[float]] = Field(default=None, description="文本嵌入向量")


class Entity(BaseModel):
    """實體模型"""

    id: str = Field(default_factory=lambda: str(uuid4()), description="實體ID")
    name: str = Field(description="實體名稱")
    type: str = Field(description="實體類型")
    description: Optional[str] = Field(default=None, description="實體描述")
    text_units: List[str] = Field(
        default_factory=list, description="相關文本單元ID列表"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="實體元數據")
    embedding: Optional[List[float]] = Field(default=None, description="實體嵌入向量")


class Relationship(BaseModel):
    """關係模型"""

    id: str = Field(default_factory=lambda: str(uuid4()), description="關係ID")
    source_entity_id: str = Field(description="源實體ID")
    target_entity_id: str = Field(description="目標實體ID")
    relationship_type: Optional[str] = Field(default=None, description="關係類型")
    description: str = Field(description="關係描述")
    text_units: List[str] = Field(
        default_factory=list, description="相關文本單元ID列表"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="關係元數據")
    weight: Optional[float] = Field(default=1.0, description="關係權重")


class Community(BaseModel):
    """社群模型"""

    id: str = Field(default_factory=lambda: str(uuid4()), description="社群ID")
    title: str = Field(description="社群標題")
    level: int = Field(description="社群層級")
    entities: List[str] = Field(description="社群包含的實體ID列表")
    relationships: List[str] = Field(description="社群包含的關係ID列表")
    summary: Optional[str] = Field(default=None, description="社群摘要")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="社群元數據")
    embedding: Optional[List[float]] = Field(default=None, description="社群嵌入向量")
    rank: Optional[float] = Field(default=0.0, description="社群排名")


class QueryResult(BaseModel):
    """查詢結果模型"""

    query: str = Field(description="原始查詢")
    text_units: List[TextUnit] = Field(default_factory=list, description="相關文本單元")
    entities: List[Entity] = Field(default_factory=list, description="相關實體")
    communities: List[Community] = Field(default_factory=list, description="相關社群")
    answer: Optional[str] = Field(default=None, description="生成的回答")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="查詢結果元數據")
