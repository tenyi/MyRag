from enum import Enum, auto
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field


class VectorStoreType(str, Enum):
    """向量存儲類型"""
    LANCEDB = "lancedb"
    FAISS = "faiss"
    MILVUS = "milvus"
    WEAVIATE = "weaviate"


class VectorStoreConfig(BaseModel):
    """向量存儲配置"""
    type: VectorStoreType = Field(description="向量存儲類型")
    uri: str = Field(description="向量存儲URI")
    collection_prefix: str = Field(default="graphrag_", description="集合名稱前綴")
    connection_args: Optional[Dict[str, Any]] = Field(default=None, description="連接參數")


class GraphRAGConfig(BaseModel):
    """GraphRAG 配置"""
    models: Dict[str, Dict[str, Any]] = Field(description="模型配置")
    vector_store: VectorStoreConfig = Field(description="向量存儲配置")
    
    # 文本處理配置
    chunk_size: int = Field(default=512, description="文本塊大小")
    chunk_overlap: int = Field(default=128, description="文本塊重疊大小")
    
    # 實體和關係提取配置
    entity_extraction_model: Optional[str] = Field(default=None, description="實體提取模型名稱")
    relationship_extraction_model: Optional[str] = Field(default=None, description="關係提取模型名稱")
    
    # 嵌入配置
    embedding_model: Optional[str] = Field(default=None, description="嵌入模型名稱")
    embedding_dimension: int = Field(default=768, description="嵌入維度")
    
    # 社群檢測配置
    community_detection_algorithm: str = Field(default="louvain", description="社群檢測演算法")
    community_min_size: int = Field(default=3, description="最小社群大小")
    community_resolution: float = Field(default=1.0, description="社群解析度參數")