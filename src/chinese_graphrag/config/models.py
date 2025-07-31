"""
GraphRAG 配置資料模型

定義所有配置相關的 Pydantic 模型，支援多種 LLM 和 Embedding 模型配置
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class LLMType(str, Enum):
    """LLM 模型類型枚舉"""
    OPENAI_CHAT = "openai_chat"
    AZURE_OPENAI_CHAT = "azure_openai_chat"
    LOCAL_CHAT = "local_chat"


class EmbeddingType(str, Enum):
    """Embedding 模型類型枚舉"""
    OPENAI_EMBEDDING = "openai_embedding"
    AZURE_OPENAI_EMBEDDING = "azure_openai_embedding"
    BGE_M3 = "bge_m3"
    LOCAL_EMBEDDING = "local_embedding"


class VectorStoreType(str, Enum):
    """向量資料庫類型枚舉"""
    LANCEDB = "lancedb"
    CHROMA = "chroma"
    FAISS = "faiss"


class DeviceType(str, Enum):
    """設備類型枚舉"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class LLMConfig(BaseModel):
    """LLM 模型配置"""
    api_key: Optional[str] = Field(None, description="API 金鑰")
    type: LLMType = Field(..., description="LLM 類型")
    model: str = Field(..., description="模型名稱")
    model_supports_json: bool = Field(True, description="是否支援 JSON 輸出")
    api_base: Optional[str] = Field(None, description="API 基礎 URL")
    api_version: Optional[str] = Field(None, description="API 版本")
    organization: Optional[str] = Field(None, description="組織 ID")
    deployment_name: Optional[str] = Field(None, description="部署名稱（Azure）")
    audience: Optional[str] = Field(None, description="受眾（Azure）")
    max_tokens: int = Field(2000, description="最大 token 數")
    temperature: float = Field(0.0, description="溫度參數")
    timeout: int = Field(60, description="請求超時時間（秒）")
    max_retries: int = Field(3, description="最大重試次數")

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('溫度參數必須在 0.0 到 2.0 之間')
        return v


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置"""
    api_key: Optional[str] = Field(None, description="API 金鑰")
    type: EmbeddingType = Field(..., description="Embedding 類型")
    model: str = Field(..., description="模型名稱")
    api_base: Optional[str] = Field(None, description="API 基礎 URL")
    api_version: Optional[str] = Field(None, description="API 版本")
    organization: Optional[str] = Field(None, description="組織 ID")
    deployment_name: Optional[str] = Field(None, description="部署名稱（Azure）")
    audience: Optional[str] = Field(None, description="受眾（Azure）")
    device: DeviceType = Field(DeviceType.AUTO, description="運算設備")
    batch_size: int = Field(32, description="批次大小")
    max_length: int = Field(512, description="最大序列長度")
    normalize_embeddings: bool = Field(True, description="是否正規化向量")
    cache_enabled: bool = Field(True, description="是否啟用快取")

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError('批次大小必須大於 0')
        return v


class VectorStoreConfig(BaseModel):
    """向量資料庫配置"""
    type: VectorStoreType = Field(..., description="向量資料庫類型")
    uri: str = Field(..., description="資料庫 URI")
    container_name: str = Field("default", description="容器名稱")
    overwrite: bool = Field(False, description="是否覆寫現有資料")
    table_name: str = Field("vectors", description="表格名稱")
    dimension: Optional[int] = Field(None, description="向量維度")
    metric: str = Field("cosine", description="距離度量")
    index_cache_size: int = Field(256, description="索引快取大小（MB）")


class ChineseProcessingConfig(BaseModel):
    """中文文本處理配置"""
    tokenizer: str = Field("jieba", description="分詞器")
    enable_traditional_chinese: bool = Field(True, description="是否支援繁體中文")
    custom_dict_path: Optional[Path] = Field(None, description="自訂詞典路徑")
    stop_words_path: Optional[Path] = Field(None, description="停用詞檔案路徑")
    enable_pos_tagging: bool = Field(False, description="是否啟用詞性標註")
    enable_ner: bool = Field(False, description="是否啟用命名實體識別")


class InputConfig(BaseModel):
    """輸入配置"""
    file_type: str = Field("text", description="檔案類型")
    supported_formats: List[str] = Field(
        default=["txt", "pdf", "docx", "md"], 
        description="支援的檔案格式"
    )
    base_dir: str = Field("input", description="輸入目錄")
    file_encoding: str = Field("utf-8", description="檔案編碼")
    file_pattern: str = Field(".*\\.(txt|pdf|docx|md)$", description="檔案模式")
    recursive: bool = Field(True, description="是否遞迴搜尋子目錄")


class ChunkConfig(BaseModel):
    """文本分塊配置"""
    size: int = Field(1000, description="分塊大小")
    overlap: int = Field(200, description="重疊大小")
    group_by_columns: List[str] = Field(default=["id"], description="分組欄位")
    strategy: str = Field("token", description="分塊策略")

    @field_validator('overlap')
    @classmethod
    def validate_overlap(cls, v, info):
        if info.data.get('size') and v >= info.data['size']:
            raise ValueError('重疊大小必須小於分塊大小')
        return v


class IndexingConfig(BaseModel):
    """索引配置"""
    enable_entity_extraction: bool = Field(True, description="是否啟用實體提取")
    enable_relationship_extraction: bool = Field(True, description="是否啟用關係提取")
    enable_community_detection: bool = Field(True, description="是否啟用社群檢測")
    enable_community_reports: bool = Field(True, description="是否啟用社群報告生成")
    enable_llm_reports: bool = Field(False, description="是否使用 LLM 生成社群報告")
    entity_types: List[str] = Field(
        default=["organization", "person", "geo", "event"],
        description="實體類型"
    )
    max_gleanings: int = Field(1, description="最大提取輪數")
    max_cluster_size: int = Field(10, description="最大聚類大小")
    min_community_size: int = Field(3, description="最小社群大小")
    max_community_size: int = Field(50, description="最大社群大小")
    enable_hierarchical_communities: bool = Field(True, description="是否啟用層次化社群檢測")
    community_resolution: float = Field(1.0, description="社群檢測解析度")
    enable_graph_embedding: bool = Field(False, description="是否啟用圖嵌入")


class QueryConfig(BaseModel):
    """查詢配置"""
    enable_global_search: bool = Field(True, description="是否啟用全域搜尋")
    enable_local_search: bool = Field(True, description="是否啟用本地搜尋")
    enable_drift_search: bool = Field(False, description="是否啟用漂移搜尋")
    max_tokens: int = Field(2000, description="最大回應 token 數")
    temperature: float = Field(0.0, description="生成溫度")
    top_k: int = Field(10, description="檢索 top-k 結果")
    similarity_threshold: float = Field(0.7, description="相似度閾值")


class StorageConfig(BaseModel):
    """儲存配置"""
    type: str = Field("file", description="儲存類型")
    base_dir: str = Field("output", description="輸出目錄")
    cache_dir: str = Field("cache", description="快取目錄")
    logs_dir: str = Field("logs", description="日誌目錄")


class ParallelizationConfig(BaseModel):
    """並行處理配置"""
    num_threads: int = Field(4, description="執行緒數量")
    stagger: float = Field(0.3, description="請求間隔（秒）")
    async_mode: str = Field("threaded", description="非同步模式")
    batch_size: int = Field(10, description="批次處理大小")

    @field_validator('num_threads')
    @classmethod
    def validate_num_threads(cls, v):
        if v <= 0:
            raise ValueError('執行緒數量必須大於 0')
        return v


class ModelSelectionConfig(BaseModel):
    """模型選擇策略配置"""
    default_llm: str = Field("default_chat_model", description="預設 LLM 模型")
    default_embedding: str = Field("chinese_embedding_model", description="預設 Embedding 模型")
    cost_optimization: bool = Field(True, description="是否啟用成本優化")
    quality_threshold: float = Field(0.8, description="品質閾值")
    fallback_models: Dict[str, str] = Field(
        default_factory=dict, 
        description="備用模型映射"
    )


class GraphRAGConfig(BaseModel):
    """GraphRAG 主配置類別"""
    # 模型配置
    models: Dict[str, Union[LLMConfig, EmbeddingConfig]] = Field(
        default_factory=dict, 
        description="模型配置字典"
    )
    
    # 向量資料庫配置
    vector_store: VectorStoreConfig = Field(..., description="向量資料庫配置")
    
    # 中文處理配置
    chinese_processing: ChineseProcessingConfig = Field(
        default_factory=ChineseProcessingConfig,
        description="中文處理配置"
    )
    
    # 輸入配置
    input: InputConfig = Field(default_factory=InputConfig, description="輸入配置")
    
    # 分塊配置
    chunks: ChunkConfig = Field(default_factory=ChunkConfig, description="分塊配置")
    
    # 索引配置
    indexing: IndexingConfig = Field(
        default_factory=IndexingConfig, 
        description="索引配置"
    )
    
    # 查詢配置
    query: QueryConfig = Field(default_factory=QueryConfig, description="查詢配置")
    
    # 儲存配置
    storage: StorageConfig = Field(
        default_factory=StorageConfig, 
        description="儲存配置"
    )
    
    # 並行處理配置
    parallelization: ParallelizationConfig = Field(
        default_factory=ParallelizationConfig,
        description="並行處理配置"
    )
    
    # 模型選擇配置
    model_selection: ModelSelectionConfig = Field(
        default_factory=ModelSelectionConfig,
        description="模型選擇配置"
    )
    
    # 編碼模型
    encoding_model: str = Field("cl100k_base", description="編碼模型")
    
    # 跳過的工作流程
    skip_workflows: List[str] = Field(default_factory=list, description="跳過的工作流程")

    model_config = {
        "use_enum_values": True,
        "validate_assignment": True,
        "extra": "forbid"
    }

    def get_llm_config(self, model_name: str) -> Optional[LLMConfig]:
        """取得 LLM 配置"""
        config = self.models.get(model_name)
        if isinstance(config, LLMConfig):
            return config
        return None

    def get_embedding_config(self, model_name: str) -> Optional[EmbeddingConfig]:
        """取得 Embedding 配置"""
        config = self.models.get(model_name)
        if isinstance(config, EmbeddingConfig):
            return config
        return None

    def get_default_llm_config(self) -> Optional[LLMConfig]:
        """取得預設 LLM 配置"""
        return self.get_llm_config(self.model_selection.default_llm)

    def get_default_embedding_config(self) -> Optional[EmbeddingConfig]:
        """取得預設 Embedding 配置"""
        return self.get_embedding_config(self.model_selection.default_embedding)