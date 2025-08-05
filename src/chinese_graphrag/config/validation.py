"""配置驗證和預設值處理系統。

本模組提供配置的驗證、預設值處理和配置完整性檢查功能，包括：
- 配置結構驗證
- 業務邏輯驗證
- 預設值應用
- 配置相依性檢查
- 配置升級和遷移
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from .models import (
    EmbeddingConfig,
    EmbeddingType,
    GraphRAGConfig,
    LLMConfig,
    LLMType,
)


class ConfigValidationError(Exception):
    """配置驗證錯誤。"""
    pass


class ConfigValidationWarning(UserWarning):
    """配置驗證警告。"""
    pass


class ConfigValidator:
    """配置驗證器。"""
    
    def __init__(self):
        """初始化配置驗證器。"""
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: GraphRAGConfig) -> Dict[str, Any]:
        """完整驗證配置。
        
        Args:
            config: 要驗證的配置
            
        Returns:
            驗證結果字典，包含 'valid', 'errors', 'warnings' 鍵
        """
        self.errors.clear()
        self.warnings.clear()
        
        try:
            # 基礎結構驗證（Pydantic）
            self._validate_basic_structure(config)
            
            # 模型配置驗證
            self._validate_model_configs(config)
            
            # 業務邏輯驗證
            self._validate_business_logic(config)
            
            # 相依性驗證
            self._validate_dependencies(config)
            
            # 資源和路徑驗證
            self._validate_resources(config)
            
            # 效能相關驗證
            self._validate_performance_settings(config)
            
            # 發出警告
            for warning in self.warnings:
                warnings.warn(warning, ConfigValidationWarning)
            
            # 回傳驗證結果
            return {
                'valid': len(self.errors) == 0,
                'errors': self.errors.copy(),
                'warnings': self.warnings.copy()
            }
            
        except ValidationError as e:
            self.errors.append(f"配置結構驗證失敗: {e}")
            return {
                'valid': False,
                'errors': self.errors.copy(),
                'warnings': self.warnings.copy()
            }
        except Exception as e:
            self.errors.append(f"驗證過程發生錯誤: {e}")
            return {
                'valid': False,
                'errors': self.errors.copy(),
                'warnings': self.warnings.copy()
            }
    
    def _validate_basic_structure(self, config: GraphRAGConfig) -> None:
        """驗證基礎配置結構。"""
        # 檢查必要的配置區塊
        required_sections = [
            "models", "vector_store", "chunks", "indexing", 
            "query", "storage", "model_selection"
        ]
        
        for section in required_sections:
            if not hasattr(config, section) or getattr(config, section) is None:
                self.errors.append(f"缺少必要配置區塊: {section}")
    
    def _validate_model_configs(self, config: GraphRAGConfig) -> None:
        """驗證模型配置。"""
        if not config.models:
            self.errors.append("至少需要配置一個模型")
            return
        
        # 驗證預設模型存在
        default_llm = config.model_selection.default_llm
        default_embedding = config.model_selection.default_embedding
        
        if default_llm not in config.models:
            self.errors.append(f"預設 LLM 模型 '{default_llm}' 未在 models 中定義")
        
        if default_embedding not in config.models:
            self.errors.append(f"預設 Embedding 模型 '{default_embedding}' 未在 models 中定義")
        
        # 驗證模型類型匹配
        if default_llm in config.models:
            if not isinstance(config.models[default_llm], LLMConfig):
                self.errors.append(f"預設 LLM 模型 '{default_llm}' 不是 LLM 配置")
        
        if default_embedding in config.models:
            if not isinstance(config.models[default_embedding], EmbeddingConfig):
                self.errors.append(f"預設 Embedding 模型 '{default_embedding}' 不是 Embedding 配置")
        
        # 驗證各個模型配置
        for name, model_config in config.models.items():
            if isinstance(model_config, LLMConfig):
                self._validate_llm_config(name, model_config)
            elif isinstance(model_config, EmbeddingConfig):
                self._validate_embedding_config(name, model_config)
    
    def _validate_llm_config(self, name: str, config: LLMConfig) -> None:
        """驗證 LLM 配置。"""
        # 檢查 API 金鑰（對於需要的提供者）
        if config.type in [LLMType.OPENAI_CHAT, LLMType.AZURE_OPENAI_CHAT]:
            if not config.api_key:
                self.warnings.append(f"LLM 模型 '{name}' 缺少 API 金鑰")
        
        # 檢查 Azure 特定配置
        if config.type == LLMType.AZURE_OPENAI_CHAT:
            if not config.api_base:
                self.errors.append(f"Azure LLM 模型 '{name}' 缺少 api_base")
            if not config.api_version:
                self.errors.append(f"Azure LLM 模型 '{name}' 缺少 api_version")
        
        # 檢查參數範圍
        if config.temperature < 0 or config.temperature > 2:
            self.errors.append(f"LLM 模型 '{name}' 的 temperature 必須在 0-2 之間")
        
        if config.max_tokens <= 0:
            self.errors.append(f"LLM 模型 '{name}' 的 max_tokens 必須大於 0")
    
    def _validate_embedding_config(self, name: str, config: EmbeddingConfig) -> None:
        """驗證 Embedding 配置。"""
        # 檢查 API 金鑰（對於需要的提供者）
        if config.type in [EmbeddingType.OPENAI_EMBEDDING, EmbeddingType.AZURE_OPENAI_EMBEDDING]:
            if not config.api_key:
                self.warnings.append(f"Embedding 模型 '{name}' 缺少 API 金鑰")
        
        # 檢查 Azure 特定配置
        if config.type == EmbeddingType.AZURE_OPENAI_EMBEDDING:
            if not config.api_base:
                self.errors.append(f"Azure Embedding 模型 '{name}' 缺少 api_base")
        
        # 檢查維度
        if config.vector_size <= 0:
            self.errors.append(f"Embedding 模型 '{name}' 的 vector_size 必須大於 0")
        
        # 檢查批次大小
        if config.batch_size <= 0:
            self.errors.append(f"Embedding 模型 '{name}' 的 batch_size 必須大於 0")
    
    def _validate_business_logic(self, config: GraphRAGConfig) -> None:
        """驗證業務邏輯。"""
        # 驗證文本分塊配置
        if config.chunks.overlap >= config.chunks.size:
            self.errors.append("chunk_overlap 必須小於 chunk_size")
        
        if config.chunks.size <= 0:
            self.errors.append("chunk_size 必須大於 0")
        
        # 驗證索引配置
        if config.indexing.max_cluster_size <= 0:
            self.errors.append("max_cluster_size 必須大於 0")
        
        if config.indexing.min_community_size <= 0:
            self.errors.append("min_community_size 必須大於 0")
        
        if config.indexing.min_community_size > config.indexing.max_community_size:
            self.errors.append("min_community_size 不能大於 max_community_size")
        
        # 驗證查詢配置
        if config.query.top_k <= 0:
            self.errors.append("query.top_k 必須大於 0")
        
        if config.query.similarity_threshold < 0 or config.query.similarity_threshold > 1:
            self.errors.append("similarity_threshold 必須在 0-1 之間")
        
        # 驗證並行配置
        if config.parallelization.num_threads <= 0:
            self.errors.append("num_threads 必須大於 0")
        
        if config.parallelization.batch_size <= 0:
            self.errors.append("batch_size 必須大於 0")
    
    def _validate_dependencies(self, config: GraphRAGConfig) -> None:
        """驗證配置間的相依性。"""
        # 驗證備用模型
        for original, fallback in config.model_selection.fallback_models.items():
            if original not in config.models:
                self.errors.append(f"備用模型映射中的原始模型 '{original}' 未定義")
            
            if fallback not in config.models:
                self.errors.append(f"備用模型映射中的備用模型 '{fallback}' 未定義")
            
            # 檢查模型類型是否匹配
            if (original in config.models and fallback in config.models):
                original_type = type(config.models[original])
                fallback_type = type(config.models[fallback])
                
                if original_type != fallback_type:
                    self.errors.append(
                        f"備用模型 '{fallback}' 的類型與原始模型 '{original}' 不匹配"
                    )
        
        # 驗證向量資料庫配置與 Embedding 的相容性
        embedding_config = config.get_default_embedding_config()
        if embedding_config and hasattr(embedding_config, 'vector_size'):
            # 這裡可以添加向量維度一致性檢查
            pass
    
    def _validate_resources(self, config: GraphRAGConfig) -> None:
        """驗證資源和路徑配置。"""
        # 檢查關鍵目錄
        storage_dir = Path(config.storage.base_dir)
        if storage_dir.exists() and not storage_dir.is_dir():
            self.errors.append(f"儲存路徑不是目錄: {storage_dir}")
        
        # 檢查向量資料庫路徑
        vector_db_path = Path(config.vector_store.uri)
        if vector_db_path.exists() and not vector_db_path.is_dir():
            self.errors.append(f"向量資料庫路徑不是目錄: {vector_db_path}")
        
        # 檢查輸入目錄（如果存在）
        input_dir = Path(config.input.base_dir)
        if input_dir.exists() and not input_dir.is_dir():
            self.errors.append(f"輸入路徑不是目錄: {input_dir}")
    
    def _validate_performance_settings(self, config: GraphRAGConfig) -> None:
        """驗證效能相關設定。"""
        # 檢查記憶體使用可能過高的設定
        if config.parallelization.num_threads > 16:
            self.warnings.append(
                f"num_threads ({config.parallelization.num_threads}) 過高，可能導致記憶體不足"
            )
        
        if config.chunks.size > 4000:
            self.warnings.append(
                f"chunk_size ({config.chunks.size}) 過大，可能影響處理效能"
            )
        
        if config.parallelization.batch_size > 100:
            self.warnings.append(
                f"batch_size ({config.parallelization.batch_size}) 過大，可能導致記憶體問題"
            )
        
        # 檢查 LLM 設定
        default_llm_config = config.get_default_llm_config()
        if default_llm_config and default_llm_config.max_tokens > 8000:
            self.warnings.append(
                f"LLM max_tokens ({default_llm_config.max_tokens}) 過大，可能增加成本"
            )


class DefaultConfigProvider:
    """預設配置提供者。"""
    
    @staticmethod
    def apply_defaults(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """應用預設值到配置字典。
        
        Args:
            config_dict: 原始配置字典
            
        Returns:
            應用預設值後的配置字典
        """
        # 深度複製以避免修改原始配置
        import copy
        result = copy.deepcopy(config_dict)
        
        # 應用預設值
        DefaultConfigProvider._apply_model_defaults(result)
        DefaultConfigProvider._apply_processing_defaults(result)
        DefaultConfigProvider._apply_storage_defaults(result)
        DefaultConfigProvider._apply_performance_defaults(result)
        
        return result
    
    @staticmethod
    def _apply_model_defaults(config: Dict[str, Any]) -> None:
        """應用模型相關預設值。"""
        if "models" not in config:
            config["models"] = {}
        
        if "model_selection" not in config:
            config["model_selection"] = {
                "default_llm": "default_chat_model",
                "default_embedding": "chinese_embedding_model",
                "cost_optimization": True,
                "quality_threshold": 0.8,
                "fallback_models": {}
            }
    
    @staticmethod
    def _apply_processing_defaults(config: Dict[str, Any]) -> None:
        """應用處理相關預設值。"""
        if "chunks" not in config:
            config["chunks"] = {
                "size": 1000,
                "overlap": 200,
                "strategy": "token"
            }
        
        if "chinese_processing" not in config:
            config["chinese_processing"] = {
                "tokenizer": "jieba",
                "enable_traditional_chinese": True,
                "enable_pos_tagging": False,
                "enable_ner": False
            }
    
    @staticmethod
    def _apply_storage_defaults(config: Dict[str, Any]) -> None:
        """應用儲存相關預設值。"""
        if "storage" not in config:
            config["storage"] = {
                "type": "file",
                "base_dir": "output",
                "cache_dir": "cache",
                "logs_dir": "logs"
            }
        
        if "vector_store" not in config:
            config["vector_store"] = {
                "type": "lancedb",
                "uri": "./data/lancedb",
                "container_name": "default",
                "overwrite": False,
                "metric": "cosine"
            }
    
    @staticmethod
    def _apply_performance_defaults(config: Dict[str, Any]) -> None:
        """應用效能相關預設值。"""
        if "parallelization" not in config:
            config["parallelization"] = {
                "num_threads": 4,
                "stagger": 0.3,
                "async_mode": "threaded",
                "batch_size": 10
            }


def validate_config(config: GraphRAGConfig) -> bool:
    """便利函數：驗證配置。
    
    Args:
        config: 要驗證的配置
        
    Returns:
        是否驗證通過
        
    Raises:
        ConfigValidationError: 驗證失敗時
    """
    validator = ConfigValidator()
    result = validator.validate_config(config)
    
    # 如果有錯誤，拋出異常
    if not result['valid']:
        error_msg = "配置驗證失敗:\n" + "\n".join(f"- {error}" for error in result['errors'])
        raise ConfigValidationError(error_msg)
    
    return result['valid']


def apply_default_values(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """便利函數：應用預設值。
    
    Args:
        config_dict: 原始配置字典
        
    Returns:
        應用預設值後的配置字典
    """
    return DefaultConfigProvider.apply_defaults(config_dict)