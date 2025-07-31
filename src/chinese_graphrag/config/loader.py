"""
配置載入器

負責從 YAML 檔案和環境變數載入配置，並進行驗證
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from .models import (
    EmbeddingConfig,
    EmbeddingType,
    GraphRAGConfig,
    LLMConfig,
    LLMType,
)


class ConfigurationError(Exception):
    """配置錯誤異常"""
    pass


class ConfigLoader:
    """配置載入器"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化配置載入器
        
        Args:
            config_path: 配置檔案路徑，預設為 settings.yaml
        """
        self.config_path = config_path or Path("settings.yaml")
        self.env_pattern = re.compile(r'\$\{([^}]+)\}')

    def load_config(self) -> GraphRAGConfig:
        """
        載入配置
        
        Returns:
            GraphRAGConfig: 載入的配置物件
            
        Raises:
            ConfigurationError: 配置載入或驗證失敗
        """
        try:
            # 載入 YAML 配置
            raw_config = self._load_yaml_config()
            
            # 替換環境變數
            processed_config = self._substitute_env_vars(raw_config)
            
            # 處理模型配置
            processed_config = self._process_model_configs(processed_config)
            
            # 驗證並建立配置物件
            config = GraphRAGConfig(**processed_config)
            
            # 執行額外驗證
            self._validate_config(config)
            
            return config
            
        except ValidationError as e:
            raise ConfigurationError(f"配置驗證失敗: {e}")
        except Exception as e:
            raise ConfigurationError(f"載入配置時發生錯誤: {e}")

    def _load_yaml_config(self) -> Dict[str, Any]:
        """載入 YAML 配置檔案"""
        if not self.config_path.exists():
            raise ConfigurationError(f"配置檔案不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML 解析錯誤: {e}")

    def _substitute_env_vars(self, config: Any) -> Any:
        """遞迴替換配置中的環境變數"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._replace_env_vars_in_string(config)
        else:
            return config

    def _replace_env_vars_in_string(self, value: str) -> str:
        """替換字串中的環境變數"""
        def replace_match(match):
            env_var = match.group(1)
            # 支援預設值語法: ${VAR:default_value}
            if ':' in env_var:
                var_name, default_value = env_var.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                env_value = os.getenv(env_var)
                if env_value is None:
                    raise ConfigurationError(f"環境變數 {env_var} 未設定")
                return env_value
        
        return self.env_pattern.sub(replace_match, value)

    def _process_model_configs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """處理模型配置，將其轉換為適當的配置物件"""
        if 'models' not in config:
            return config
        
        processed_models = {}
        
        for model_name, model_config in config['models'].items():
            if not isinstance(model_config, dict):
                continue
                
            model_type = model_config.get('type')
            
            # 根據類型建立適當的配置物件
            if model_type in [t.value for t in LLMType]:
                try:
                    processed_models[model_name] = LLMConfig(**model_config)
                except ValidationError as e:
                    raise ConfigurationError(
                        f"LLM 模型 '{model_name}' 配置錯誤: {e}"
                    )
            elif model_type in [t.value for t in EmbeddingType]:
                try:
                    processed_models[model_name] = EmbeddingConfig(**model_config)
                except ValidationError as e:
                    raise ConfigurationError(
                        f"Embedding 模型 '{model_name}' 配置錯誤: {e}"
                    )
            else:
                raise ConfigurationError(
                    f"未知的模型類型: {model_type} (模型: {model_name})"
                )
        
        config['models'] = processed_models
        return config

    def _validate_config(self, config: GraphRAGConfig) -> None:
        """執行額外的配置驗證"""
        # 驗證預設模型是否存在
        default_llm = config.model_selection.default_llm
        if default_llm not in config.models:
            raise ConfigurationError(f"預設 LLM 模型 '{default_llm}' 未在 models 中定義")
        
        default_embedding = config.model_selection.default_embedding
        if default_embedding not in config.models:
            raise ConfigurationError(
                f"預設 Embedding 模型 '{default_embedding}' 未在 models 中定義"
            )
        
        # 驗證預設模型類型是否正確
        if not isinstance(config.models[default_llm], LLMConfig):
            raise ConfigurationError(f"預設 LLM 模型 '{default_llm}' 不是 LLM 配置")
        
        if not isinstance(config.models[default_embedding], EmbeddingConfig):
            raise ConfigurationError(
                f"預設 Embedding 模型 '{default_embedding}' 不是 Embedding 配置"
            )
        
        # 驗證備用模型
        for original, fallback in config.model_selection.fallback_models.items():
            if original not in config.models:
                raise ConfigurationError(f"備用模型映射中的原始模型 '{original}' 未定義")
            if fallback not in config.models:
                raise ConfigurationError(f"備用模型映射中的備用模型 '{fallback}' 未定義")
        
        # 驗證目錄路徑
        self._validate_directories(config)

    def _validate_directories(self, config: GraphRAGConfig) -> None:
        """驗證目錄配置"""
        # 建立輸出目錄（如果不存在）
        output_dir = Path(config.storage.base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cache_dir = Path(config.storage.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logs_dir = Path(config.storage.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 輸入目錄在實際使用時才檢查，這裡只是警告
        input_dir = Path(config.input.base_dir)
        if not input_dir.exists():
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"輸入目錄不存在，將在使用時建立: {input_dir}")

    def create_default_config(self, output_path: Optional[Path] = None) -> Path:
        """
        建立預設配置檔案
        
        Args:
            output_path: 輸出路徑，預設為 settings.yaml
            
        Returns:
            Path: 建立的配置檔案路徑
        """
        output_path = output_path or Path("settings.yaml")
        
        default_config = {
            "encoding_model": "cl100k_base",
            "models": {
                "default_chat_model": {
                    "api_key": "${GRAPHRAG_API_KEY}",
                    "type": "openai_chat",
                    "model": "gpt-4o",
                    "model_supports_json": True,
                    "max_tokens": 2000,
                    "temperature": 0.0
                },
                "default_embedding_model": {
                    "api_key": "${GRAPHRAG_API_KEY}",
                    "type": "openai_embedding",
                    "model": "text-embedding-3-small"
                },
                "chinese_embedding_model": {
                    "type": "bge_m3",
                    "model": "BAAI/bge-m3",
                    "device": "auto",
                    "batch_size": 32,
                    "normalize_embeddings": True,
                    "cache_enabled": True
                }
            },
            "vector_store": {
                "type": "lancedb",
                "uri": "./data/lancedb",
                "container_name": "default",
                "overwrite": False,
                "metric": "cosine"
            },
            "chinese_processing": {
                "tokenizer": "jieba",
                "enable_traditional_chinese": True,
                "enable_pos_tagging": False,
                "enable_ner": False
            },
            "input": {
                "file_type": "text",
                "supported_formats": ["txt", "pdf", "docx", "md"],
                "base_dir": "input",
                "file_encoding": "utf-8",
                "recursive": True
            },
            "chunks": {
                "size": 1000,
                "overlap": 200,
                "strategy": "token"
            },
            "indexing": {
                "enable_entity_extraction": True,
                "enable_relationship_extraction": True,
                "enable_community_detection": True,
                "enable_community_reports": True,
                "enable_llm_reports": False,
                "entity_types": ["organization", "person", "geo", "event"],
                "max_gleanings": 1,
                "max_cluster_size": 10,
                "min_community_size": 3,
                "max_community_size": 50,
                "enable_hierarchical_communities": True,
                "community_resolution": 1.0
            },
            "query": {
                "enable_global_search": True,
                "enable_local_search": True,
                "max_tokens": 2000,
                "temperature": 0.0,
                "top_k": 10,
                "similarity_threshold": 0.7
            },
            "storage": {
                "type": "file",
                "base_dir": "output",
                "cache_dir": "cache",
                "logs_dir": "logs"
            },
            "parallelization": {
                "num_threads": 4,
                "stagger": 0.3,
                "async_mode": "threaded",
                "batch_size": 10
            },
            "model_selection": {
                "default_llm": "default_chat_model",
                "default_embedding": "chinese_embedding_model",
                "cost_optimization": True,
                "quality_threshold": 0.8,
                "fallback_models": {
                    "chinese_embedding_model": "default_embedding_model"
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                default_config, 
                f, 
                default_flow_style=False, 
                allow_unicode=True,
                sort_keys=False
            )
        
        return output_path


def load_config(config_path: Optional[Path] = None) -> GraphRAGConfig:
    """
    便利函數：載入配置
    
    Args:
        config_path: 配置檔案路徑
        
    Returns:
        GraphRAGConfig: 載入的配置物件
    """
    loader = ConfigLoader(config_path)
    return loader.load_config()


def create_default_config(output_path: Optional[Path] = None) -> Path:
    """
    便利函數：建立預設配置檔案
    
    Args:
        output_path: 輸出路徑
        
    Returns:
        Path: 建立的配置檔案路徑
    """
    loader = ConfigLoader()
    return loader.create_default_config(output_path)