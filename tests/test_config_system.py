"""配置管理系統測試。

測試配置管理的核心功能：
- 配置載入和驗證
- 環境變數管理
- 預設值處理
- 配置驗證
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.chinese_graphrag.config import (
    ConfigLoader,
    ConfigValidationError,
    GraphRAGConfig,
    apply_default_values,
    get_env_var,
    validate_config,
)
from src.chinese_graphrag.config.env import EnvironmentManager
from src.chinese_graphrag.config.validation import ConfigValidator


class TestConfigLoader:
    """配置載入器測試。"""
    
    def test_load_valid_config(self, tmp_path):
        """測試載入有效配置。"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "encoding_model": "cl100k_base",
            "models": {
                "test_llm": {
                    "type": "openai_chat",
                    "model": "gpt-4o-mini",
                    "api_key": "test-key",
                    "max_tokens": 2000,
                    "temperature": 0.0
                },
                "test_embedding": {
                    "type": "bge_m3",
                    "model": "BAAI/bge-m3",
                    "device": "cpu",
                    "batch_size": 16
                }
            },
            "vector_store": {
                "type": "lancedb",
                "uri": "./test_data/lancedb",
                "container_name": "test",
                "overwrite": False
            },
            "model_selection": {
                "default_llm": "test_llm",
                "default_embedding": "test_embedding"
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(config_file)
        config = loader.load_config()
        
        assert isinstance(config, GraphRAGConfig)
        assert config.encoding_model == "cl100k_base"
        assert len(config.models) == 2
        assert config.model_selection.default_llm == "test_llm"
    
    def test_load_config_with_env_vars(self, tmp_path):
        """測試載入包含環境變數的配置。"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "models": {
                "test_llm": {
                    "type": "openai_chat",
                    "model": "gpt-4o-mini",
                    "api_key": "${TEST_API_KEY}",
                    "max_tokens": "${TEST_MAX_TOKENS:4000}",
                    "temperature": 0.0
                }
            },
            "model_selection": {
                "default_llm": "test_llm",
                "default_embedding": "test_embedding"
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        # 設定環境變數
        with patch.dict(os.environ, {"TEST_API_KEY": "secret-key"}):
            loader = ConfigLoader(config_file)
            config = loader.load_config()
            
            llm_config = config.get_llm_config("test_llm")
            assert llm_config.api_key == "secret-key"
            assert llm_config.max_tokens == 4000  # 預設值
    
    def test_load_nonexistent_config(self, tmp_path):
        """測試載入不存在的配置檔案。"""
        config_file = tmp_path / "nonexistent.yaml"
        
        loader = ConfigLoader(config_file)
        with pytest.raises(Exception):
            loader.load_config()
    
    def test_create_default_config(self, tmp_path):
        """測試建立預設配置。"""
        output_file = tmp_path / "default_config.yaml"
        
        loader = ConfigLoader()
        created_file = loader.create_default_config(output_file)
        
        assert created_file.exists()
        
        # 驗證建立的配置可以載入
        loader = ConfigLoader(created_file)
        config = loader.load_config()
        assert isinstance(config, GraphRAGConfig)


class TestEnvironmentManager:
    """環境變數管理器測試。"""
    
    def test_get_env_var_with_default(self):
        """測試取得環境變數（使用預設值）。"""
        manager = EnvironmentManager()
        
        # 不存在的環境變數，使用預設值
        value = manager.get_var("NON_EXISTENT_VAR", default="default_value", required=False)
        assert value == "default_value"
    
    def test_get_env_var_required_missing(self):
        """測試取得必需但缺失的環境變數。"""
        manager = EnvironmentManager()
        
        with pytest.raises(Exception):
            manager.get_var("NON_EXISTENT_REQUIRED_VAR", required=True)
    
    def test_type_conversion(self):
        """測試環境變數類型轉換。"""
        manager = EnvironmentManager()
        
        with patch.dict(os.environ, {
            "TEST_INT": "42",
            "TEST_FLOAT": "3.14",
            "TEST_BOOL_TRUE": "true",
            "TEST_BOOL_FALSE": "false",
            "TEST_LIST": "a,b,c"
        }):
            assert manager.get_var("TEST_INT", var_type=int) == 42
            assert manager.get_var("TEST_FLOAT", var_type=float) == 3.14
            assert manager.get_var("TEST_BOOL_TRUE", var_type=bool) is True
            assert manager.get_var("TEST_BOOL_FALSE", var_type=bool) is False
            assert manager.get_var("TEST_LIST", var_type=list) == ["a", "b", "c"]
    
    def test_load_env_file(self, tmp_path):
        """測試載入 .env 檔案。"""
        env_file = tmp_path / ".env"
        env_content = """
# 測試環境變數
TEST_VAR1=value1
TEST_VAR2="value with spaces"
TEST_VAR3='single quotes'
# 註解行
TEST_VAR4=value4
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        # 清除可能存在的環境變數  
        test_vars = ["TEST_VAR1", "TEST_VAR2", "TEST_VAR3", "TEST_VAR4"]
        for var in test_vars:
            if var in os.environ:
                del os.environ[var]
        
        manager = EnvironmentManager(env_file)
        
        assert os.environ.get("TEST_VAR1") == "value1"
        assert os.environ.get("TEST_VAR2") == "value with spaces"
        assert os.environ.get("TEST_VAR3") == "single quotes"
        assert os.environ.get("TEST_VAR4") == "value4"


class TestConfigValidator:
    """配置驗證器測試。"""
    
    def test_validate_valid_config(self):
        """測試驗證有效配置。"""
        # 使用預設配置進行測試
        config_dict = apply_default_values({
            "models": {
                "test_llm": {
                    "type": "openai_chat",
                    "model": "gpt-4o-mini",
                    "api_key": "test-key"
                },
                "test_embedding": {
                    "type": "bge_m3",
                    "model": "BAAI/bge-m3"
                }
            },
            "model_selection": {
                "default_llm": "test_llm",
                "default_embedding": "test_embedding"
            }
        })
        
        config = GraphRAGConfig(**config_dict)
        validator = ConfigValidator()
        
        # 應該通過驗證
        assert validator.validate_config(config) is True
    
    def test_validate_invalid_chunk_config(self):
        """測試驗證無效的分塊配置。"""
        config_dict = apply_default_values({
            "chunks": {
                "size": 100,
                "overlap": 200  # 重疊大於大小
            }
        })
        
        config = GraphRAGConfig(**config_dict)
        validator = ConfigValidator()
        
        with pytest.raises(ConfigValidationError):
            validator.validate_config(config)
    
    def test_validate_missing_default_models(self):
        """測試驗證缺失預設模型的配置。"""
        config_dict = apply_default_values({
            "models": {
                "existing_model": {
                    "type": "openai_chat",
                    "model": "gpt-4o-mini"
                }
            },
            "model_selection": {
                "default_llm": "non_existent_model",  # 不存在的模型
                "default_embedding": "another_non_existent_model"
            }
        })
        
        config = GraphRAGConfig(**config_dict)
        validator = ConfigValidator()
        
        with pytest.raises(ConfigValidationError):
            validator.validate_config(config)


class TestConfigIntegration:
    """配置系統整合測試。"""
    
    def test_full_config_workflow(self, tmp_path):
        """測試完整的配置工作流程。"""
        # 1. 建立配置檔案
        config_file = tmp_path / "integration_test.yaml"
        config_data = {
            "encoding_model": "cl100k_base",
            "models": {
                "main_llm": {
                    "type": "openai_chat",
                    "model": "gpt-4o-mini",
                    "api_key": "${INTEGRATION_TEST_API_KEY:test-key}",
                    "max_tokens": 2000,
                    "temperature": 0.0
                },
                "main_embedding": {
                    "type": "bge_m3",
                    "model": "BAAI/bge-m3",
                    "device": "cpu",
                    "batch_size": 32
                }
            },
            "vector_store": {
                "type": "lancedb",
                "uri": str(tmp_path / "lancedb"),
                "container_name": "integration_test"
            },
            "chunks": {
                "size": 1000,
                "overlap": 200
            },
            "model_selection": {
                "default_llm": "main_llm",
                "default_embedding": "main_embedding"
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        # 2. 載入配置
        loader = ConfigLoader(config_file)
        config = loader.load_config()
        
        # 3. 驗證配置
        assert validate_config(config) is True
        
        # 4. 測試配置使用
        assert config.encoding_model == "cl100k_base"
        
        llm_config = config.get_default_llm_config()
        assert llm_config is not None
        assert llm_config.model == "gpt-4o-mini"
        assert llm_config.api_key == "test-key"  # 使用預設值
        
        embedding_config = config.get_default_embedding_config()
        assert embedding_config is not None
        assert embedding_config.model == "BAAI/bge-m3"
        assert embedding_config.device == "cpu"
    
    def test_config_with_validation_warnings(self, tmp_path):
        """測試配置驗證警告。"""
        config_file = tmp_path / "warning_test.yaml"
        config_data = {
            "models": {
                "test_llm": {
                    "type": "openai_chat",
                    "model": "gpt-4o-mini",
                    "max_tokens": 10000,  # 過大的值，應該產生警告
                    "temperature": 0.0
                }
            },
            "parallelization": {
                "num_threads": 32,  # 過大的值，應該產生警告
                "batch_size": 200   # 過大的值，應該產生警告  
            },
            "chunks": {
                "size": 5000,  # 過大的值，應該產生警告
                "overlap": 200
            },
            "model_selection": {
                "default_llm": "test_llm",
                "default_embedding": "test_embedding"
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(config_file)
        config = loader.load_config()
        
        # 應該能載入，但會有警告
        validator = ConfigValidator()
        
        # 捕獲警告
        with pytest.warns(UserWarning):
            assert validator.validate_config(config) is True
        
        # 檢查是否有警告記錄
        assert len(validator.warnings) > 0


# 測試夾具
@pytest.fixture
def temp_env_file(tmp_path):
    """建立臨時 .env 檔案。"""
    env_file = tmp_path / ".env"
    env_content = """
TEST_API_KEY=test-secret-key
TEST_MODEL=gpt-4o-mini
TEST_TEMPERATURE=0.7
TEST_DEBUG=true
"""
    
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    return env_file


@pytest.fixture
def sample_config_dict():
    """提供範例配置字典。"""
    return {
        "encoding_model": "cl100k_base",
        "models": {
            "default_chat_model": {
                "type": "openai_chat",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "max_tokens": 2000,
                "temperature": 0.0
            },
            "chinese_embedding_model": {
                "type": "bge_m3",
                "model": "BAAI/bge-m3",
                "device": "auto",
                "batch_size": 32
            }
        },
        "vector_store": {
            "type": "lancedb",
            "uri": "./test_data/lancedb",
            "container_name": "test"
        },
        "model_selection": {
            "default_llm": "default_chat_model",
            "default_embedding": "chinese_embedding_model"
        }
    }


if __name__ == "__main__":
    pytest.main([__file__])