"""
配置載入器測試

測試 YAML 配置檔案載入、環境變數替換和配置驗證
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from chinese_graphrag.config.loader import ConfigLoader, ConfigurationError
from chinese_graphrag.config.models import EmbeddingType, LLMType


class TestConfigLoader:
    """配置載入器測試"""

    def test_load_valid_config(self):
        """測試載入有效配置"""
        config_data = {
            "models": {
                "default_chat_model": {
                    "type": "openai_chat",
                    "model": "gpt-5-mini",
                    "api_key": "test-key",
                },
                "ollama_embedding_model": {
                    "type": "bge_m3",
                    "model": "BAAI/bge-m3",
                    "device": "auto",
                },
            },
            "vector_store": {"type": "lancedb", "uri": "./data/lancedb"},
            "model_selection": {
                "default_llm": "default_chat_model",
                "default_embedding": "ollama_embedding_model",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            config = loader.load_config()
            assert len(config.models) == 2
            assert config.vector_store.type.value == "lancedb"

            # 檢查模型配置類型
            llm_config = config.get_llm_config("default_chat_model")
            assert llm_config is not None
            assert llm_config.type == LLMType.OPENAI_CHAT

            embedding_config = config.get_embedding_config("ollama_embedding_model")
            assert embedding_config is not None
            assert embedding_config.type == EmbeddingType.BGE_M3

        finally:
            config_path.unlink()

    def test_env_var_substitution(self):
        """測試環境變數替換"""
        # 設定測試環境變數
        os.environ["TEST_API_KEY"] = "secret-key"
        os.environ["TEST_MODEL"] = "gpt-4"

        config_data = {
            "models": {
                "test_llm_model": {
                    "type": "openai_chat",
                    "model": "${TEST_MODEL}",
                    "api_key": "${TEST_API_KEY}",
                },
                "test_embedding_model": {"type": "bge_m3", "model": "BAAI/bge-m3"},
            },
            "vector_store": {"type": "lancedb", "uri": "./data/lancedb"},
            "model_selection": {
                "default_llm": "test_llm_model",
                "default_embedding": "test_embedding_model",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            config = loader.load_config()

            llm_config = config.get_llm_config("test_llm_model")
            assert llm_config.model == "gpt-4"
            assert llm_config.api_key == "secret-key"

        finally:
            config_path.unlink()
            # 清理環境變數
            del os.environ["TEST_API_KEY"]
            del os.environ["TEST_MODEL"]

    def test_env_var_with_default(self):
        """測試帶預設值的環境變數"""
        config_data = {
            "models": {
                "test_llm_model": {
                    "type": "openai_chat",
                    "model": "${NONEXISTENT_VAR:gpt-5-mini}",
                    "api_key": "test-key",
                },
                "test_embedding_model": {"type": "bge_m3", "model": "BAAI/bge-m3"},
            },
            "vector_store": {"type": "lancedb", "uri": "./data/lancedb"},
            "model_selection": {
                "default_llm": "test_llm_model",
                "default_embedding": "test_embedding_model",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            config = loader.load_config()

            llm_config = config.get_llm_config("test_llm_model")
            assert llm_config.model == "gpt-5-mini"  # 使用預設值

        finally:
            config_path.unlink()

    def test_missing_env_var_error(self):
        """測試缺少環境變數的錯誤"""
        config_data = {
            "models": {
                "test_model": {
                    "type": "openai_chat",
                    "model": "gpt-4",
                    "api_key": "${MISSING_API_KEY}",  # 沒有預設值
                }
            },
            "vector_store": {"type": "lancedb", "uri": "./data/lancedb"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            with pytest.raises(
                ConfigurationError, match="環境變數 MISSING_API_KEY 未設定"
            ):
                loader.load_config()

        finally:
            config_path.unlink()

    def test_invalid_model_type_error(self):
        """測試無效模型類型錯誤"""
        config_data = {
            "models": {
                "invalid_model": {
                    "type": "invalid_type",  # 無效類型
                    "model": "test-model",
                }
            },
            "vector_store": {"type": "lancedb", "uri": "./data/lancedb"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            with pytest.raises(ConfigurationError, match="未知的模型類型"):
                loader.load_config()

        finally:
            config_path.unlink()

    def test_missing_default_model_error(self):
        """測試缺少預設模型錯誤"""
        config_data = {
            "models": {"some_model": {"type": "openai_chat", "model": "gpt-4"}},
            "vector_store": {"type": "lancedb", "uri": "./data/lancedb"},
            "model_selection": {
                "default_llm": "nonexistent_model",  # 不存在的模型
                "default_embedding": "some_model",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            with pytest.raises(
                ConfigurationError, match="預設 LLM 模型.*未在 models 中定義"
            ):
                loader.load_config()

        finally:
            config_path.unlink()

    def test_create_default_config(self):
        """測試建立預設配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_settings.yaml"

            loader = ConfigLoader()
            created_path = loader.create_default_config(config_path)

            assert created_path == config_path
            assert config_path.exists()

            # 驗證建立的配置可以載入
            loader = ConfigLoader(config_path)
            # 設定必要的環境變數
            os.environ["GRAPHRAG_API_KEY"] = "test-key"

            try:
                config = loader.load_config()
                assert config is not None
                assert len(config.models) >= 2
            finally:
                if "GRAPHRAG_API_KEY" in os.environ:
                    del os.environ["GRAPHRAG_API_KEY"]

    def test_nonexistent_config_file(self):
        """測試不存在的配置檔案"""
        loader = ConfigLoader(Path("nonexistent.yaml"))

        with pytest.raises(ConfigurationError, match="配置檔案不存在"):
            loader.load_config()

    def test_invalid_yaml_syntax(self):
        """測試無效的 YAML 語法"""
        invalid_yaml = """
        models:
          test_model:
            type: openai_chat
            model: gpt-4
            invalid_yaml: [unclosed_bracket
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            with pytest.raises(ConfigurationError, match="YAML 解析錯誤"):
                loader.load_config()

        finally:
            config_path.unlink()
