"""
配置模型測試

測試所有配置相關的 Pydantic 模型
"""

import pytest
from pydantic import ValidationError

from chinese_graphrag.config.models import (
    DeviceType,
    EmbeddingConfig,
    EmbeddingType,
    GraphRAGConfig,
    LLMConfig,
    LLMType,
    VectorStoreConfig,
    VectorStoreType,
)


class TestLLMConfig:
    """LLM 配置測試"""

    def test_valid_llm_config(self):
        """測試有效的 LLM 配置"""
        config = LLMConfig(
            type=LLMType.OPENAI_CHAT,
            model="gpt-4",
            api_key="test-key",
            temperature=0.5
        )
        
        assert config.type == LLMType.OPENAI_CHAT
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000  # 預設值

    def test_invalid_temperature(self):
        """測試無效的溫度參數"""
        with pytest.raises(ValidationError):
            LLMConfig(
                type=LLMType.OPENAI_CHAT,
                model="gpt-4",
                temperature=3.0  # 超出範圍
            )

    def test_azure_openai_config(self):
        """測試 Azure OpenAI 配置"""
        config = LLMConfig(
            type=LLMType.AZURE_OPENAI_CHAT,
            model="gpt-4",
            api_base="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4-deployment"
        )
        
        assert config.type == LLMType.AZURE_OPENAI_CHAT
        assert config.api_base == "https://test.openai.azure.com"
        assert config.deployment_name == "gpt-4-deployment"


class TestEmbeddingConfig:
    """Embedding 配置測試"""

    def test_openai_embedding_config(self):
        """測試 OpenAI Embedding 配置"""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI_EMBEDDING,
            model="text-embedding-3-small",
            api_key="test-key"
        )
        
        assert config.type == EmbeddingType.OPENAI_EMBEDDING
        assert config.model == "text-embedding-3-small"
        assert config.device == DeviceType.AUTO  # 預設值

    def test_bge_m3_config(self):
        """測試 BGE-M3 配置"""
        config = EmbeddingConfig(
            type=EmbeddingType.BGE_M3,
            model="BAAI/bge-m3",
            device=DeviceType.CUDA,
            batch_size=64
        )
        
        assert config.type == EmbeddingType.BGE_M3
        assert config.device == DeviceType.CUDA
        assert config.batch_size == 64

    def test_invalid_batch_size(self):
        """測試無效的批次大小"""
        with pytest.raises(ValidationError):
            EmbeddingConfig(
                type=EmbeddingType.BGE_M3,
                model="BAAI/bge-m3",
                batch_size=0  # 無效值
            )


class TestVectorStoreConfig:
    """向量資料庫配置測試"""

    def test_lancedb_config(self):
        """測試 LanceDB 配置"""
        config = VectorStoreConfig(
            type=VectorStoreType.LANCEDB,
            uri="./data/lancedb",
            container_name="test_container"
        )
        
        assert config.type == VectorStoreType.LANCEDB
        assert config.uri == "./data/lancedb"
        assert config.container_name == "test_container"
        assert config.metric == "cosine"  # 預設值


class TestGraphRAGConfig:
    """GraphRAG 主配置測試"""

    def test_minimal_config(self):
        """測試最小配置"""
        config = GraphRAGConfig(
            models={
                "default_llm": LLMConfig(
                    type=LLMType.OPENAI_CHAT,
                    model="gpt-4"
                ),
                "default_embedding": EmbeddingConfig(
                    type=EmbeddingType.BGE_M3,
                    model="BAAI/bge-m3"
                )
            },
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LANCEDB,
                uri="./data/lancedb"
            )
        )
        
        assert len(config.models) == 2
        assert config.vector_store.type == VectorStoreType.LANCEDB

    def test_get_llm_config(self):
        """測試取得 LLM 配置"""
        llm_config = LLMConfig(
            type=LLMType.OPENAI_CHAT,
            model="gpt-4"
        )
        
        config = GraphRAGConfig(
            models={"test_llm": llm_config},
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LANCEDB,
                uri="./data/lancedb"
            )
        )
        
        retrieved_config = config.get_llm_config("test_llm")
        assert retrieved_config == llm_config
        
        # 測試不存在的模型
        assert config.get_llm_config("nonexistent") is None

    def test_get_embedding_config(self):
        """測試取得 Embedding 配置"""
        embedding_config = EmbeddingConfig(
            type=EmbeddingType.BGE_M3,
            model="BAAI/bge-m3"
        )
        
        config = GraphRAGConfig(
            models={"test_embedding": embedding_config},
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LANCEDB,
                uri="./data/lancedb"
            )
        )
        
        retrieved_config = config.get_embedding_config("test_embedding")
        assert retrieved_config == embedding_config
        
        # 測試不存在的模型
        assert config.get_embedding_config("nonexistent") is None

    def test_default_model_configs(self):
        """測試預設模型配置"""
        llm_config = LLMConfig(
            type=LLMType.OPENAI_CHAT,
            model="gpt-4"
        )
        embedding_config = EmbeddingConfig(
            type=EmbeddingType.BGE_M3,
            model="BAAI/bge-m3"
        )
        
        config = GraphRAGConfig(
            models={
                "my_llm": llm_config,
                "my_embedding": embedding_config
            },
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LANCEDB,
                uri="./data/lancedb"
            )
        )
        
        # 設定預設模型
        config.model_selection.default_llm = "my_llm"
        config.model_selection.default_embedding = "my_embedding"
        
        assert config.get_default_llm_config() == llm_config
        assert config.get_default_embedding_config() == embedding_config