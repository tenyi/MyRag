"""
中文優化 Embedding 服務測試
"""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from chinese_graphrag.embeddings.base import EmbeddingResult
from chinese_graphrag.embeddings.chinese_optimized import (
    ChineseEmbeddingConfig,
    ChineseOptimizedEmbeddingService,
    create_chinese_optimized_service,
)


class TestChineseEmbeddingConfig:
    """測試中文 Embedding 配置"""

    def test_default_config(self):
        """測試預設配置"""
        config = ChineseEmbeddingConfig()

        assert config.primary_model == "BAAI/bge-m3"
        assert config.enable_preprocessing is True
        assert config.normalize_embeddings is True
        assert config.apply_chinese_weighting is True
        assert config.min_chinese_ratio == 0.3
        assert config.max_segment_length == 512
        assert "text2vec-base-chinese" in config.fallback_models
        assert "m3e-base" in config.fallback_models

    def test_custom_config(self):
        """測試自訂配置"""
        config = ChineseEmbeddingConfig(
            primary_model="custom-model",
            fallback_models=["model1", "model2"],
            enable_preprocessing=False,
            min_chinese_ratio=0.5,
            max_segment_length=1024,
        )

        assert config.primary_model == "custom-model"
        assert config.fallback_models == ["model1", "model2"]
        assert config.enable_preprocessing is False
        assert config.min_chinese_ratio == 0.5
        assert config.max_segment_length == 1024


class TestChineseOptimizedEmbeddingService:
    """測試中文優化 Embedding 服務"""

    @pytest.fixture
    def mock_config(self):
        """模擬配置"""
        return ChineseEmbeddingConfig(
            primary_model="test-model",
            fallback_models=["fallback-model"],
            enable_preprocessing=True,
            apply_chinese_weighting=True,
        )

    @pytest.fixture
    def mock_service(self, mock_config):
        """模擬服務"""
        with patch(
            "chinese_graphrag.embeddings.chinese_optimized.SENTENCE_TRANSFORMERS_AVAILABLE",
            True,
        ):
            service = ChineseOptimizedEmbeddingService(config=mock_config, device="cpu")
            return service

    def test_init(self, mock_service, mock_config):
        """測試初始化"""
        assert mock_service.config == mock_config
        assert mock_service.model_name == "test-model"
        assert mock_service.device == "cpu"
        assert mock_service.text_processor is not None
        assert mock_service.chinese_char_weights is not None

    def test_build_chinese_char_weights(self, mock_service):
        """測試中文字符權重建立"""
        weights = mock_service.chinese_char_weights

        # 檢查常用字符權重
        assert weights.get("的", 0) == 1.0
        assert weights.get("是", 0) == 1.0

        # 檢查標點符號權重
        assert weights.get("，", 0) == 0.5
        assert weights.get("。", 0) == 0.5

    def test_calculate_chinese_weight(self, mock_service):
        """測試中文權重計算"""
        # 純中文文本
        chinese_text = "這是一個中文測試文本"
        weight = mock_service._calculate_chinese_weight(chinese_text)
        assert weight > 1.0  # 中文文本應該有更高權重

        # 英文文本
        english_text = "This is an English text"
        weight = mock_service._calculate_chinese_weight(english_text)
        assert weight == 1.0  # 非中文文本保持基礎權重

        # 混合文本
        mixed_text = "This is 中文 mixed text"
        weight = mock_service._calculate_chinese_weight(mixed_text)
        assert 1.0 < weight < 1.3  # 混合文本權重介於兩者之間

        # 空文本
        empty_text = ""
        weight = mock_service._calculate_chinese_weight(empty_text)
        assert weight == 1.0

    def test_preprocess_texts(self, mock_service):
        """測試文本預處理"""
        texts = [
            "這是一個中文測試文本。",
            "This is English text.",
            "",
            "   空白文本   ",
            "很長的文本" * 100,  # 超長文本
        ]

        processed = mock_service._preprocess_texts(texts)

        assert len(processed) == len(texts)
        assert processed[0]  # 中文文本應該被保留
        assert processed[1]  # 英文文本也應該被保留
        assert processed[2] == ""  # 空文本保持為空
        assert processed[3].strip()  # 空白文本應該被清理
        assert (
            len(processed[4]) <= mock_service.config.max_segment_length
        )  # 長文本應該被截斷

    def test_postprocess_embeddings(self, mock_service):
        """測試 embedding 後處理"""
        # 模擬 embedding 矩陣
        embeddings = np.random.rand(3, 768).astype(np.float32)
        texts = ["這是中文文本", "This is English", "中英混合 mixed text"]

        processed = mock_service._postprocess_embeddings(embeddings, texts)

        assert processed.shape == embeddings.shape
        assert isinstance(processed, np.ndarray)

        # 檢查正規化
        if mock_service.config.normalize_embeddings:
            norms = np.linalg.norm(processed, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    @pytest.mark.asyncio
    async def test_embed_texts_success(self, mock_service):
        """測試成功的文本向量化"""
        # 模擬主要服務
        mock_primary_service = AsyncMock()
        mock_embedding_result = EmbeddingResult(
            embeddings=np.random.rand(2, 768).astype(np.float32),
            texts=["測試文本1", "測試文本2"],
            model_name="test-model",
            dimensions=768,
            processing_time=0.1,
        )
        mock_primary_service.embed_texts.return_value = mock_embedding_result
        mock_primary_service.model_name = "test-model"

        mock_service.primary_service = mock_primary_service
        mock_service.current_service = mock_primary_service
        mock_service.is_loaded = True

        texts = ["這是測試文本1", "這是測試文本2"]
        result = await mock_service.embed_texts(texts)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert result.dimensions == 768
        assert "chinese_optimized" in result.model_name
        assert result.metadata["chinese_optimized"] is True

        # 驗證主要服務被調用
        mock_primary_service.embed_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_fallback(self, mock_service):
        """測試備用服務降級"""
        # 模擬主要服務失敗
        mock_primary_service = AsyncMock()
        mock_primary_service.embed_texts.side_effect = Exception(
            "Primary service failed"
        )
        mock_primary_service.model_name = "primary-model"

        # 模擬備用服務成功
        mock_fallback_service = AsyncMock()
        mock_embedding_result = EmbeddingResult(
            embeddings=np.random.rand(1, 768).astype(np.float32),
            texts=["測試文本"],
            model_name="fallback-model",
            dimensions=768,
            processing_time=0.1,
        )
        mock_fallback_service.embed_texts.return_value = mock_embedding_result
        mock_fallback_service.model_name = "fallback-model"

        mock_service.primary_service = mock_primary_service
        mock_service.current_service = mock_primary_service
        mock_service.fallback_services = {"fallback-model": mock_fallback_service}
        mock_service.is_loaded = True

        texts = ["測試文本"]
        result = await mock_service.embed_texts(texts)

        assert isinstance(result, EmbeddingResult)
        assert "chinese_optimized_fallback-model" in result.model_name
        assert result.metadata["fallback_used"] is True
        assert result.metadata["fallback_service"] == "fallback-model"

        # 驗證服務切換
        assert mock_service.current_service == mock_fallback_service

    @pytest.mark.asyncio
    async def test_embed_texts_all_fail(self, mock_service):
        """測試所有服務都失敗的情況"""
        # 模擬所有服務都失敗
        mock_primary_service = AsyncMock()
        mock_primary_service.embed_texts.side_effect = Exception("Primary failed")

        mock_fallback_service = AsyncMock()
        mock_fallback_service.embed_texts.side_effect = Exception("Fallback failed")

        mock_service.primary_service = mock_primary_service
        mock_service.current_service = mock_primary_service
        mock_service.fallback_services = {"fallback": mock_fallback_service}
        mock_service.is_loaded = True

        texts = ["測試文本"]

        with pytest.raises(Exception) as exc_info:
            await mock_service.embed_texts(texts)

        assert "所有 embedding 服務都失敗" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluate_chinese_quality(self, mock_service):
        """測試中文品質評估"""
        # 模擬 embed_texts 方法
        mock_embedding_result = EmbeddingResult(
            embeddings=np.random.rand(3, 768).astype(np.float32),
            texts=["中文文本1", "中文文本2", "English text"],
            model_name="test-model",
            dimensions=768,
            processing_time=0.1,
        )

        mock_service.embed_texts = AsyncMock(return_value=mock_embedding_result)

        texts = ["中文文本1", "中文文本2", "English text"]
        result = await mock_service.evaluate_chinese_quality(texts)

        assert "overall_quality" in result
        assert "text_quality" in result
        assert "embedding_metrics" in result
        assert "chinese_metrics" in result

        # 檢查品質分數
        overall_quality = result["overall_quality"]
        assert "overall" in overall_quality
        assert 0 <= overall_quality["overall"] <= 1

    def test_is_chinese_dominant(self, mock_service):
        """測試中文主導判斷"""
        # 純中文文本
        chinese_text = "這是一個純中文文本"
        assert mock_service._is_chinese_dominant(chinese_text) is True

        # 英文文本
        english_text = "This is pure English text"
        assert mock_service._is_chinese_dominant(english_text) is False

        # 中文主導的混合文本
        mixed_chinese = "這是中文為主的文本 with some English"
        # 結果取決於中文比例是否超過閾值
        result = mock_service._is_chinese_dominant(mixed_chinese)
        assert isinstance(result, bool)

        # 空文本
        empty_text = ""
        assert mock_service._is_chinese_dominant(empty_text) is False

    def test_get_model_info(self, mock_service):
        """測試模型資訊獲取"""
        info = mock_service.get_model_info()

        assert info["chinese_optimized"] is True
        assert info["primary_model"] == "test-model"
        assert "fallback_models" in info
        assert "preprocessing_enabled" in info
        assert "chinese_weighting_enabled" in info
        assert "quality_check_enabled" in info
        assert "min_chinese_ratio" in info


class TestCreateChineseOptimizedService:
    """測試便利函數"""

    def test_create_chinese_optimized_service(self):
        """測試創建中文優化服務"""
        with patch(
            "chinese_graphrag.embeddings.chinese_optimized.SENTENCE_TRANSFORMERS_AVAILABLE",
            True,
        ):
            service = create_chinese_optimized_service(
                primary_model="test-model",
                fallback_models=["fallback1", "fallback2"],
                device="cpu",
            )

            assert isinstance(service, ChineseOptimizedEmbeddingService)
            assert service.config.primary_model == "test-model"
            assert service.config.fallback_models == ["fallback1", "fallback2"]
            assert service.device == "cpu"

    def test_create_with_default_params(self):
        """測試使用預設參數創建服務"""
        with patch(
            "chinese_graphrag.embeddings.chinese_optimized.SENTENCE_TRANSFORMERS_AVAILABLE",
            True,
        ):
            service = create_chinese_optimized_service()

            assert isinstance(service, ChineseOptimizedEmbeddingService)
            assert service.config.primary_model == "BAAI/bge-m3"
            assert "text2vec-base-chinese" in service.config.fallback_models


@pytest.mark.integration
class TestChineseOptimizedIntegration:
    """整合測試（需要實際模型）"""

    @pytest.mark.skip(reason="需要實際模型檔案")
    @pytest.mark.asyncio
    async def test_real_model_integration(self):
        """測試真實模型整合"""
        # 這個測試需要實際的模型檔案，在 CI 環境中跳過
        service = create_chinese_optimized_service(device="cpu")

        try:
            await service.load_model()

            texts = [
                "這是一個中文測試文本",
                "人工智慧技術發展迅速",
                "自然語言處理很重要",
            ]

            result = await service.embed_texts(texts)

            assert len(result.embeddings) == 3
            assert result.dimensions > 0
            assert result.processing_time > 0

            # 測試品質評估
            quality = await service.evaluate_chinese_quality(texts)
            assert "overall_quality" in quality

        finally:
            await service.unload_model()
