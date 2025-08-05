"""
OpenAI Embedding 服務實作

使用 OpenAI 的 embedding API 進行文本向量化
支援 text-embedding-ada-002 和 text-embedding-3-small/large
"""

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger

try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai 套件未安裝，OpenAI embedding 服務將不可用")

from .base import (
    EmbeddingComputeError,
    EmbeddingModelType,
    EmbeddingResult,
    EmbeddingService,
    ModelLoadError,
)


@dataclass
class OpenAIModelConfig:
    """OpenAI 模型配置"""

    name: str
    dimensions: int
    max_tokens: int
    cost_per_token: float  # 每千 tokens 的成本（USD）


# 支援的 OpenAI embedding 模型
OPENAI_MODELS = {
    "text-embedding-ada-002": OpenAIModelConfig(
        name="text-embedding-ada-002",
        dimensions=1536,
        max_tokens=8191,
        cost_per_token=0.0001,
    ),
    "text-embedding-3-small": OpenAIModelConfig(
        name="text-embedding-3-small",
        dimensions=1536,
        max_tokens=8191,
        cost_per_token=0.00002,
    ),
    "text-embedding-3-large": OpenAIModelConfig(
        name="text-embedding-3-large",
        dimensions=3072,
        max_tokens=8191,
        cost_per_token=0.00013,
    ),
}


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI Embedding 服務

    使用 OpenAI API 進行文本向量化
    支援批次處理和重試機制
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        organization: Optional[str] = None,
        max_batch_size: int = 100,  # OpenAI 支援較大的批次
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
        dimensions: Optional[int] = None,  # 對於支援的模型可自訂維度
    ):
        """初始化 OpenAI Embedding 服務

        Args:
            model_name: OpenAI 模型名稱
            api_key: OpenAI API 金鑰
            api_base: API 基礎 URL
            organization: OpenAI 組織 ID
            max_batch_size: 最大批次大小
            max_retries: 最大重試次數
            retry_delay: 重試延遲（秒）
            timeout: 請求超時（秒）
            dimensions: 自訂向量維度（僅部分模型支援）
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("請安裝 openai 套件: pip install openai")

        if model_name not in OPENAI_MODELS:
            raise ValueError(f"不支援的 OpenAI 模型: {model_name}")

        self.model_config = OPENAI_MODELS[model_name]

        # 如果指定了自訂維度，檢查模型是否支援
        if dimensions is not None:
            if model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
                self.custom_dimensions = dimensions
            else:
                logger.warning(
                    f"模型 {model_name} 不支援自訂維度，忽略 dimensions 參數"
                )
                self.custom_dimensions = None
        else:
            self.custom_dimensions = None

        super().__init__(
            model_name=model_name,
            model_type=EmbeddingModelType.OPENAI,
            max_batch_size=max_batch_size,
            max_sequence_length=self.model_config.max_tokens,
            device="cloud",  # OpenAI 是雲端服務
        )

        self.api_key = api_key
        self.api_base = api_base
        self.organization = organization
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        self.client: Optional[AsyncOpenAI] = None
        self.total_tokens_used = 0
        self.estimated_cost = 0.0

        logger.info(f"初始化 OpenAI Embedding 服務: {model_name}")

    async def load_model(self) -> None:
        """載入（初始化）OpenAI 客戶端"""
        if self.is_loaded:
            logger.info("OpenAI 客戶端已初始化")
            return

        try:
            logger.info(f"初始化 OpenAI 客戶端: {self.model_name}")

            client_kwargs = {"timeout": self.timeout, "max_retries": self.max_retries}

            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            if self.organization:
                client_kwargs["organization"] = self.organization

            self.client = AsyncOpenAI(**client_kwargs)

            # 測試 API 連接
            await self._test_connection()

            self.is_loaded = True
            logger.info(f"OpenAI 客戶端初始化成功")

        except Exception as e:
            logger.error(f"初始化 OpenAI 客戶端失敗: {e}")
            raise ModelLoadError(f"無法初始化 OpenAI 客戶端: {e}")

    async def _test_connection(self) -> None:
        """測試 API 連接"""
        try:
            # 使用簡單文本測試連接
            test_response = await self.client.embeddings.create(
                model=self.model_name,
                input=["測試連接"],
                dimensions=self.custom_dimensions,
            )

            if not test_response.data:
                raise Exception("API 返回空回應")

            logger.debug("OpenAI API 連接測試成功")

        except Exception as e:
            raise Exception(f"OpenAI API 連接測試失敗: {e}")

    async def unload_model(self) -> None:
        """關閉 OpenAI 客戶端"""
        if not self.is_loaded:
            logger.info("OpenAI 客戶端未初始化")
            return

        try:
            if self.client:
                await self.client.close()
                self.client = None

            self.is_loaded = False
            logger.info("OpenAI 客戶端已關閉")

        except Exception as e:
            logger.error(f"關閉 OpenAI 客戶端時發生錯誤: {e}")

    def get_embedding_dimension(self) -> int:
        """取得向量維度"""
        if self.custom_dimensions:
            return self.custom_dimensions
        return self.model_config.dimensions

    async def embed_texts(
        self, texts: List[str], normalize: bool = True, show_progress: bool = False
    ) -> EmbeddingResult:
        """批次文本向量化

        Args:
            texts: 文本列表
            normalize: 是否正規化向量（OpenAI 向量預設已正規化）
            show_progress: 是否顯示進度

        Returns:
            EmbeddingResult: 向量化結果
        """
        if not self.is_loaded or self.client is None:
            raise EmbeddingComputeError("OpenAI 客戶端未初始化")

        try:
            start_time = time.time()

            # 驗證和清理文本
            valid_texts = self._validate_texts(texts)

            logger.debug(f"開始向量化 {len(valid_texts)} 個文本")

            # 分批處理
            all_embeddings = []
            total_tokens = 0

            batches = self._split_into_batches(valid_texts)

            for i, batch in enumerate(batches):
                if show_progress:
                    logger.info(f"處理批次 {i+1}/{len(batches)}: {len(batch)} 個文本")

                batch_embeddings, batch_tokens = await self._embed_batch_with_retry(
                    batch
                )
                all_embeddings.extend(batch_embeddings)
                total_tokens += batch_tokens

                # 避免 API 限制，批次間加入小延遲
                if i < len(batches) - 1:
                    await asyncio.sleep(0.1)

            # 轉換為 numpy 陣列
            embeddings_array = np.array(all_embeddings)

            # 正規化（雖然 OpenAI 向量已經正規化）
            if normalize:
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                embeddings_array = embeddings_array / np.maximum(norms, 1e-8)

            processing_time = time.time() - start_time

            # 更新使用統計
            self.total_tokens_used += total_tokens
            self.estimated_cost += (
                total_tokens * self.model_config.cost_per_token / 1000
            )

            # 更新效能指標
            self.metrics.update_metrics(processing_time, success=True)

            result = EmbeddingResult(
                embeddings=embeddings_array,
                texts=valid_texts,
                model_name=self.model_name,
                dimensions=self.get_embedding_dimension(),
                processing_time=processing_time,
                metadata={
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": round(
                        total_tokens * self.model_config.cost_per_token / 1000, 6
                    ),
                    "batch_count": len(batches),
                    "custom_dimensions": self.custom_dimensions,
                },
            )

            logger.debug(
                f"向量化完成，耗時: {processing_time:.3f}秒，使用 {total_tokens} tokens"
            )
            return result

        except Exception as e:
            self.metrics.update_metrics(0, success=False)
            logger.error(f"OpenAI 向量化失敗: {e}")
            raise EmbeddingComputeError(f"OpenAI 向量化失敗: {e}")

    async def _embed_batch_with_retry(
        self, batch_texts: List[str]
    ) -> tuple[List[List[float]], int]:
        """使用重試機制處理批次向量化

        Returns:
            tuple: (embeddings, tokens_used)
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                request_kwargs = {"model": self.model_name, "input": batch_texts}

                if self.custom_dimensions:
                    request_kwargs["dimensions"] = self.custom_dimensions

                response = await self.client.embeddings.create(**request_kwargs)

                embeddings = [data.embedding for data in response.data]
                tokens_used = response.usage.total_tokens

                return embeddings, tokens_used

            except Exception as e:
                last_error = e

                # 檢查是否為速率限制錯誤
                if "rate_limit" in str(e).lower():
                    wait_time = self.retry_delay * (2**attempt)  # 指數退避
                    logger.warning(
                        f"遇到速率限制，等待 {wait_time:.1f} 秒後重試 (嘗試 {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                # 其他錯誤也重試，但延遲較短
                if attempt < self.max_retries:
                    wait_time = self.retry_delay
                    logger.warning(f"請求失敗，{wait_time} 秒後重試: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

        raise EmbeddingComputeError(
            f"批次向量化失敗，已重試 {self.max_retries} 次: {last_error}"
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """取得使用統計

        Returns:
            Dict[str, Any]: 使用統計資訊
        """
        return {
            "total_tokens_used": self.total_tokens_used,
            "estimated_cost_usd": round(self.estimated_cost, 6),
            "cost_per_token": self.model_config.cost_per_token,
            "model_name": self.model_name,
        }

    def reset_usage_stats(self) -> None:
        """重置使用統計"""
        self.total_tokens_used = 0
        self.estimated_cost = 0.0
        logger.info("重置 OpenAI 使用統計")

    def get_model_info(self) -> Dict[str, Any]:
        """取得模型詳細資訊"""
        base_info = super().get_model_info()

        openai_specific_info = {
            "api_base": self.api_base,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "custom_dimensions": self.custom_dimensions,
            "max_tokens": self.model_config.max_tokens,
            "cost_per_token": self.model_config.cost_per_token,
            "usage_stats": self.get_usage_stats(),
        }

        return {**base_info, **openai_specific_info}

    async def health_check(self) -> Dict[str, Any]:
        """健康狀態檢查"""
        try:
            base_result = await super().health_check()

            if base_result["status"] == "healthy":
                # 添加 OpenAI 特定的健康資訊
                base_result["usage_stats"] = self.get_usage_stats()
                base_result["api_status"] = "connected"

            return base_result

        except Exception as e:
            return {
                "status": "error",
                "model_name": self.model_name,
                "error": f"健康檢查失敗: {str(e)}",
                "api_status": "disconnected",
            }


def create_openai_service(
    model_name: str = "text-embedding-3-small", api_key: str = None, **kwargs
) -> OpenAIEmbeddingService:
    """建立 OpenAI embedding 服務的便利函數

    Args:
        model_name: OpenAI 模型名稱
        api_key: OpenAI API 金鑰
        **kwargs: 其他參數

    Returns:
        OpenAIEmbeddingService: OpenAI 服務實例
    """
    return OpenAIEmbeddingService(model_name=model_name, api_key=api_key, **kwargs)
