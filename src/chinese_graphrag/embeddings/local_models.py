"""
本地中文 Embedding 模型服務

支援 text2vec 和 m3e 等中文優化的本地模型
提供統一的介面和高效的本地推理
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers 未安裝，本地模型服務將不可用")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 未安裝，將使用 CPU 模式")

from .base import (
    EmbeddingComputeError,
    EmbeddingModelType,
    EmbeddingResult,
    EmbeddingService,
    ModelLoadError,
)


@dataclass
class LocalModelConfig:
    """本地模型配置"""

    name: str
    model_path: str
    dimensions: int
    max_seq_length: int
    optimized_for_chinese: bool
    description: str


# 支援的本地中文模型
LOCAL_MODELS = {
    # text2vec 系列
    "text2vec-base-chinese": LocalModelConfig(
        name="text2vec-base-chinese",
        model_path="shibing624/text2vec-base-chinese",
        dimensions=768,
        max_seq_length=512,
        optimized_for_chinese=True,
        description="中文文本向量化模型，基於 BERT",
    ),
    "text2vec-large-chinese": LocalModelConfig(
        name="text2vec-large-chinese",
        model_path="shibing624/text2vec-large-chinese",
        dimensions=1024,
        max_seq_length=512,
        optimized_for_chinese=True,
        description="大型中文文本向量化模型",
    ),
    "text2vec-base-multilingual": LocalModelConfig(
        name="text2vec-base-multilingual",
        model_path="shibing624/text2vec-base-multilingual",
        dimensions=768,
        max_seq_length=512,
        optimized_for_chinese=True,
        description="多語言文本向量化模型，支援中英文",
    ),
    # m3e 系列
    "m3e-base": LocalModelConfig(
        name="m3e-base",
        model_path="moka-ai/m3e-base",
        dimensions=768,
        max_seq_length=512,
        optimized_for_chinese=True,
        description="M3E 中文文本向量化模型，基礎版",
    ),
    "m3e-large": LocalModelConfig(
        name="m3e-large",
        model_path="moka-ai/m3e-large",
        dimensions=1024,
        max_seq_length=512,
        optimized_for_chinese=True,
        description="M3E 中文文本向量化模型，大型版",
    ),
    "m3e-small": LocalModelConfig(
        name="m3e-small",
        model_path="moka-ai/m3e-small",
        dimensions=512,
        max_seq_length=512,
        optimized_for_chinese=True,
        description="M3E 中文文本向量化模型，小型版",
    ),
    # 其他中文模型
    "chinese-roberta-wwm-ext": LocalModelConfig(
        name="chinese-roberta-wwm-ext",
        model_path="hfl/chinese-roberta-wwm-ext",
        dimensions=768,
        max_seq_length=512,
        optimized_for_chinese=True,
        description="中文 RoBERTa 模型，全詞遮蔽",
    ),
    "chinese-bert-wwm-ext": LocalModelConfig(
        name="chinese-bert-wwm-ext",
        model_path="hfl/chinese-bert-wwm-ext",
        dimensions=768,
        max_seq_length=512,
        optimized_for_chinese=True,
        description="中文 BERT 模型，全詞遮蔽",
    ),
}


class LocalEmbeddingService(EmbeddingService):
    """本地 Embedding 模型服務

    支援多種中文優化的本地模型
    提供高效的本地推理和靈活的配置
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        max_batch_size: int = 32,
        max_sequence_length: Optional[int] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        """初始化本地 Embedding 服務

        Args:
            model_name: 模型名稱
            model_path: 模型路徑，如果為 None 則使用預設配置
            max_batch_size: 最大批次大小
            max_sequence_length: 最大序列長度
            device: 計算裝置
            trust_remote_code: 是否信任遠程程式碼
            normalize_embeddings: 是否正規化 embedding
            use_fp16: 是否使用半精度浮點數
            cache_dir: 模型快取目錄
            local_files_only: 是否只使用本地檔案
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "請安裝 sentence-transformers 套件: pip install sentence-transformers"
            )

        # 取得模型配置
        if model_name in LOCAL_MODELS:
            self.model_config = LOCAL_MODELS[model_name]
            actual_model_path = model_path or self.model_config.model_path
            actual_max_seq_length = (
                max_sequence_length or self.model_config.max_seq_length
            )
        else:
            # 自訂模型
            if not model_path:
                raise ValueError(f"未知模型 {model_name}，請提供 model_path")

            self.model_config = LocalModelConfig(
                name=model_name,
                model_path=model_path,
                dimensions=768,  # 預設值，載入後會更新
                max_seq_length=max_sequence_length or 512,
                optimized_for_chinese=False,
                description="自訂本地模型",
            )
            actual_model_path = model_path
            actual_max_seq_length = max_sequence_length or 512

        # 自動偵測裝置
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = "cuda"
            elif (
                TORCH_AVAILABLE
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"  # Apple Silicon
            else:
                device = "cpu"

        # 決定模型類型
        if "text2vec" in model_name.lower():
            model_type = EmbeddingModelType.TEXT2VEC
        elif "m3e" in model_name.lower():
            model_type = EmbeddingModelType.M3E
        else:
            model_type = EmbeddingModelType.LOCAL

        super().__init__(
            model_name=model_name,
            model_type=model_type,
            max_batch_size=max_batch_size,
            max_sequence_length=actual_max_seq_length,
            device=device,
        )

        self.model_path = actual_model_path
        self.trust_remote_code = trust_remote_code
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16 and device != "cpu"  # CPU 不支援 fp16
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.model: Optional[SentenceTransformer] = None

        logger.info(
            f"初始化本地 Embedding 服務: {model_name}, 路徑: {actual_model_path}, 裝置: {device}"
        )

    async def load_model(self) -> None:
        """載入本地模型"""
        if self.is_loaded:
            logger.info("本地模型已載入")
            return

        try:
            logger.info(f"開始載入本地模型: {self.model_name} from {self.model_path}")
            start_time = time.time()

            # 在執行緒池中載入模型以避免阻塞
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, self._load_model_sync)

            # 更新實際的模型維度
            actual_dimensions = self.model.get_sentence_embedding_dimension()
            self.model_config.dimensions = actual_dimensions

            load_time = time.time() - start_time
            self.is_loaded = True

            logger.info(f"本地模型載入成功，耗時: {load_time:.2f}秒")
            logger.info(f"模型維度: {actual_dimensions}")

        except Exception as e:
            logger.error(f"載入本地模型失敗: {e}")
            raise ModelLoadError(f"無法載入本地模型: {e}")

    def _load_model_sync(self) -> SentenceTransformer:
        """同步載入模型（在執行緒池中執行）"""
        model_kwargs = {
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.cache_dir:
            model_kwargs["cache_folder"] = self.cache_dir

        if self.local_files_only:
            model_kwargs["local_files_only"] = True

        try:
            model = SentenceTransformer(self.model_path, **model_kwargs)
        except Exception as e:
            # 如果從 HuggingFace Hub 載入失敗，嘗試本地路徑
            if Path(self.model_path).exists():
                logger.info(f"嘗試從本地路徑載入: {self.model_path}")
                model = SentenceTransformer(
                    str(Path(self.model_path).resolve()), **model_kwargs
                )
            else:
                raise e

        # 設定模型參數
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = self.max_sequence_length

        # 啟用 fp16（如果支援）
        if self.use_fp16 and hasattr(model, "half"):
            model = model.half()

        return model

    async def unload_model(self) -> None:
        """卸載模型，釋放記憶體"""
        if not self.is_loaded:
            logger.info("本地模型未載入")
            return

        try:
            # 清理 GPU 記憶體
            if self.model is not None:
                if hasattr(self.model, "to"):
                    self.model = self.model.to("cpu")
                del self.model
                self.model = None

            # 清理 CUDA 快取
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            logger.info("本地模型已卸載")

        except Exception as e:
            logger.error(f"卸載本地模型時發生錯誤: {e}")

    def get_embedding_dimension(self) -> int:
        """取得向量維度"""
        if not self.is_loaded or self.model is None:
            return self.model_config.dimensions

        return self.model.get_sentence_embedding_dimension()

    async def embed_texts(
        self, texts: List[str], normalize: bool = True, show_progress: bool = False
    ) -> EmbeddingResult:
        """批次文本向量化

        Args:
            texts: 文本列表
            normalize: 是否正規化向量
            show_progress: 是否顯示進度

        Returns:
            EmbeddingResult: 向量化結果
        """
        if not self.is_loaded or self.model is None:
            raise EmbeddingComputeError("本地模型未載入")

        try:
            start_time = time.time()

            # 驗證和清理文本
            valid_texts = self._validate_texts(texts)

            logger.debug(f"開始向量化 {len(valid_texts)} 個文本")

            # 在執行緒池中執行向量化以避免阻塞
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self._embed_texts_sync, valid_texts, normalize, show_progress
            )

            processing_time = time.time() - start_time

            # 更新效能指標
            self.metrics.update_metrics(processing_time, success=True)

            result = EmbeddingResult(
                embeddings=embeddings,
                texts=valid_texts,
                model_name=self.model_name,
                dimensions=self.get_embedding_dimension(),
                processing_time=processing_time,
                metadata={
                    "device": self.device,
                    "batch_size": len(valid_texts),
                    "use_fp16": self.use_fp16,
                    "normalize": normalize,
                    "model_path": self.model_path,
                    "model_type": (
                        self.model_type.value
                        if hasattr(self.model_type, "value")
                        else str(self.model_type)
                    ),
                    "optimized_for_chinese": self.model_config.optimized_for_chinese,
                },
            )

            logger.debug(f"向量化完成，耗時: {processing_time:.3f}秒")
            return result

        except Exception as e:
            self.metrics.update_metrics(0, success=False)
            logger.error(f"本地模型向量化失敗: {e}")
            raise EmbeddingComputeError(f"本地模型向量化失敗: {e}")

    def _embed_texts_sync(
        self, texts: List[str], normalize: bool, show_progress: bool
    ) -> np.ndarray:
        """同步向量化（在執行緒池中執行）"""
        encode_kwargs = {
            "normalize_embeddings": normalize or self.normalize_embeddings,
            "show_progress_bar": show_progress,
            "batch_size": self.max_batch_size,
            "convert_to_numpy": True,
        }

        embeddings = self.model.encode(texts, **encode_kwargs)

        # 確保返回正確的 numpy 陣列格式
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # 確保是二維陣列
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """取得模型詳細資訊"""
        base_info = super().get_model_info()

        local_specific_info = {
            "model_path": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "normalize_embeddings": self.normalize_embeddings,
            "use_fp16": self.use_fp16,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "model_description": self.model_config.description,
            "optimized_for_chinese": self.model_config.optimized_for_chinese,
            "actual_dimensions": self.model_config.dimensions,
        }

        return {**base_info, **local_specific_info}

    @classmethod
    def list_available_models(cls) -> List[Dict[str, Any]]:
        """列出所有可用的本地模型

        Returns:
            List[Dict[str, Any]]: 模型資訊列表
        """
        models_info = []
        for name, config in LOCAL_MODELS.items():
            models_info.append(
                {
                    "name": name,
                    "model_path": config.model_path,
                    "dimensions": config.dimensions,
                    "max_seq_length": config.max_seq_length,
                    "optimized_for_chinese": config.optimized_for_chinese,
                    "description": config.description,
                }
            )

        return models_info

    def supports_chinese(self) -> bool:
        """檢查是否針對中文優化"""
        return self.model_config.optimized_for_chinese


def create_text2vec_service(
    model_name: str = "text2vec-base-chinese", device: str = None, **kwargs
) -> LocalEmbeddingService:
    """建立 text2vec embedding 服務的便利函數

    Args:
        model_name: text2vec 模型名稱
        device: 計算裝置
        **kwargs: 其他參數

    Returns:
        LocalEmbeddingService: text2vec 服務實例
    """
    if model_name not in LOCAL_MODELS or "text2vec" not in model_name:
        raise ValueError(f"不支援的 text2vec 模型: {model_name}")

    return LocalEmbeddingService(model_name=model_name, device=device, **kwargs)


def create_m3e_service(
    model_name: str = "m3e-base", device: str = None, **kwargs
) -> LocalEmbeddingService:
    """建立 m3e embedding 服務的便利函數

    Args:
        model_name: m3e 模型名稱
        device: 計算裝置
        **kwargs: 其他參數

    Returns:
        LocalEmbeddingService: m3e 服務實例
    """
    if model_name not in LOCAL_MODELS or "m3e" not in model_name:
        raise ValueError(f"不支援的 m3e 模型: {model_name}")

    return LocalEmbeddingService(model_name=model_name, device=device, **kwargs)


def create_local_service(
    model_name: str, model_path: str = None, **kwargs
) -> LocalEmbeddingService:
    """建立自訂本地 embedding 服務的便利函數

    Args:
        model_name: 模型名稱
        model_path: 模型路徑
        **kwargs: 其他參數

    Returns:
        LocalEmbeddingService: 本地服務實例
    """
    return LocalEmbeddingService(model_name=model_name, model_path=model_path, **kwargs)
