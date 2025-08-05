"""
中文優化 Embedding 服務

專門針對中文文本優化的 embedding 服務，整合多種中文模型
提供中文文本預處理、後處理和品質評估功能
"""

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers 未安裝，中文優化服務將不可用")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..processors.chinese_text_processor import ChineseTextProcessor
from .base import (
    EmbeddingComputeError,
    EmbeddingModelType,
    EmbeddingResult,
    EmbeddingService,
    ModelLoadError,
)
from .bge_m3 import BGEM3EmbeddingService
from .local_models import LOCAL_MODELS, LocalEmbeddingService


@dataclass
class ChineseEmbeddingConfig:
    """中文 Embedding 配置"""

    # 模型配置
    primary_model: str = "BAAI/bge-m3"  # 主要模型
    fallback_models: List[str] = None  # 備用模型列表

    # 中文預處理配置
    enable_preprocessing: bool = True
    remove_stopwords: bool = False  # embedding 通常不移除停用詞
    normalize_text: bool = True
    segment_long_text: bool = True
    max_segment_length: int = 512

    # 後處理配置
    normalize_embeddings: bool = True
    apply_chinese_weighting: bool = True  # 對中文字符給予更高權重

    # 品質評估配置
    enable_quality_check: bool = True
    min_chinese_ratio: float = 0.3  # 最小中文字符比例

    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = ["text2vec-base-chinese", "m3e-base"]


class ChineseOptimizedEmbeddingService(EmbeddingService):
    """中文優化 Embedding 服務

    專門針對中文文本優化的 embedding 服務
    整合多種中文模型並提供智慧切換功能
    """

    def __init__(
        self,
        config: Optional[ChineseEmbeddingConfig] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """初始化中文優化 Embedding 服務

        Args:
            config: 中文 embedding 配置
            device: 計算裝置
            cache_dir: 模型快取目錄
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("請安裝 sentence-transformers 套件")

        self.config = config or ChineseEmbeddingConfig()

        # 自動偵測裝置
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = "cuda"
            elif (
                TORCH_AVAILABLE
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"

        super().__init__(
            model_name=self.config.primary_model,
            model_type=EmbeddingModelType.BGE_M3,
            max_batch_size=32,
            max_sequence_length=self.config.max_segment_length,
            device=device,
        )

        self.cache_dir = cache_dir

        # 初始化服務
        self.primary_service: Optional[EmbeddingService] = None
        self.fallback_services: Dict[str, EmbeddingService] = {}
        self.current_service: Optional[EmbeddingService] = None

        # 初始化中文文本處理器
        if self.config.enable_preprocessing:
            self.text_processor = ChineseTextProcessor(
                chunk_size=self.config.max_segment_length, chunk_overlap=50
            )
        else:
            self.text_processor = None

        # 中文字符權重映射
        self.chinese_char_weights = self._build_chinese_char_weights()

        logger.info(f"初始化中文優化 Embedding 服務: {self.config.primary_model}")

    def _build_chinese_char_weights(self) -> Dict[str, float]:
        """建立中文字符權重映射

        Returns:
            Dict[str, float]: 字符權重映射
        """
        weights = {}

        # 常用中文字符給予標準權重 1.0
        # 生僻字符給予較高權重 1.2（因為資訊量更大）
        # 標點符號給予較低權重 0.5

        # 這裡簡化處理，實際應用中可以基於字頻統計
        common_chars = "的一是在不了有和人這中大為上個國我以要他時來用們生到作地於出就分對成會可主發年動同工也能下過子說產種面而方後多定行學法所民得經十三之進著等部度家電力裡如水化高自二理起小物現實加量都兩體制機當使點從業本去把性好應開它合還因由其些然前外天政四日那社義事平形相全表間樣與關各重新線內數正心反你明看原又麼利比或但質氣第向道命此變條只沒結解問意建月公無系軍很情者最立代想已通並提直題黨程展五果料象員革位入常文總次品式活設及管特件長求老頭基資邊流路級少圖山統接知較將組見計別她手角期根論運農指幾九區強放決西被幹做必戰先回則任取據處隊南給色光門即保治北造百規熱領七海口東導器壓志世金增爭濟階油思術極交受聯什認六共權收證改清己美再採轉更單風切打白教速花帶安場身車例真務具萬每目至達走積示議聲報鬥完類八離華名確才科張信馬節話米整空元況今集溫傳土許步群廣石記需段研界拉林律叫且究觀越織裝影算低持音眾書布复容兒須際商非驗連斷深難近礦千週委素技備半辦青省列習響約支般史感勞便團往酸歷市克何除消構府稱太準精值號率族維劃選標寫存候毛親快效斯院查江型眼王按格養易置派層片始卻專狀育廠京識適屬圓包火住調滿縣局照參紅細引聽該鐵價嚴"

        for char in common_chars:
            weights[char] = 1.0

        # 標點符號權重
        punctuation = "，。！？；：「」『』（）【】《》〈〉、…—－·‧"
        for char in punctuation:
            weights[char] = 0.5

        return weights

    async def load_model(self) -> None:
        """載入模型"""
        if self.is_loaded:
            logger.info("中文優化 Embedding 服務已載入")
            return

        try:
            logger.info("開始載入中文優化 Embedding 服務")

            # 載入主要服務
            await self._load_primary_service()

            # 載入備用服務（異步載入，不阻塞主流程）
            asyncio.create_task(self._load_fallback_services())

            self.current_service = self.primary_service
            self.is_loaded = True

            logger.info("中文優化 Embedding 服務載入完成")

        except Exception as e:
            logger.error(f"載入中文優化 Embedding 服務失敗: {e}")
            raise ModelLoadError(f"無法載入中文優化 Embedding 服務: {e}")

    async def _load_primary_service(self) -> None:
        """載入主要服務"""
        primary_model = self.config.primary_model

        try:
            if "bge-m3" in primary_model.lower():
                self.primary_service = BGEM3EmbeddingService(
                    model_name=primary_model,
                    device=self.device,
                    cache_dir=self.cache_dir,
                    max_sequence_length=self.config.max_segment_length,
                )
            elif primary_model in LOCAL_MODELS:
                self.primary_service = LocalEmbeddingService(
                    model_name=primary_model,
                    device=self.device,
                    cache_dir=self.cache_dir,
                    max_sequence_length=self.config.max_segment_length,
                )
            else:
                # 嘗試作為通用模型載入
                self.primary_service = BGEM3EmbeddingService(
                    model_name=primary_model,
                    device=self.device,
                    cache_dir=self.cache_dir,
                    max_sequence_length=self.config.max_segment_length,
                )

            await self.primary_service.load_model()
            logger.info(f"主要服務載入成功: {primary_model}")

        except Exception as e:
            logger.error(f"載入主要服務失敗: {e}")
            raise

    async def _load_fallback_services(self) -> None:
        """載入備用服務"""
        for model_name in self.config.fallback_models:
            try:
                if model_name in LOCAL_MODELS:
                    service = LocalEmbeddingService(
                        model_name=model_name,
                        device=self.device,
                        cache_dir=self.cache_dir,
                        max_sequence_length=self.config.max_segment_length,
                    )
                else:
                    service = BGEM3EmbeddingService(
                        model_name=model_name,
                        device=self.device,
                        cache_dir=self.cache_dir,
                        max_sequence_length=self.config.max_segment_length,
                    )

                await service.load_model()
                self.fallback_services[model_name] = service
                logger.info(f"備用服務載入成功: {model_name}")

            except Exception as e:
                logger.warning(f"載入備用服務 {model_name} 失敗: {e}")

    async def unload_model(self) -> None:
        """卸載模型"""
        if not self.is_loaded:
            return

        try:
            # 卸載主要服務
            if self.primary_service:
                await self.primary_service.unload_model()
                self.primary_service = None

            # 卸載備用服務
            for service in self.fallback_services.values():
                try:
                    await service.unload_model()
                except Exception as e:
                    logger.warning(f"卸載備用服務失敗: {e}")

            self.fallback_services.clear()
            self.current_service = None
            self.is_loaded = False

            logger.info("中文優化 Embedding 服務已卸載")

        except Exception as e:
            logger.error(f"卸載中文優化 Embedding 服務失敗: {e}")

    def get_embedding_dimension(self) -> int:
        """取得向量維度"""
        if self.current_service:
            return self.current_service.get_embedding_dimension()

        # 預設維度（BGE-M3）
        return 1024

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """中文文本預處理

        Args:
            texts: 原始文本列表

        Returns:
            List[str]: 預處理後的文本列表
        """
        if not self.config.enable_preprocessing or not self.text_processor:
            return texts

        processed_texts = []

        for text in texts:
            if not text or not text.strip():
                processed_texts.append("")
                continue

            # 文本清理
            if self.config.normalize_text:
                cleaned_text = self.text_processor.clean_text(text)
            else:
                cleaned_text = text

            # 檢查中文比例
            if self.config.enable_quality_check:
                stats = self.text_processor.get_text_statistics(cleaned_text)
                chinese_ratio = (
                    stats["chinese_char_count"] / stats["char_count"]
                    if stats["char_count"] > 0
                    else 0
                )

                if chinese_ratio < self.config.min_chinese_ratio:
                    logger.warning(
                        f"文本中文比例過低: {chinese_ratio:.2f}, 文本: {text[:50]}..."
                    )

            # 長文本分段
            if (
                self.config.segment_long_text
                and len(cleaned_text) > self.config.max_segment_length
            ):
                # 簡單分段策略：按句子分割
                sentences = self.text_processor._split_into_sentences(cleaned_text)

                # 重新組合句子，確保不超過長度限制
                segments = []
                current_segment = ""

                for sentence in sentences:
                    if (
                        len(current_segment) + len(sentence)
                        <= self.config.max_segment_length
                    ):
                        current_segment += sentence + " "
                    else:
                        if current_segment.strip():
                            segments.append(current_segment.strip())
                        current_segment = sentence + " "

                if current_segment.strip():
                    segments.append(current_segment.strip())

                # 對於長文本，使用第一個段落作為代表
                # 實際應用中可能需要更複雜的策略
                if segments:
                    processed_texts.append(segments[0])
                else:
                    processed_texts.append(
                        cleaned_text[: self.config.max_segment_length]
                    )
            else:
                processed_texts.append(cleaned_text)

        return processed_texts

    def _postprocess_embeddings(
        self, embeddings: np.ndarray, texts: List[str]
    ) -> np.ndarray:
        """中文 embedding 後處理

        Args:
            embeddings: 原始 embedding 矩陣
            texts: 對應的文本列表

        Returns:
            np.ndarray: 後處理後的 embedding 矩陣
        """
        if not self.config.apply_chinese_weighting:
            return embeddings

        # 計算每個文本的中文字符權重
        processed_embeddings = embeddings.copy()

        for i, text in enumerate(texts):
            if not text:
                continue

            # 計算中文字符權重
            chinese_weight = self._calculate_chinese_weight(text)

            # 應用權重調整
            if chinese_weight > 1.0:
                processed_embeddings[i] *= chinese_weight

        # 重新正規化
        if self.config.normalize_embeddings:
            norms = np.linalg.norm(processed_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            processed_embeddings = processed_embeddings / norms

        return processed_embeddings

    def _calculate_chinese_weight(self, text: str) -> float:
        """計算文本的中文字符權重

        Args:
            text: 輸入文本

        Returns:
            float: 中文權重係數
        """
        if not text:
            return 1.0

        # 統計中文字符
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_count = len(chinese_chars)
        total_chars = len(text)

        if total_chars == 0:
            return 1.0

        # 中文比例
        chinese_ratio = chinese_count / total_chars

        # 權重計算：中文比例越高，權重越大
        # 基礎權重 1.0，最大權重 1.3
        weight = 1.0 + (chinese_ratio * 0.3)

        return min(weight, 1.3)

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
        if not self.is_loaded or not self.current_service:
            raise EmbeddingComputeError("中文優化 Embedding 服務未載入")

        try:
            start_time = time.time()

            # 驗證輸入
            if not texts:
                raise ValueError("文本列表不能為空")

            # 中文預處理
            processed_texts = self._preprocess_texts(texts)

            # 使用當前服務進行向量化
            try:
                result = await self.current_service.embed_texts(
                    processed_texts,
                    normalize=False,  # 我們會在後處理中進行正規化
                    show_progress=show_progress,
                )

                # 中文後處理
                processed_embeddings = self._postprocess_embeddings(
                    result.embeddings, processed_texts
                )

                # 更新結果
                result.embeddings = processed_embeddings
                result.texts = processed_texts
                result.model_name = (
                    f"chinese_optimized_{self.current_service.model_name}"
                )

                processing_time = time.time() - start_time
                result.processing_time = processing_time

                # 添加中文優化元數據
                result.metadata.update(
                    {
                        "chinese_optimized": True,
                        "preprocessing_enabled": self.config.enable_preprocessing,
                        "chinese_weighting_applied": self.config.apply_chinese_weighting,
                        "primary_model": self.config.primary_model,
                        "current_service": self.current_service.model_name,
                    }
                )

                # 更新效能指標
                self.metrics.update_metrics(processing_time, success=True)

                return result

            except Exception as e:
                logger.error(f"主要服務向量化失敗: {e}")

                # 嘗試備用服務
                if self.fallback_services:
                    for (
                        fallback_name,
                        fallback_service,
                    ) in self.fallback_services.items():
                        try:
                            logger.info(f"嘗試備用服務: {fallback_name}")

                            result = await fallback_service.embed_texts(
                                processed_texts,
                                normalize=False,
                                show_progress=show_progress,
                            )

                            # 中文後處理
                            processed_embeddings = self._postprocess_embeddings(
                                result.embeddings, processed_texts
                            )

                            result.embeddings = processed_embeddings
                            result.texts = processed_texts
                            result.model_name = f"chinese_optimized_{fallback_name}"

                            processing_time = time.time() - start_time
                            result.processing_time = processing_time

                            result.metadata.update(
                                {
                                    "chinese_optimized": True,
                                    "fallback_used": True,
                                    "fallback_service": fallback_name,
                                }
                            )

                            # 切換到成功的備用服務
                            self.current_service = fallback_service
                            logger.info(f"切換到備用服務: {fallback_name}")

                            self.metrics.update_metrics(processing_time, success=True)
                            return result

                        except Exception as fallback_error:
                            logger.warning(
                                f"備用服務 {fallback_name} 也失敗: {fallback_error}"
                            )
                            continue

                # 所有服務都失敗
                self.metrics.update_metrics(0, success=False)
                raise EmbeddingComputeError(f"所有 embedding 服務都失敗: {e}")

        except Exception as e:
            self.metrics.update_metrics(0, success=False)
            logger.error(f"中文優化向量化失敗: {e}")
            raise EmbeddingComputeError(f"中文優化向量化失敗: {e}")

    async def evaluate_chinese_quality(
        self, texts: List[str], embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """評估中文 embedding 品質

        Args:
            texts: 文本列表
            embeddings: embedding 矩陣，如果為 None 則重新計算

        Returns:
            Dict[str, Any]: 品質評估結果
        """
        if not texts:
            return {"error": "文本列表為空"}

        try:
            # 計算 embeddings（如果未提供）
            if embeddings is None:
                result = await self.embed_texts(texts)
                embeddings = result.embeddings

            # 文本品質評估
            text_quality_scores = []
            if self.text_processor:
                for text in texts:
                    quality = self.text_processor.evaluate_text_quality(text)
                    text_quality_scores.append(quality["overall_score"])

            # Embedding 品質指標
            embedding_metrics = self._calculate_embedding_metrics(embeddings)

            # 中文特定指標
            chinese_metrics = self._calculate_chinese_metrics(texts, embeddings)

            # 綜合評估
            overall_quality = self._calculate_overall_quality(
                text_quality_scores, embedding_metrics, chinese_metrics
            )

            return {
                "overall_quality": overall_quality,
                "text_quality": {
                    "average_score": (
                        np.mean(text_quality_scores) if text_quality_scores else 0.0
                    ),
                    "scores": text_quality_scores,
                },
                "embedding_metrics": embedding_metrics,
                "chinese_metrics": chinese_metrics,
                "evaluation_time": time.time(),
            }

        except Exception as e:
            logger.error(f"中文品質評估失敗: {e}")
            return {"error": str(e)}

    def _calculate_embedding_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """計算 embedding 品質指標

        Args:
            embeddings: embedding 矩陣

        Returns:
            Dict[str, float]: 品質指標
        """
        if embeddings.size == 0:
            return {}

        # 向量範數統計
        norms = np.linalg.norm(embeddings, axis=1)

        # 向量相似度統計（使用正規化的餘弦相似度）
        if len(embeddings) > 1:
            # 正規化 embeddings
            normalized_embeddings = embeddings / (norms[:, np.newaxis] + 1e-8)
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            # 排除對角線（自相似度）
            mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
            similarities = similarity_matrix[mask]
        else:
            similarities = np.array([])

        # 向量維度利用率（非零維度比例）
        non_zero_dims = np.count_nonzero(embeddings, axis=0)
        dim_utilization = np.mean(non_zero_dims > 0) if embeddings.shape[1] > 0 else 0.0

        return {
            "norm_mean": float(np.mean(norms)),
            "norm_std": float(np.std(norms)),
            "norm_min": float(np.min(norms)),
            "norm_max": float(np.max(norms)),
            "similarity_mean": (
                float(np.mean(similarities)) if len(similarities) > 0 else 0.0
            ),
            "similarity_std": (
                float(np.std(similarities)) if len(similarities) > 0 else 0.0
            ),
            "dimension_utilization": float(dim_utilization),
            "embedding_dimension": embeddings.shape[1],
        }

    def _calculate_chinese_metrics(
        self, texts: List[str], embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """計算中文特定指標

        Args:
            texts: 文本列表
            embeddings: embedding 矩陣

        Returns:
            Dict[str, Any]: 中文特定指標
        """
        if not self.text_processor:
            return {}

        chinese_ratios = []
        text_lengths = []
        word_counts = []

        for text in texts:
            if not text:
                continue

            stats = self.text_processor.get_text_statistics(text)

            # 中文字符比例
            chinese_ratio = (
                stats["chinese_char_count"] / stats["char_count"]
                if stats["char_count"] > 0
                else 0
            )
            chinese_ratios.append(chinese_ratio)

            text_lengths.append(stats["char_count"])
            word_counts.append(stats["word_count"])

        # 計算中文文本間的語義相似度分佈
        chinese_texts = [
            text for text in texts if text and self._is_chinese_dominant(text)
        ]
        chinese_similarity_coherence = 0.0

        if len(chinese_texts) > 1:
            # 簡化的語義連貫性評估
            # 實際應用中可以使用更複雜的方法
            chinese_indices = [
                i
                for i, text in enumerate(texts)
                if text and self._is_chinese_dominant(text)
            ]
            if len(chinese_indices) > 1:
                chinese_embeddings = embeddings[chinese_indices]
                chinese_sim_matrix = np.dot(chinese_embeddings, chinese_embeddings.T)
                mask = ~np.eye(chinese_sim_matrix.shape[0], dtype=bool)
                chinese_similarities = chinese_sim_matrix[mask]
                chinese_similarity_coherence = float(np.mean(chinese_similarities))

        return {
            "chinese_ratio_mean": (
                float(np.mean(chinese_ratios)) if chinese_ratios else 0.0
            ),
            "chinese_ratio_std": (
                float(np.std(chinese_ratios)) if chinese_ratios else 0.0
            ),
            "chinese_text_count": len(chinese_texts),
            "total_text_count": len(texts),
            "avg_text_length": float(np.mean(text_lengths)) if text_lengths else 0.0,
            "avg_word_count": float(np.mean(word_counts)) if word_counts else 0.0,
            "chinese_similarity_coherence": chinese_similarity_coherence,
        }

    def _is_chinese_dominant(self, text: str) -> bool:
        """判斷文本是否以中文為主

        Args:
            text: 輸入文本

        Returns:
            bool: 是否以中文為主
        """
        if not text:
            return False

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_ratio = len(chinese_chars) / len(text)

        return chinese_ratio >= self.config.min_chinese_ratio

    def _calculate_overall_quality(
        self,
        text_quality_scores: List[float],
        embedding_metrics: Dict[str, float],
        chinese_metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """計算綜合品質分數

        Args:
            text_quality_scores: 文本品質分數列表
            embedding_metrics: embedding 指標
            chinese_metrics: 中文特定指標

        Returns:
            Dict[str, float]: 綜合品質分數
        """
        scores = {}

        # 文本品質分數 (0-1)
        if text_quality_scores:
            scores["text_quality"] = max(0.0, min(1.0, np.mean(text_quality_scores)))
        else:
            scores["text_quality"] = 0.5

        # Embedding 品質分數 (0-1)
        if embedding_metrics:
            # 基於向量範數的穩定性
            norm_stability = 1.0 - min(embedding_metrics.get("norm_std", 0.5), 0.5)

            # 基於維度利用率
            dim_utilization = embedding_metrics.get("dimension_utilization", 0.5)

            # 基於相似度分佈的合理性（餘弦相似度應該在 -1 到 1 之間）
            sim_mean = embedding_metrics.get("similarity_mean", 0.0)
            # 確保相似度在合理範圍內，並轉換為品質分數
            sim_mean = max(-1.0, min(1.0, sim_mean))
            sim_reasonableness = 1.0 - abs(sim_mean)  # 相似度不應該太高或太低

            scores["embedding_quality"] = max(
                0.0,
                min(1.0, (norm_stability + dim_utilization + sim_reasonableness) / 3),
            )
        else:
            scores["embedding_quality"] = 0.5

        # 中文特定品質分數 (0-1)
        if chinese_metrics:
            chinese_ratio = chinese_metrics.get("chinese_ratio_mean", 0.0)
            chinese_coherence = chinese_metrics.get("chinese_similarity_coherence", 0.0)

            # 中文比例分數
            chinese_ratio_score = min(
                chinese_ratio / max(self.config.min_chinese_ratio, 0.1), 1.0
            )

            # 中文語義連貫性分數（假設 coherence 在 -1 到 1 之間）
            chinese_coherence = max(-1.0, min(1.0, chinese_coherence))
            chinese_coherence_score = (chinese_coherence + 1.0) / 2.0  # 轉換到 0-1 範圍

            scores["chinese_quality"] = max(
                0.0, min(1.0, (chinese_ratio_score + chinese_coherence_score) / 2)
            )
        else:
            scores["chinese_quality"] = 0.5

        # 綜合分數（加權平均）
        weights = {
            "text_quality": 0.3,
            "embedding_quality": 0.4,
            "chinese_quality": 0.3,
        }

        overall_score = sum(scores[key] * weights[key] for key in scores.keys())
        scores["overall"] = max(0.0, min(1.0, overall_score))

        return scores

    def get_model_info(self) -> Dict[str, Any]:
        """取得模型詳細資訊"""
        base_info = super().get_model_info()

        chinese_info = {
            "chinese_optimized": True,
            "primary_model": self.config.primary_model,
            "fallback_models": self.config.fallback_models,
            "current_service": (
                self.current_service.model_name if self.current_service else None
            ),
            "preprocessing_enabled": self.config.enable_preprocessing,
            "chinese_weighting_enabled": self.config.apply_chinese_weighting,
            "quality_check_enabled": self.config.enable_quality_check,
            "min_chinese_ratio": self.config.min_chinese_ratio,
            "max_segment_length": self.config.max_segment_length,
            "available_fallback_services": list(self.fallback_services.keys()),
        }

        return {**base_info, **chinese_info}


def create_chinese_optimized_service(
    primary_model: str = "BAAI/bge-m3",
    fallback_models: List[str] = None,
    device: str = None,
    **kwargs,
) -> ChineseOptimizedEmbeddingService:
    """建立中文優化 embedding 服務的便利函數

    Args:
        primary_model: 主要模型名稱
        fallback_models: 備用模型列表
        device: 計算裝置
        **kwargs: 其他配置參數

    Returns:
        ChineseOptimizedEmbeddingService: 中文優化服務實例
    """
    config = ChineseEmbeddingConfig(
        primary_model=primary_model, fallback_models=fallback_models, **kwargs
    )

    return ChineseOptimizedEmbeddingService(config=config, device=device)
