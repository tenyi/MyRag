"""
BGE-M3 Embedding 服務實作

使用 BGE-M3 模型進行中文文本向量化
"""

import asyncio
from typing import List, Optional, Dict, Any, Union
import time
import numpy as np
from pathlib import Path

from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers 未安裝，BGE-M3 服務將不可用")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 未安裝，將使用 CPU 模式")

from .base import (
    EmbeddingService, 
    EmbeddingResult, 
    EmbeddingModelType,
    ModelLoadError,
    EmbeddingComputeError
)


class BGEM3EmbeddingService(EmbeddingService):
    """BGE-M3 Embedding 服務
    
    專門針對中文文本優化的 embedding 模型服務
    支援多語言和長文本處理
    """
    
    DEFAULT_MODEL_NAME = "BAAI/bge-m3"
    
    def __init__(
        self,
        model_name: str = None,
        max_batch_size: int = 32,
        max_sequence_length: int = 8192,  # BGE-M3 支援長序列
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        cache_dir: Optional[str] = None
    ):
        """初始化 BGE-M3 Embedding 服務
        
        Args:
            model_name: 模型名稱，預設為 BAAI/bge-m3
            max_batch_size: 最大批次大小
            max_sequence_length: 最大序列長度
            device: 計算裝置
            trust_remote_code: 是否信任遠程程式碼
            normalize_embeddings: 是否正規化 embedding
            use_fp16: 是否使用半精度浮點數
            cache_dir: 模型快取目錄
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("請安裝 sentence-transformers 套件: pip install sentence-transformers")
        
        model_name = model_name or self.DEFAULT_MODEL_NAME
        
        # 自動偵測裝置
        if device is None or device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = "cuda"
            elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
            else:
                device = "cpu"
        
        super().__init__(
            model_name=model_name,
            model_type=EmbeddingModelType.BGE_M3,
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
            device=device
        )
        
        self.trust_remote_code = trust_remote_code
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16 and device != "cpu"  # CPU 不支援 fp16
        self.cache_dir = cache_dir
        self.model: Optional[SentenceTransformer] = None
        
        logger.info(f"初始化 BGE-M3 服務: {model_name}, 裝置: {device}")
    
    async def load_model(self) -> None:
        """載入 BGE-M3 模型"""
        if self.is_loaded:
            logger.info("BGE-M3 模型已載入")
            return
        
        try:
            logger.info(f"開始載入 BGE-M3 模型: {self.model_name}")
            start_time = time.time()
            
            # 在執行緒池中載入模型以避免阻塞
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                self._load_model_sync
            )
            
            load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"BGE-M3 模型載入成功，耗時: {load_time:.2f}秒")
            logger.info(f"模型維度: {self.get_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"載入 BGE-M3 模型失敗: {e}")
            raise ModelLoadError(f"無法載入 BGE-M3 模型: {e}")
    
    def _load_model_sync(self) -> SentenceTransformer:
        """同步載入模型（在執行緒池中執行）"""
        model_kwargs = {
            'device': self.device,
            'trust_remote_code': self.trust_remote_code
        }
        
        if self.cache_dir:
            model_kwargs['cache_folder'] = self.cache_dir
        
        model = SentenceTransformer(self.model_name, **model_kwargs)
        
        # 設定模型參數
        if hasattr(model, 'max_seq_length'):
            model.max_seq_length = self.max_sequence_length
        
        # 啟用 fp16（如果支援）
        if self.use_fp16 and hasattr(model, 'half'):
            model = model.half()
        
        return model
    
    async def unload_model(self) -> None:
        """卸載模型，釋放記憶體"""
        if not self.is_loaded:
            logger.info("BGE-M3 模型未載入")
            return
        
        try:
            # 清理 GPU 記憶體
            if self.model is not None:
                if hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
                del self.model
                self.model = None
            
            # 清理 CUDA 快取
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info("BGE-M3 模型已卸載")
            
        except Exception as e:
            logger.error(f"卸載 BGE-M3 模型時發生錯誤: {e}")
    
    def get_embedding_dimension(self) -> int:
        """取得向量維度"""
        if not self.is_loaded or self.model is None:
            return 1024  # BGE-M3 預設維度
        
        return self.model.get_sentence_embedding_dimension()
    
    async def embed_texts(
        self, 
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False
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
            raise EmbeddingComputeError("BGE-M3 模型未載入")
        
        try:
            start_time = time.time()
            
            # 驗證和清理文本
            valid_texts = self._validate_texts(texts)
            
            logger.debug(f"開始向量化 {len(valid_texts)} 個文本")
            
            # 在執行緒池中執行向量化以避免阻塞
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._embed_texts_sync,
                valid_texts,
                normalize,
                show_progress
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
                    'device': self.device,
                    'batch_size': len(valid_texts),
                    'use_fp16': self.use_fp16,
                    'normalize': normalize
                }
            )
            
            logger.debug(f"向量化完成，耗時: {processing_time:.3f}秒")
            return result
            
        except Exception as e:
            self.metrics.update_metrics(0, success=False)
            logger.error(f"BGE-M3 向量化失敗: {e}")
            raise EmbeddingComputeError(f"BGE-M3 向量化失敗: {e}")
    
    def _embed_texts_sync(
        self, 
        texts: List[str], 
        normalize: bool, 
        show_progress: bool
    ) -> np.ndarray:
        """同步向量化（在執行緒池中執行）"""
        encode_kwargs = {
            'normalize_embeddings': normalize or self.normalize_embeddings,
            'show_progress_bar': show_progress,
            'batch_size': self.max_batch_size,
            'convert_to_numpy': True
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
        
        bge_specific_info = {
            'trust_remote_code': self.trust_remote_code,
            'normalize_embeddings': self.normalize_embeddings,
            'use_fp16': self.use_fp16,
            'cache_dir': self.cache_dir,
            'supports_long_sequence': True,
            'multilingual': True,
            'optimized_for_chinese': True
        }
        
        return {**base_info, **bge_specific_info}
    
    async def compute_dense_embeddings(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """計算密集向量（BGE-M3 的密集表示）
        
        Args:
            texts: 文本列表
            normalize: 是否正規化
            
        Returns:
            np.ndarray: 密集向量
        """
        return (await self.embed_texts(texts, normalize=normalize)).embeddings
    
    async def compute_sparse_embeddings(
        self,
        texts: List[str]
    ) -> List[Dict[int, float]]:
        """計算稀疏向量（BGE-M3 的稀疏表示）
        
        BGE-M3 支援稀疏向量表示，適合用於關鍵詞匹配
        
        Args:
            texts: 文本列表
            
        Returns:
            List[Dict[int, float]]: 稀疏向量列表
        """
        if not self.is_loaded or self.model is None:
            raise EmbeddingComputeError("BGE-M3 模型未載入")
        
        try:
            # 檢查模型是否支援稀疏表示
            if not hasattr(self.model, 'encode') or not hasattr(self.model[0], 'sparse'):
                logger.warning("當前 BGE-M3 模型不支援稀疏向量，返回空結果")
                return [{} for _ in texts]
            
            logger.debug(f"計算 {len(texts)} 個文本的稀疏向量")
            
            # 這是一個簡化的實作，實際的 BGE-M3 稀疏向量需要特殊的 tokenizer
            # 這裡僅作為介面預留
            return [{} for _ in texts]
            
        except Exception as e:
            logger.error(f"計算稀疏向量失敗: {e}")
            return [{} for _ in texts]
    
    async def compute_colbert_embeddings(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """計算 ColBERT 風格的 token 級別向量
        
        BGE-M3 支援 ColBERT 風格的 late interaction
        
        Args:
            texts: 文本列表
            
        Returns:
            List[np.ndarray]: token 級別向量列表
        """
        if not self.is_loaded or self.model is None:
            raise EmbeddingComputeError("BGE-M3 模型未載入")
        
        try:
            # 檢查模型是否支援 ColBERT 表示
            if not hasattr(self.model, 'encode') or not hasattr(self.model[0], 'colbert'):
                logger.warning("當前 BGE-M3 模型不支援 ColBERT 向量，返回空結果")
                return [np.array([]) for _ in texts]
            
            logger.debug(f"計算 {len(texts)} 個文本的 ColBERT 向量")
            
            # 這是一個簡化的實作，實際的 BGE-M3 ColBERT 向量需要特殊處理
            # 這裡僅作為介面預留
            return [np.array([]) for _ in texts]
            
        except Exception as e:
            logger.error(f"計算 ColBERT 向量失敗: {e}")
            return [np.array([]) for _ in texts]


def create_bge_m3_service(
    model_name: str = None,
    device: str = None,
    **kwargs
) -> BGEM3EmbeddingService:
    """建立 BGE-M3 embedding 服務的便利函數
    
    Args:
        model_name: 模型名稱
        device: 計算裝置
        **kwargs: 其他參數
        
    Returns:
        BGEM3EmbeddingService: BGE-M3 服務實例
    """
    return BGEM3EmbeddingService(
        model_name=model_name,
        device=device,
        **kwargs
    )