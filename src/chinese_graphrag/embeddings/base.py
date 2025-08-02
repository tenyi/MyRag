"""
Embedding 服務基礎類別和介面定義
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum

import numpy as np
from loguru import logger


class EmbeddingModelType(Enum):
    """Embedding 模型類型枚舉"""
    BGE_M3 = "bge-m3"
    OPENAI = "openai"
    TEXT2VEC = "text2vec"
    M3E = "m3e"
    LOCAL = "local"


@dataclass
class EmbeddingResult:
    """Embedding 結果資料類別"""
    embeddings: np.ndarray  # 向量結果
    texts: List[str]  # 原始文本
    model_name: str  # 使用的模型名稱
    dimensions: int  # 向量維度
    processing_time: float  # 處理時間（秒）
    metadata: Dict[str, Any] = None  # 額外元數據
    
    def __post_init__(self):
        """後處理初始化"""
        if self.metadata is None:
            self.metadata = {}
        
        # 驗證向量維度
        if len(self.embeddings.shape) == 2:
            actual_dims = self.embeddings.shape[1]
        else:
            actual_dims = self.embeddings.shape[0]
            
        if actual_dims != self.dimensions:
            logger.warning(f"向量維度不匹配: 預期 {self.dimensions}, 實際 {actual_dims}")
            self.dimensions = actual_dims


@dataclass
class ModelMetrics:
    """模型效能指標"""
    model_name: str
    total_requests: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    
    def update_metrics(self, processing_time: float, success: bool = True, memory_usage: float = 0.0):
        """更新效能指標"""
        self.total_requests += 1
        
        if success:
            self.total_processing_time += processing_time
            self.average_processing_time = self.total_processing_time / (self.total_requests - self.error_count)
        else:
            self.error_count += 1
            
        self.success_rate = (self.total_requests - self.error_count) / self.total_requests
        
        if memory_usage > 0:
            self.memory_usage_mb = memory_usage


class EmbeddingService(ABC):
    """Embedding 服務抽象基類
    
    定義所有 embedding 服務實作必須遵循的介面
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: EmbeddingModelType,
        max_batch_size: int = 32,
        max_sequence_length: int = 512,
        device: Optional[str] = None
    ):
        """初始化 Embedding 服務
        
        Args:
            model_name: 模型名稱
            model_type: 模型類型
            max_batch_size: 最大批次大小
            max_sequence_length: 最大序列長度
            device: 裝置（'cpu', 'cuda', 'mps' 等）
        """
        self.model_name = model_name
        self.model_type = model_type
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.is_loaded = False
        self.metrics = ModelMetrics(model_name)
        
        # 安全地獲取類型字符串
        if hasattr(model_type, 'value'):
            type_str = model_type.value
        else:
            type_str = str(model_type)
        logger.info(f"初始化 Embedding 服務: {model_name} ({type_str})")
    
    @abstractmethod
    async def load_model(self) -> None:
        """載入模型"""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """卸載模型，釋放記憶體"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """取得向量維度"""
        pass
    
    @abstractmethod
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
        pass
    
    async def embed_single_text(
        self, 
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """單一文本向量化
        
        Args:
            text: 單一文本
            normalize: 是否正規化向量
            
        Returns:
            np.ndarray: 向量結果
        """
        result = await self.embed_texts([text], normalize=normalize)
        return result.embeddings[0]
    
    async def compute_similarity(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]],
        method: str = "cosine"
    ) -> Union[float, np.ndarray]:
        """計算文本相似度
        
        Args:
            texts1: 第一組文本
            texts2: 第二組文本  
            method: 相似度計算方法 ('cosine', 'dot', 'euclidean')
            
        Returns:
            相似度分數或分數矩陣
        """
        # 確保輸入為列表
        if isinstance(texts1, str):
            texts1 = [texts1]
        if isinstance(texts2, str):
            texts2 = [texts2]
        
        # 向量化
        embeddings1 = await self.embed_texts(texts1, normalize=True)
        embeddings2 = await self.embed_texts(texts2, normalize=True)
        
        # 計算相似度
        if method == "cosine":
            similarity = np.dot(embeddings1.embeddings, embeddings2.embeddings.T)
        elif method == "dot":
            similarity = np.dot(embeddings1.embeddings, embeddings2.embeddings.T)
        elif method == "euclidean":
            # 歐幾里得距離，轉換為相似度（距離越小相似度越高）
            from scipy.spatial.distance import cdist
            distances = cdist(embeddings1.embeddings, embeddings2.embeddings, metric='euclidean')
            similarity = 1 / (1 + distances)  # 轉換為相似度分數
        else:
            raise ValueError(f"不支援的相似度計算方法: {method}")
        
        # 如果只有單一文本對，返回標量
        if len(texts1) == 1 and len(texts2) == 1:
            return float(similarity[0, 0])
        
        return similarity
    
    def get_metrics(self) -> ModelMetrics:
        """取得模型效能指標"""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """重置效能指標"""
        self.metrics = ModelMetrics(self.model_name)
        logger.info(f"重置模型 {self.model_name} 的效能指標")
    
    def get_model_info(self) -> Dict[str, Any]:
        """取得模型基本資訊"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value if hasattr(self.model_type, 'value') else str(self.model_type),
            "max_batch_size": self.max_batch_size,
            "max_sequence_length": self.max_sequence_length,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "embedding_dimension": self.get_embedding_dimension() if self.is_loaded else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康狀態檢查"""
        try:
            if not self.is_loaded:
                return {
                    "status": "not_loaded",
                    "model_name": self.model_name,
                    "error": "模型未載入"
                }
            
            # 測試向量化一個簡單文本
            test_text = "健康檢查測試文本"
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await self.embed_single_text(test_text)
                processing_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "status": "healthy",
                    "model_name": self.model_name,
                    "embedding_dimension": len(result),
                    "test_processing_time": round(processing_time, 4),
                    "metrics": {
                        "total_requests": self.metrics.total_requests,
                        "average_processing_time": round(self.metrics.average_processing_time, 4),
                        "success_rate": round(self.metrics.success_rate, 3),
                        "error_count": self.metrics.error_count
                    }
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "model_name": self.model_name,
                    "error": str(e),
                    "processing_time": round(asyncio.get_event_loop().time() - start_time, 4)
                }
                
        except Exception as e:
            return {
                "status": "error",
                "model_name": self.model_name,
                "error": f"健康檢查失敗: {str(e)}"
            }
    
    def _validate_texts(self, texts: List[str]) -> List[str]:
        """驗證和清理輸入文本"""
        if not texts:
            raise ValueError("文本列表不能為空")
        
        # 過濾空文本
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not valid_texts:
            raise ValueError("沒有有效的文本內容")
        
        # 檢查文本長度
        if self.max_sequence_length:
            truncated_texts = []
            for text in valid_texts:
                if len(text) > self.max_sequence_length:
                    truncated_text = text[:self.max_sequence_length]
                    truncated_texts.append(truncated_text)
                    logger.warning(f"文本超過最大長度限制，已截斷: {len(text)} -> {len(truncated_text)}")
                else:
                    truncated_texts.append(text)
            valid_texts = truncated_texts
        
        return valid_texts
    
    def _split_into_batches(self, texts: List[str]) -> List[List[str]]:
        """將文本列表分割成批次"""
        batches = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            batches.append(batch)
        return batches
    
    async def __aenter__(self):
        """異步上下文管理器入口"""
        if not self.is_loaded:
            await self.load_model()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器出口"""
        if self.is_loaded:
            await self.unload_model()


class EmbeddingServiceError(Exception):
    """Embedding 服務相關異常"""
    pass


class ModelLoadError(EmbeddingServiceError):
    """模型載入異常"""
    pass


class EmbeddingComputeError(EmbeddingServiceError):
    """向量計算異常"""
    pass