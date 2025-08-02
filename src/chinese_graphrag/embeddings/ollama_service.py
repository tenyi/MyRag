"""
Ollama Embedding 服務實現。

提供透過 Ollama API 進行文本向量化的功能，支持多種 Ollama 模型。
"""

import time
import asyncio
import aiohttp
import numpy as np
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from .base import EmbeddingService, EmbeddingResult, ModelMetrics, EmbeddingModelType
from ..monitoring import get_logger

# 檢查 Ollama 相關依賴
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class OllamaModelConfig:
    """Ollama 模型配置"""
    name: str
    base_url: str = "http://localhost:11434"
    timeout: int = 60
    max_retries: int = 3
    dimensions: Optional[int] = None  # 自動檢測


class OllamaEmbeddingService(EmbeddingService):
    """
    Ollama Embedding 服務
    
    透過 Ollama API 提供文本向量化功能，支持：
    - 多種 Ollama 模型
    - 批次處理
    - 自動維度檢測
    - 錯誤重試機制
    """
    
    def __init__(
        self,
        model_name: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        max_retries: int = 3,
        dimensions: Optional[int] = None,
        device: str = "auto",
        **kwargs
    ):
        """
        初始化 Ollama Embedding 服務
        
        Args:
            model_name: Ollama 模型名稱
            base_url: Ollama API 基礎 URL
            timeout: 請求超時時間（秒）
            max_retries: 最大重試次數
            dimensions: 向量維度（自動檢測）
            device: 計算設備（對 Ollama 無效，保留兼容性）
        """
        super().__init__()
        
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.dimensions = dimensions
        self.device = device
        
        self.model_type = EmbeddingModelType.REMOTE_API
        self.is_loaded = False
        
        # API 端點
        self.embed_url = f"{self.base_url}/api/embeddings"
        self.models_url = f"{self.base_url}/api/tags"
        
        # 初始化指標
        self.metrics = ModelMetrics(
            model_name=self.model_name,
            model_type=self.model_type
        )
        
        logger.info(f"初始化 Ollama Embedding 服務: {model_name} @ {base_url}")
    
    async def load_model(self) -> None:
        """
        加載模型（檢查連接性和模型可用性）
        """
        if self.is_loaded:
            logger.info(f"模型 {self.model_name} 已經加載")
            return
        
        try:
            # 檢查 Ollama 服務是否可用
            await self._check_service_availability()
            
            # 檢查模型是否可用
            await self._check_model_availability()
            
            # 檢測向量維度
            if not self.dimensions:
                await self._detect_dimensions()
            
            self.is_loaded = True
            logger.info(f"成功加載 Ollama 模型: {self.model_name} (維度: {self.dimensions})")
            
        except Exception as e:
            logger.error(f"加載 Ollama 模型失敗: {e}")
            raise
    
    async def unload_model(self) -> None:
        """
        卸載模型（Ollama 無需卸載）
        """
        self.is_loaded = False
        logger.info(f"標記 Ollama 模型為未加載: {self.model_name}")
    
    async def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        批次文本向量化
        
        Args:
            texts: 文本列表
            normalize: 是否正規化向量
            show_progress: 是否顯示進度
            
        Returns:
            EmbeddingResult: 向量化結果
        """
        if not self.is_loaded:
            await self.load_model()
        
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                texts=[],
                model_name=self.model_name,
                dimensions=self.dimensions or 0,
                processing_time=0.0
            )
        
        start_time = time.time()
        
        try:
            # 批次處理文本
            all_embeddings = []
            
            if show_progress:
                from rich.progress import track
                texts_iter = track(texts, description="Ollama 向量化")
            else:
                texts_iter = texts
            
            # 單個處理（Ollama API 通常一次處理一個文本）
            for text in texts_iter:
                embedding = await self._embed_single_text(text)
                all_embeddings.append(embedding)
            
            # 轉換為 numpy 陣列
            embeddings_array = np.array(all_embeddings)
            
            # 正規化
            if normalize and len(embeddings_array) > 0:
                embeddings_array = self._normalize_embeddings(embeddings_array)
            
            processing_time = time.time() - start_time
            
            # 更新指標
            self.metrics.total_requests += 1
            self.metrics.total_processing_time += processing_time
            self.metrics.total_texts_processed += len(texts)
            
            result = EmbeddingResult(
                embeddings=embeddings_array,
                texts=texts,
                model_name=self.model_name,
                dimensions=embeddings_array.shape[1] if len(embeddings_array) > 0 else 0,
                processing_time=processing_time,
                metadata={
                    "provider": "ollama",
                    "base_url": self.base_url,
                    "normalized": normalize
                }
            )
            
            logger.debug(f"Ollama 向量化完成: {len(texts)} 個文本，耗時 {processing_time:.3f} 秒")
            return result
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Ollama 向量化失敗: {e}")
            raise
    
    async def _embed_single_text(self, text: str) -> np.ndarray:
        """
        單一文本向量化
        
        Args:
            text: 文本內容
            
        Returns:
            np.ndarray: 向量結果
        """
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(self.embed_url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            embedding = result.get("embedding", [])
                            
                            if not embedding:
                                raise ValueError("Ollama API 返回空向量")
                            
                            return np.array(embedding, dtype=np.float32)
                        else:
                            error_text = await response.text()
                            raise Exception(f"Ollama API 錯誤 {response.status}: {error_text}")
                            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(f"Ollama API 調用失敗 (嘗試 {attempt + 1}/{self.max_retries}): {e}，{wait_time} 秒後重試")
                await asyncio.sleep(wait_time)
    
    async def _check_service_availability(self) -> None:
        """檢查 Ollama 服務可用性"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.base_url}/api/version") as response:
                    if response.status != 200:
                        raise Exception(f"Ollama 服務不可用，狀態碼: {response.status}")
                    
                    result = await response.json()
                    logger.info(f"Ollama 服務可用，版本: {result.get('version', 'unknown')}")
                    
        except Exception as e:
            raise Exception(f"無法連接到 Ollama 服務 ({self.base_url}): {e}")
    
    async def _check_model_availability(self) -> None:
        """檢查模型可用性"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(self.models_url) as response:
                    if response.status == 200:
                        result = await response.json()
                        models = result.get("models", [])
                        model_names = [model["name"] for model in models]
                        
                        if self.model_name not in model_names:
                            logger.warning(f"模型 {self.model_name} 不在已安裝列表中: {model_names}")
                            # 嘗試拉取模型
                            await self._pull_model()
                        else:
                            logger.info(f"模型 {self.model_name} 已可用")
                    else:
                        logger.warning(f"無法獲取模型列表，狀態碼: {response.status}")
                        
        except Exception as e:
            logger.warning(f"檢查模型可用性失敗: {e}")
    
    async def _pull_model(self) -> None:
        """拉取模型"""
        logger.info(f"正在拉取模型: {self.model_name}")
        
        payload = {"name": self.model_name}
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                async with session.post(f"{self.base_url}/api/pull", json=payload) as response:
                    if response.status == 200:
                        logger.info(f"成功拉取模型: {self.model_name}")
                    else:
                        error_text = await response.text()
                        logger.error(f"拉取模型失敗: {error_text}")
                        
        except Exception as e:
            logger.error(f"拉取模型時發生錯誤: {e}")
    
    async def _detect_dimensions(self) -> None:
        """檢測向量維度"""
        try:
            test_embedding = await self._embed_single_text("test")
            self.dimensions = len(test_embedding)
            logger.info(f"檢測到向量維度: {self.dimensions}")
            
        except Exception as e:
            logger.error(f"檢測向量維度失敗: {e}")
            self.dimensions = 1024  # 默認維度
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """正規化向量"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        return embeddings / norms
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型資訊"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value if hasattr(self.model_type, 'value') else str(self.model_type),
            "provider": "ollama",
            "base_url": self.base_url,
            "dimensions": self.dimensions,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "supports_batch": False,  # Ollama 通常單個處理
            "max_batch_size": 1
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康檢查"""
        try:
            start_time = time.time()
            
            # 檢查服務可用性
            await self._check_service_availability()
            
            # 簡單的向量化測試
            if self.is_loaded:
                test_result = await self._embed_single_text("health check")
                response_time = time.time() - start_time
                
                return {
                    "status": "healthy",
                    "model_name": self.model_name,
                    "response_time": round(response_time, 3),
                    "dimensions": len(test_result),
                    "base_url": self.base_url
                }
            else:
                return {
                    "status": "not_loaded",
                    "model_name": self.model_name,
                    "base_url": self.base_url
                }
                
        except Exception as e:
            return {
                "status": "error",
                "model_name": self.model_name,
                "error": str(e),
                "base_url": self.base_url
            }


def create_ollama_service(
    model_name: str = "bge-m3",
    base_url: str = "http://localhost:11434",
    **kwargs
) -> OllamaEmbeddingService:
    """
    建立 Ollama embedding 服務的便利函數
    
    Args:
        model_name: Ollama 模型名稱
        base_url: Ollama API 基礎 URL
        **kwargs: 其他參數
        
    Returns:
        OllamaEmbeddingService: Ollama 服務實例
    """
    return OllamaEmbeddingService(
        model_name=model_name,
        base_url=base_url,
        **kwargs
    )