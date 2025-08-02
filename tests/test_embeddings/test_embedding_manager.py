"""
測試 Embedding 管理器功能
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock

from src.chinese_graphrag.embeddings import (
    EmbeddingManager,
    EmbeddingService,
    EmbeddingResult,
    EmbeddingModelType,
    EmbeddingServiceError
)


class MockEmbeddingService(EmbeddingService):
    """模擬 Embedding 服務用於測試"""
    
    def __init__(self, name: str, dimensions: int = 768, fail_on_load: bool = False):
        super().__init__(
            model_name=name,
            model_type=EmbeddingModelType.LOCAL,
            max_batch_size=32,
            max_sequence_length=512
        )
        self.dimensions = dimensions
        self.fail_on_load = fail_on_load
    
    async def load_model(self) -> None:
        if self.fail_on_load:
            raise Exception("模擬載入失敗")
        self.is_loaded = True
    
    async def unload_model(self) -> None:
        self.is_loaded = False
    
    def get_embedding_dimension(self) -> int:
        return self.dimensions
    
    async def embed_texts(
        self, 
        texts: list[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> EmbeddingResult:
        if not self.is_loaded:
            raise Exception("模型未載入")
        
        # 生成模擬向量
        embeddings = np.random.rand(len(texts), self.dimensions).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)  # 避免除零
        
        return EmbeddingResult(
            embeddings=embeddings,
            texts=texts,
            model_name=self.model_name,
            dimensions=self.dimensions,
            processing_time=0.1,
            metadata={}
        )


class TestEmbeddingManager:
    """測試 EmbeddingManager 類別"""
    
    def test_init(self):
        """測試初始化"""
        manager = EmbeddingManager(config_or_model="test_model")
        assert manager.default_model == "test_model"
        assert manager.enable_fallback is True
        assert len(manager.services) == 0
    
    def test_register_service(self):
        """測試註冊服務"""
        manager = EmbeddingManager()
        service = MockEmbeddingService("test_service")
        
        manager.register_service("test", service, set_as_default=True)
        
        assert "test" in manager.services
        assert manager.default_model == "test"
        assert manager.get_service("test") == service
    
    def test_unregister_service(self):
        """測試取消註冊服務"""
        manager = EmbeddingManager()
        service = MockEmbeddingService("test_service")
        
        manager.register_service("test", service, set_as_default=True)
        manager.unregister_service("test")
        
        assert "test" not in manager.services
        assert manager.default_model is None
    
    def test_get_service_not_found(self):
        """測試取得不存在的服務"""
        manager = EmbeddingManager()
        
        with pytest.raises(ValueError, match="Embedding 服務 'nonexistent' 不存在"):
            manager.get_service("nonexistent")
    
    def test_list_services(self):
        """測試列出服務"""
        manager = EmbeddingManager()
        service1 = MockEmbeddingService("service1")
        service2 = MockEmbeddingService("service2")
        
        manager.register_service("test1", service1, set_as_default=True)
        manager.register_service("test2", service2)
        
        services_info = manager.list_services()
        
        assert len(services_info) == 2
        assert any(info['service_name'] == 'test1' and info['is_default'] for info in services_info)
        assert any(info['service_name'] == 'test2' and not info['is_default'] for info in services_info)
    
    @pytest.mark.asyncio
    async def test_load_all_models(self):
        """測試載入所有模型"""
        manager = EmbeddingManager()
        service1 = MockEmbeddingService("service1")
        service2 = MockEmbeddingService("service2", fail_on_load=True)
        
        manager.register_service("test1", service1)
        manager.register_service("test2", service2)
        
        results = await manager.load_all_models()
        
        assert results["test1"] is True
        assert results["test2"] is False
        assert service1.is_loaded
        assert not service2.is_loaded
    
    @pytest.mark.asyncio
    async def test_embed_texts_success(self):
        """測試成功的文本向量化"""
        manager = EmbeddingManager()
        service = MockEmbeddingService("test_service")
        
        manager.register_service("test", service, set_as_default=True)
        await service.load_model()
        
        texts = ["測試文本1", "測試文本2"]
        result = await manager.embed_texts(texts)
        
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings.shape == (2, 768)
        assert len(result.texts) == 2
        assert result.model_name == "test_service"
    
    @pytest.mark.asyncio
    async def test_embed_texts_fallback(self):
        """測試降級處理"""
        manager = EmbeddingManager(enable_fallback=True)
        
        # 主要服務會失敗
        primary_service = MockEmbeddingService("primary")
        primary_service.embed_texts = AsyncMock(side_effect=Exception("主要服務失敗"))
        
        # 降級服務正常
        fallback_service = MockEmbeddingService("fallback")
        
        manager.register_service("primary", primary_service, set_as_default=True)
        manager.register_service("fallback", fallback_service)
        
        await primary_service.load_model()
        await fallback_service.load_model()
        
        texts = ["測試文本"]
        result = await manager.embed_texts(texts)
        
        # 應該使用降級服務
        assert result.model_name == "fallback"
    
    @pytest.mark.asyncio
    async def test_embed_texts_no_fallback(self):
        """測試禁用降級處理時的行為"""
        manager = EmbeddingManager(enable_fallback=False)
        service = MockEmbeddingService("test_service")
        service.embed_texts = AsyncMock(side_effect=Exception("服務失敗"))
        
        manager.register_service("test", service, set_as_default=True)
        await service.load_model()
        
        texts = ["測試文本"]
        
        with pytest.raises(Exception, match="服務失敗"):
            await manager.embed_texts(texts)
    
    @pytest.mark.asyncio
    async def test_smart_route_request_fastest(self):
        """測試智慧路由 - 最快策略"""
        manager = EmbeddingManager()
        
        # 建立兩個服務，一個較快一個較慢
        fast_service = MockEmbeddingService("fast")
        slow_service = MockEmbeddingService("slow")
        
        # 設定效能指標
        fast_service.metrics.average_processing_time = 0.1
        slow_service.metrics.average_processing_time = 0.5
        
        manager.register_service("fast", fast_service)
        manager.register_service("slow", slow_service)
        
        await fast_service.load_model()
        await slow_service.load_model()
        
        texts = ["測試文本"]
        result = await manager.smart_route_request(texts, strategy="fastest")
        
        # 應該選擇較快的服務
        assert result.model_name == "fast"
    
    @pytest.mark.asyncio
    async def test_smart_route_request_least_loaded(self):
        """測試智慧路由 - 最少負載策略"""
        manager = EmbeddingManager()
        
        # 建立兩個服務，一個負載高一個負載低
        high_load_service = MockEmbeddingService("high_load")
        low_load_service = MockEmbeddingService("low_load")
        
        # 設定請求數
        high_load_service.metrics.total_requests = 100
        low_load_service.metrics.total_requests = 10
        
        manager.register_service("high_load", high_load_service)
        manager.register_service("low_load", low_load_service)
        
        await high_load_service.load_model()
        await low_load_service.load_model()
        
        texts = ["測試文本"]
        result = await manager.smart_route_request(texts, strategy="least_loaded")
        
        # 應該選擇負載較低的服務
        assert result.model_name == "low_load"
    
    @pytest.mark.asyncio
    async def test_health_check_all(self):
        """測試所有服務的健康檢查"""
        manager = EmbeddingManager()
        service1 = MockEmbeddingService("service1")
        service2 = MockEmbeddingService("service2")
        
        manager.register_service("test1", service1)
        manager.register_service("test2", service2)
        
        await service1.load_model()
        await service2.load_model()
        
        health_results = await manager.health_check_all()
        
        assert "test1" in health_results
        assert "test2" in health_results
        assert health_results["test1"]["status"] == "healthy"
        assert health_results["test2"]["status"] == "healthy"
    
    def test_get_metrics_summary(self):
        """測試取得效能指標摘要"""
        manager = EmbeddingManager()
        service = MockEmbeddingService("test_service")
        
        manager.register_service("test", service, set_as_default=True)
        
        summary = manager.get_metrics_summary()
        
        assert summary["total_services"] == 1
        assert summary["loaded_services"] == 0  # 未載入
        assert summary["default_service"] == "test"
        assert "test" in summary["services"]
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """測試異步上下文管理器"""
        manager = EmbeddingManager()
        service = MockEmbeddingService("test_service")
        
        manager.register_service("test", service)
        
        async with manager:
            assert service.is_loaded
        
        assert not service.is_loaded


if __name__ == "__main__":
    pytest.main([__file__])