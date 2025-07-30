"""
測試 Embedding 效能優化功能

測試快取、GPU 加速、記憶體優化和使用量監控功能
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.chinese_graphrag.embeddings import (
    # 快取相關
    MemoryCache,
    DiskCache,
    MultiLevelCache,
    CacheEntry,
    LRUStrategy,
    LFUStrategy,
    create_embedding_cache,
    
    # GPU 加速相關
    DeviceManager,
    MemoryOptimizer,
    BatchProcessor,
    get_device_manager,
    get_memory_optimizer,
    create_batch_processor,
    
    # 監控相關
    UsageMonitor,
    UsageRecord,
    ModelStats,
    Alert,
    AlertLevel,
    get_usage_monitor,
    record_embedding_usage,
    
    # 管理器
    EmbeddingManager
)


class TestEmbeddingCache:
    """測試 Embedding 快取功能"""
    
    @pytest.fixture
    def sample_embeddings(self):
        """樣本 embedding 資料"""
        return np.random.rand(10, 768).astype(np.float32)
    
    @pytest.fixture
    def sample_texts(self):
        """樣本文本"""
        return [f"測試文本 {i}" for i in range(10)]
    
    @pytest.fixture
    def cache_entry(self, sample_embeddings, sample_texts):
        """樣本快取條目"""
        return CacheEntry(
            key="test_key",
            embeddings=sample_embeddings,
            texts=sample_texts,
            model_name="test_model",
            timestamp=time.time()
        )
    
    def test_cache_entry_creation(self, cache_entry):
        """測試快取條目建立"""
        assert cache_entry.key == "test_key"
        assert cache_entry.model_name == "test_model"
        assert len(cache_entry.texts) == 10
        assert cache_entry.embeddings.shape == (10, 768)
        assert cache_entry.size_bytes > 0
        assert cache_entry.age_seconds >= 0
    
    def test_lru_strategy(self):
        """測試 LRU 淘汰策略"""
        strategy = LRUStrategy()
        
        # 建立測試條目
        old_entry = CacheEntry(
            key="old",
            embeddings=np.random.rand(5, 100),
            texts=["old text"],
            model_name="test",
            timestamp=time.time() - 7200,  # 2 小時前
            last_access=time.time() - 7200
        )
        
        cache_stats = {"memory_usage_ratio": 0.85}
        
        # 應該被淘汰
        assert strategy.should_evict(old_entry, cache_stats)
        
        # 優先級應該很高（數值大）
        priority = strategy.get_eviction_priority(old_entry)
        assert priority > 3600  # 超過 1 小時
    
    def test_lfu_strategy(self):
        """測試 LFU 淘汰策略"""
        strategy = LFUStrategy()
        
        # 建立低頻存取條目
        low_freq_entry = CacheEntry(
            key="low_freq",
            embeddings=np.random.rand(5, 100),
            texts=["low freq text"],
            model_name="test",
            timestamp=time.time(),
            access_count=1
        )
        
        cache_stats = {
            "memory_usage_ratio": 0.85,
            "avg_access_count": 10.0
        }
        
        # 應該被淘汰
        assert strategy.should_evict(low_freq_entry, cache_stats)
        
        # 優先級應該較高
        priority = strategy.get_eviction_priority(low_freq_entry)
        assert priority > 0.1
    
    @pytest.mark.asyncio
    async def test_memory_cache(self, cache_entry):
        """測試記憶體快取"""
        cache = MemoryCache(max_size_mb=1)  # 1MB 限制
        
        # 測試存入和取得
        success = await cache.put("test_key", cache_entry)
        assert success
        
        retrieved = await cache.get("test_key")
        assert retrieved is not None
        assert retrieved.key == cache_entry.key
        assert np.array_equal(retrieved.embeddings, cache_entry.embeddings)
        
        # 測試統計資訊
        stats = await cache.get_stats()
        assert stats["entry_count"] == 1
        assert stats["hit_rate"] > 0
        
        # 測試刪除
        deleted = await cache.delete("test_key")
        assert deleted
        
        # 確認已刪除
        retrieved = await cache.get("test_key")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_disk_cache(self, cache_entry):
        """測試磁碟快取"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(
                cache_dir=temp_dir,
                max_size_mb=10
            )
            
            # 等待索引載入
            await asyncio.sleep(0.1)
            
            # 測試存入和取得
            success = await cache.put("test_key", cache_entry)
            assert success
            
            retrieved = await cache.get("test_key")
            assert retrieved is not None
            assert retrieved.key == cache_entry.key
            
            # 檢查檔案是否存在
            cache_files = list(Path(temp_dir).rglob("*.pkl"))
            assert len(cache_files) > 0
            
            # 測試清空
            await cache.clear()
            stats = await cache.get_stats()
            assert stats["entry_count"] == 0
    
    @pytest.mark.asyncio
    async def test_multi_level_cache(self, cache_entry):
        """測試多層次快取"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = MultiLevelCache(
                memory_cache_mb=1,
                disk_cache_mb=10,
                cache_dir=temp_dir
            )
            
            # 測試存入
            success = await cache.put("test_key", cache_entry)
            assert success
            
            # 應該能從記憶體快取取得
            retrieved = await cache.get("test_key")
            assert retrieved is not None
            
            # 清空記憶體快取，應該能從磁碟快取取得
            await cache.memory_cache.clear()
            retrieved = await cache.get("test_key")
            assert retrieved is not None
            
            # 測試統計資訊
            stats = await cache.get_stats()
            assert "memory_cache" in stats
            assert "disk_cache" in stats
    
    def test_create_embedding_cache(self):
        """測試快取建立函數"""
        # 測試記憶體快取
        memory_cache = create_embedding_cache("memory", max_size_mb=100)
        assert isinstance(memory_cache, MemoryCache)
        
        # 測試磁碟快取
        with tempfile.TemporaryDirectory() as temp_dir:
            disk_cache = create_embedding_cache("disk", cache_dir=temp_dir)
            assert isinstance(disk_cache, DiskCache)
        
        # 測試多層次快取
        with tempfile.TemporaryDirectory() as temp_dir:
            multi_cache = create_embedding_cache("multi_level", cache_dir=temp_dir)
            assert isinstance(multi_cache, MultiLevelCache)
        
        # 測試無效類型
        with pytest.raises(ValueError):
            create_embedding_cache("invalid_type")


class TestGPUAcceleration:
    """測試 GPU 加速功能"""
    
    def test_device_manager_singleton(self):
        """測試裝置管理器單例模式"""
        manager1 = get_device_manager()
        manager2 = get_device_manager()
        assert manager1 is manager2
    
    def test_device_manager_detection(self):
        """測試裝置偵測"""
        manager = DeviceManager()
        
        # 至少應該有 CPU
        assert "cpu" in manager.available_devices
        
        # 測試取得最佳裝置
        device = manager.get_optimal_device(prefer_gpu=False)
        assert device == "cpu"
        
        # 測試裝置資訊
        info = manager.get_device_info("cpu")
        assert info["device"] == "cpu"
        assert info["type"] == "CPU"
    
    def test_memory_optimizer(self):
        """測試記憶體優化器"""
        optimizer = MemoryOptimizer(
            memory_threshold=0.9,
            enable_auto_cleanup=False  # 測試時不啟用自動清理
        )
        
        # 測試記憶體統計
        stats = optimizer.get_memory_stats()
        assert stats.system_total > 0
        assert stats.process_used > 0
        assert 0 <= stats.system_usage_ratio <= 1
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self):
        """測試記憶體清理"""
        optimizer = MemoryOptimizer(enable_auto_cleanup=False)
        
        # 執行清理
        result = await optimizer.cleanup_memory()
        
        assert "before_cleanup" in result
        assert "after_cleanup" in result
        assert "actions_taken" in result
        assert "cleanup_time" in result
        assert result["cleanup_time"] > 0
    
    def test_batch_processor_creation(self):
        """測試批次處理器建立"""
        processor = create_batch_processor(
            initial_batch_size=16,
            min_batch_size=1,
            max_batch_size=128
        )
        
        assert isinstance(processor, BatchProcessor)
        assert processor.initial_batch_size == 16
        assert processor.min_batch_size == 1
        assert processor.max_batch_size == 128
    
    def test_batch_size_calculation(self):
        """測試批次大小計算"""
        processor = create_batch_processor()
        
        # 測試 CPU 裝置
        batch_size = processor.calculate_optimal_batch_size(
            device="cpu",
            estimated_memory_per_item=10
        )
        
        assert processor.min_batch_size <= batch_size <= processor.max_batch_size
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """測試批次處理"""
        processor = create_batch_processor(
            initial_batch_size=5,
            min_batch_size=1,
            max_batch_size=10
        )
        
        # 模擬處理函數
        async def mock_process_func(items):
            await asyncio.sleep(0.01)  # 模擬處理時間
            return [f"processed_{item}" for item in items]
        
        # 測試資料
        test_items = list(range(20))
        
        # 執行批次處理
        results = await processor.process_in_batches(
            items=test_items,
            process_func=mock_process_func,
            device="cpu",
            estimated_memory_per_item=1
        )
        
        assert len(results) == 20
        assert all(result.startswith("processed_") for result in results)
        
        # 檢查統計資訊
        stats = processor.get_batch_stats()
        assert stats["total_batches"] > 0
        assert stats["success_rate"] > 0


class TestUsageMonitoring:
    """測試使用量監控功能"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """臨時儲存目錄"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_usage_record_creation(self):
        """測試使用量記錄建立"""
        record = UsageRecord(
            timestamp=time.time(),
            model_name="test_model",
            operation="embed_texts",
            input_count=10,
            input_tokens=500,
            output_dimensions=768,
            processing_time=1.5,
            memory_used=100.0,
            device="cpu",
            success=True
        )
        
        assert record.model_name == "test_model"
        assert record.throughput > 0
        assert record.tokens_per_second > 0
    
    def test_model_stats_calculation(self):
        """測試模型統計計算"""
        stats = ModelStats("test_model")
        
        # 模擬一些請求
        stats.total_requests = 10
        stats.successful_requests = 8
        stats.failed_requests = 2
        stats.total_processing_time = 15.0
        stats.total_input_count = 100
        
        assert stats.success_rate == 0.8
        assert stats.average_processing_time == 15.0 / 8
        assert stats.average_throughput == 100 / 15.0
    
    def test_usage_monitor_creation(self, temp_storage_dir):
        """測試使用量監控器建立"""
        monitor = UsageMonitor(
            storage_dir=temp_storage_dir,
            max_records_in_memory=100,
            save_interval=60,
            enable_alerts=True
        )
        
        assert monitor.storage_dir == Path(temp_storage_dir)
        assert monitor.max_records_in_memory == 100
        assert monitor.enable_alerts
    
    def test_record_usage(self, temp_storage_dir):
        """測試使用量記錄"""
        monitor = UsageMonitor(
            storage_dir=temp_storage_dir,
            enable_alerts=False  # 測試時不啟用警報
        )
        
        # 記錄使用量
        monitor.record_usage(
            model_name="test_model",
            operation="embed_texts",
            input_count=10,
            processing_time=1.0,
            memory_used=50.0,
            device="cpu",
            success=True
        )
        
        # 檢查統計
        stats = monitor.get_model_stats("test_model")
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.total_processing_time == 1.0
    
    def test_usage_summary(self, temp_storage_dir):
        """測試使用量摘要"""
        monitor = UsageMonitor(storage_dir=temp_storage_dir, enable_alerts=False)
        
        # 記錄多個使用量
        for i in range(5):
            monitor.record_usage(
                model_name="test_model",
                operation="embed_texts",
                input_count=10,
                processing_time=1.0,
                success=True
            )
        
        # 取得摘要
        summary = monitor.get_usage_summary(time_range_hours=1)
        
        assert summary["summary"]["total_requests"] == 5
        assert summary["summary"]["successful_requests"] == 5
        assert summary["summary"]["success_rate"] == 1.0
        assert "test_model" in summary["model_breakdown"]
    
    def test_alert_creation(self, temp_storage_dir):
        """測試警報建立"""
        monitor = UsageMonitor(storage_dir=temp_storage_dir, enable_alerts=True)
        
        # 記錄一個失敗的請求（觸發錯誤率警報需要多個樣本）
        for i in range(15):
            success = i < 5  # 前 5 個成功，後 10 個失敗
            monitor.record_usage(
                model_name="test_model",
                operation="embed_texts",
                input_count=10,
                processing_time=1.0,
                success=success,
                error_message="測試錯誤" if not success else None
            )
        
        # 檢查是否有警報
        alerts = monitor.get_alerts(unresolved_only=True)
        # 可能會有錯誤率警報
        assert isinstance(alerts, list)
    
    def test_usage_monitor_singleton(self):
        """測試使用量監控器單例模式"""
        monitor1 = get_usage_monitor()
        monitor2 = get_usage_monitor()
        assert monitor1 is monitor2
    
    def test_record_embedding_usage_function(self):
        """測試記錄 embedding 使用量函數"""
        # 這個函數應該不會拋出異常
        record_embedding_usage(
            model_name="test_model",
            operation="embed_texts",
            input_count=10,
            processing_time=1.0,
            success=True
        )


class TestEmbeddingManagerIntegration:
    """測試 EmbeddingManager 整合功能"""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """模擬 embedding 服務"""
        service = Mock()
        service.model_name = "mock_model"
        service.model_type = "mock"
        service.is_loaded = True
        service.device = "cpu"
        
        # 模擬 embed_texts 方法
        async def mock_embed_texts(texts, normalize=True, show_progress=False):
            await asyncio.sleep(0.01)  # 模擬處理時間
            embeddings = np.random.rand(len(texts), 768).astype(np.float32)
            from src.chinese_graphrag.embeddings.base import EmbeddingResult
            return EmbeddingResult(
                embeddings=embeddings,
                texts=texts,
                model_name="mock_model",
                dimensions=768,
                processing_time=0.01
            )
        
        service.embed_texts = mock_embed_texts
        return service
    
    @pytest.mark.asyncio
    async def test_embedding_manager_with_cache(self, mock_embedding_service):
        """測試帶快取的 EmbeddingManager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = EmbeddingManager(
                enable_cache=True,
                cache_config={"cache_dir": temp_dir, "memory_cache_mb": 10},
                enable_gpu_acceleration=False,  # 簡化測試
                enable_monitoring=False
            )
            
            # 註冊模擬服務
            manager.register_service("mock_service", mock_embedding_service, set_as_default=True)
            
            # 第一次調用（應該快取）
            texts = ["測試文本1", "測試文本2"]
            result1 = await manager.embed_texts(texts)
            
            assert result1.embeddings.shape == (2, 768)
            assert len(result1.texts) == 2
            
            # 第二次調用（應該命中快取）
            result2 = await manager.embed_texts(texts)
            
            # 檢查快取統計
            cache_stats = await manager.get_cache_stats()
            assert cache_stats is not None
            assert cache_stats["hit_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_embedding_manager_with_monitoring(self, mock_embedding_service):
        """測試帶監控的 EmbeddingManager"""
        manager = EmbeddingManager(
            enable_cache=False,
            enable_gpu_acceleration=False,
            enable_monitoring=True
        )
        
        # 註冊模擬服務
        manager.register_service("mock_service", mock_embedding_service, set_as_default=True)
        
        # 執行一些請求
        texts = ["測試文本1", "測試文本2"]
        await manager.embed_texts(texts)
        await manager.embed_texts(["另一個測試"])
        
        # 檢查使用量統計
        usage_stats = manager.get_usage_stats(time_range_hours=1)
        assert usage_stats is not None
        assert usage_stats["summary"]["total_requests"] >= 2
    
    def test_embedding_manager_comprehensive_stats(self, mock_embedding_service):
        """測試綜合統計資訊"""
        manager = EmbeddingManager(
            enable_cache=True,
            enable_gpu_acceleration=True,
            enable_monitoring=True
        )
        
        # 註冊模擬服務
        manager.register_service("mock_service", mock_embedding_service, set_as_default=True)
        
        # 取得綜合統計
        stats = manager.get_comprehensive_stats()
        
        assert "manager_info" in stats
        assert "services" in stats
        assert stats["manager_info"]["total_services"] == 1
        assert stats["manager_info"]["enable_cache"]
        assert stats["manager_info"]["enable_gpu_acceleration"]
        assert stats["manager_info"]["enable_monitoring"]


if __name__ == "__main__":
    # 執行測試
    pytest.main([__file__, "-v"])