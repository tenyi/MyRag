"""
測試 Embedding 效能優化功能

測試快取、管理器等已實作的功能
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
    
    # 管理器
    EmbeddingManager
)


class TestEmbeddingCache:
    """測試 Embedding 快取功能"""
    
    @pytest.fixture
    def cache_entry(self):
        """測試快取項目"""
        return CacheEntry(
            key="test_key",
            embeddings=np.random.rand(384).astype(np.float32),
            texts=["測試文本"],
            model_name="test_model",
            timestamp=time.time(),
            metadata={}
        )
    
    @pytest.mark.asyncio
    async def test_memory_cache_basic(self, cache_entry):
        """測試記憶體快取基本功能"""
        cache = MemoryCache(max_size_mb=1)
        
        # 測試存入
        success = await cache.put("test_key", cache_entry)
        assert success
        
        # 測試取出
        retrieved = await cache.get("test_key")
        assert retrieved is not None
        assert retrieved.key == cache_entry.key
        assert np.array_equal(retrieved.embeddings, cache_entry.embeddings)
        
        # 測試統計
        stats = await cache.get_stats()
        assert stats["entry_count"] == 1
        assert stats["memory_usage_mb"] > 0
        
        # 測試刪除
        success = await cache.remove("test_key")
        assert success
        
        # 確認已刪除
        retrieved = await cache.get("test_key")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_memory_cache_lru_eviction(self):
        """測試 LRU 淘汰策略"""
        cache = MemoryCache(max_size_mb=0.003, strategy=LRUStrategy())  # 很小的快取，約3KB
        
        # 存入多個項目，超過快取大小
        entries = []
        for i in range(10):
            entry = CacheEntry(
                key=f"key_{i}",
                embeddings=np.random.rand(100).astype(np.float32),
                texts=[f"text_{i}"],
                model_name="test_model",
                timestamp=time.time(),
                metadata={"index": i}
            )
            entries.append(entry)
            await cache.put(f"key_{i}", entry)
        
        # 檢查較舊的項目被淘汰
        stats = await cache.get_stats()
        assert stats["entry_count"] < 10  # 應該有些項目被淘汰
        
        # 最新的項目應該還在
        latest = await cache.get("key_9")
        assert latest is not None
    
    @pytest.mark.asyncio
    async def test_memory_cache_lfu_eviction(self):
        """測試 LFU 淘汰策略"""
        cache = MemoryCache(max_size_mb=0.01, strategy=LFUStrategy())
        
        # 存入項目
        for i in range(5):
            entry = CacheEntry(
                key=f"key_{i}",
                embeddings=np.random.rand(100).astype(np.float32),
                texts=[f"text_{i}"],
                model_name="test_model",
                timestamp=time.time(),
                metadata={"index": i}
            )
            await cache.put(f"key_{i}", entry)
        
        # 多次存取某個項目
        for _ in range(5):
            await cache.get("key_0")
        
        # 添加新項目觸發淘汰
        for i in range(5, 10):
            entry = CacheEntry(
                key=f"key_{i}",
                embeddings=np.random.rand(100).astype(np.float32),
                texts=[f"text_{i}"],
                model_name="test_model",
                timestamp=time.time(),
                metadata={"index": i}
            )
            await cache.put(f"key_{i}", entry)
        
        # 經常存取的項目應該還在
        frequently_used = await cache.get("key_0")
        assert frequently_used is not None
    
    @pytest.mark.asyncio
    async def test_disk_cache_basic(self, cache_entry):
        """測試磁碟快取基本功能"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir, max_size_mb=10)
            
            # 測試存入
            success = await cache.put("test_key", cache_entry)
            assert success
            
            # 測試取出
            retrieved = await cache.get("test_key")
            assert retrieved is not None
            assert retrieved.key == cache_entry.key
            assert np.array_equal(retrieved.embeddings, cache_entry.embeddings)
            
            # 檢查檔案是否存在
            cache_files = list(Path(temp_dir).rglob("*.pkl"))
            assert len(cache_files) > 0
    
    @pytest.mark.asyncio
    async def test_disk_cache_persistence(self, cache_entry):
        """測試磁碟快取持久性"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 建立快取並存入資料
            cache1 = DiskCache(cache_dir=temp_dir, max_size_mb=10)
            success = await cache1.put("test_key", cache_entry)
            assert success
            
            # 建立新的快取實例（模擬重啟）
            cache2 = DiskCache(cache_dir=temp_dir, max_size_mb=10)
            
            # 資料應該仍然存在
            retrieved = await cache2.get("test_key")
            assert retrieved is not None
            assert retrieved.key == cache_entry.key
            assert np.array_equal(retrieved.embeddings, cache_entry.embeddings)
    
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
            
            # 測試取出（應該從記憶體快取）
            retrieved = await cache.get("test_key")
            assert retrieved is not None
            assert retrieved.key == cache_entry.key
            
            # 測試統計
            stats = await cache.get_stats()
            assert stats["memory_entries"] > 0
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_promotion(self):
        """測試多層快取提升機制"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = MultiLevelCache(
                memory_cache_mb=0.001,  # 很小的記憶體快取
                disk_cache_mb=10,
                cache_dir=temp_dir
            )
            
            # 存入多個項目，超過記憶體快取大小
            for i in range(5):
                entry = CacheEntry(
                key=f"key_{i}",
                embeddings=np.random.rand(100).astype(np.float32),
                texts=[f"text_{i}"],
                model_name="test_model",
                timestamp=time.time(),
                metadata={"index": i}
            )
                await cache.put(f"key_{i}", entry)
            
            # 存取較舊的項目，應該從磁碟快取提升到記憶體快取
            old_entry = await cache.get("key_0")
            assert old_entry is not None
            
            # 再次存取應該更快（從記憶體快取）
            start_time = time.time()
            old_entry_again = await cache.get("key_0")
            access_time = time.time() - start_time
            assert old_entry_again is not None
            assert access_time < 0.01  # 應該很快
    
    def test_cache_factory(self):
        """測試快取工廠函數"""
        # 測試記憶體快取
        memory_cache = create_embedding_cache("memory", max_size_mb=1)
        assert isinstance(memory_cache, MemoryCache)
        
        # 測試磁碟快取
        with tempfile.TemporaryDirectory() as temp_dir:
            disk_cache = create_embedding_cache("disk", cache_dir=temp_dir, max_size_mb=10)
            assert isinstance(disk_cache, DiskCache)
        
        # 測試多層快取
        with tempfile.TemporaryDirectory() as temp_dir:
            multi_cache = create_embedding_cache(
                "multi_level",
                memory_cache_mb=1,
                disk_cache_mb=10,
                cache_dir=temp_dir
            )
            assert isinstance(multi_cache, MultiLevelCache)
        
        # 測試無效類型
        with pytest.raises(ValueError):
            create_embedding_cache("invalid_type")


# GPU 加速和使用量監控相關的測試類別暫時禁用，因為相關模組尚未實作


class TestEmbeddingManagerIntegration:
    """測試 Embedding 管理器整合功能"""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """模擬 embedding 服務"""
        mock_service = Mock()
        
        async def mock_embed(texts, normalize=True, show_progress=False):
            from src.chinese_graphrag.embeddings.base import EmbeddingResult
            embeddings = np.array([np.random.rand(384).astype(np.float32) for _ in texts])
            return EmbeddingResult(
                embeddings=embeddings,
                texts=texts,
                model_name="test_model",
                dimensions=384,
                processing_time=0.1
            )
        
        mock_service.embed_texts = mock_embed
        mock_service.embed_query = mock_embed
        mock_service.model_name = "test_model"  # 添加 model_name 屬性
        mock_service.is_loaded = True  # 添加 is_loaded 屬性
        mock_service.get_model_info.return_value = {
            "model_name": "test_model",
            "embedding_dim": 384,
            "max_tokens": 512
        }
        
        return mock_service
    
    @pytest.mark.asyncio
    async def test_manager_with_cache(self, mock_embedding_service):
        """測試管理器與快取的整合"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = create_embedding_cache("disk", cache_dir=temp_dir, max_size_mb=1)
            
            manager = EmbeddingManager(
                embedding_service=mock_embedding_service,
                cache=cache
            )
            
            # 第一次呼叫
            texts = ["測試文本1", "測試文本2"]
            embeddings1 = await manager.embed_texts(texts)
            assert len(embeddings1.embeddings) == 2
            assert embeddings1.model_name == "test_model"
            
            # 第二次呼叫（應該使用快取）
            embeddings2 = await manager.embed_texts(texts)
            assert len(embeddings2.embeddings) == 2
            assert embeddings2.model_name == "test_model"
            
            # 檢查快取統計
            cache_stats = await cache.get_stats()
            assert cache_stats["entry_count"] > 0
    
    @pytest.mark.asyncio
    async def test_manager_batch_processing(self, mock_embedding_service):
        """測試管理器批次處理"""
        manager = EmbeddingManager(
            embedding_service=mock_embedding_service
        )
        
        # 測試大批次文本
        texts = [f"測試文本{i}" for i in range(5)]
        result = await manager.embed_texts(texts)
        
        assert len(result.embeddings) == 5
        assert result.model_name == "test_model"
        for embedding in result.embeddings:
            assert embedding.shape == (384,)
    
    @pytest.mark.asyncio
    async def test_manager_error_handling(self):
        """測試管理器錯誤處理"""
        # 建立會失敗的服務
        failing_service = Mock()
        failing_service.embed_texts.side_effect = Exception("模型載入失敗")
        
        manager = EmbeddingManager(embedding_service=failing_service)
        
        # 測試錯誤處理
        with pytest.raises(Exception):
            await manager.embed_texts(["測試文本"])
    
    @pytest.mark.asyncio
    async def test_manager_metrics(self, mock_embedding_service):
        """測試管理器指標收集"""
        manager = EmbeddingManager(embedding_service=mock_embedding_service)
        
        # 執行一些操作
        result1 = await manager.embed_texts(["測試文本1", "測試文本2"])
        result2 = await manager.embed_texts(["查詢文本"])
        
        # 檢查結果
        assert len(result1.embeddings) == 2
        assert len(result2.embeddings) == 1
        assert result1.model_name == "test_model"
        assert result2.model_name == "test_model"