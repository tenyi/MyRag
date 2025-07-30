#!/usr/bin/env python3
"""
簡單的效能優化功能測試
"""

import asyncio
import tempfile
from pathlib import Path

# 測試快取功能
async def test_cache():
    print("測試快取功能...")
    
    from chinese_graphrag.embeddings.cache import MemoryCache, CacheEntry
    import numpy as np
    import time
    
    # 建立記憶體快取
    cache = MemoryCache(max_size_mb=10)
    
    # 建立測試資料
    embeddings = np.random.rand(5, 768).astype(np.float32)
    texts = [f"測試文本 {i}" for i in range(5)]
    
    entry = CacheEntry(
        key="test_key",
        embeddings=embeddings,
        texts=texts,
        model_name="test_model",
        timestamp=time.time()
    )
    
    # 測試存入
    success = await cache.put("test_key", entry)
    print(f"  存入結果: {success}")
    
    # 測試取得
    retrieved = await cache.get("test_key")
    print(f"  取得結果: {retrieved is not None}")
    
    if retrieved:
        print(f"  向量形狀: {retrieved.embeddings.shape}")
        print(f"  文本數量: {len(retrieved.texts)}")
    
    # 測試統計
    stats = await cache.get_stats()
    print(f"  快取統計: 條目數={stats['entry_count']}, 命中率={stats['hit_rate']:.1%}")
    
    print("✓ 快取功能測試完成")


def test_device_manager():
    print("測試裝置管理...")
    
    from chinese_graphrag.embeddings.gpu_acceleration import DeviceManager
    
    manager = DeviceManager()
    
    print(f"  可用裝置: {manager.available_devices}")
    
    # 取得最佳裝置
    device = manager.get_optimal_device(prefer_gpu=False)
    print(f"  推薦裝置: {device}")
    
    # 裝置資訊
    info = manager.get_device_info(device)
    print(f"  裝置類型: {info.get('type', 'unknown')}")
    
    print("✓ 裝置管理測試完成")


def test_memory_optimizer():
    print("測試記憶體優化...")
    
    from chinese_graphrag.embeddings.gpu_acceleration import MemoryOptimizer
    
    optimizer = MemoryOptimizer(enable_auto_cleanup=False)
    
    # 取得記憶體統計
    stats = optimizer.get_memory_stats()
    print(f"  系統記憶體: {stats.system_used}/{stats.system_total} MB ({stats.system_usage_ratio:.1%})")
    print(f"  程序記憶體: {stats.process_used} MB")
    
    print("✓ 記憶體優化測試完成")


def test_usage_monitor():
    print("測試使用量監控...")
    
    from chinese_graphrag.embeddings.monitoring import UsageMonitor
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = UsageMonitor(
            storage_dir=temp_dir,
            enable_alerts=False
        )
        
        # 記錄一些使用量
        monitor.record_usage(
            model_name="test_model",
            operation="embed_texts",
            input_count=10,
            processing_time=1.0,
            success=True
        )
        
        monitor.record_usage(
            model_name="test_model",
            operation="embed_texts",
            input_count=5,
            processing_time=0.5,
            success=True
        )
        
        # 取得統計
        stats = monitor.get_model_stats("test_model")
        print(f"  模型統計: 請求數={stats.total_requests}, 成功率={stats.success_rate:.1%}")
        print(f"  平均處理時間: {stats.average_processing_time:.3f}s")
        
        # 取得摘要
        summary = monitor.get_usage_summary(time_range_hours=1)
        print(f"  使用量摘要: 總請求={summary['summary']['total_requests']}")
    
    print("✓ 使用量監控測試完成")


async def test_embedding_manager():
    print("測試 EmbeddingManager 整合...")
    
    from chinese_graphrag.embeddings import EmbeddingManager
    from unittest.mock import Mock
    import numpy as np
    
    # 建立帶優化功能的管理器
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = EmbeddingManager(
            enable_cache=True,
            cache_config={"cache_dir": temp_dir, "memory_cache_mb": 10},
            enable_gpu_acceleration=True,
            enable_monitoring=True
        )
        
        # 建立模擬服務
        mock_service = Mock()
        mock_service.model_name = "mock_model"
        mock_service.is_loaded = True
        mock_service.device = "cpu"
        
        async def mock_embed_texts(texts, normalize=True, show_progress=False):
            await asyncio.sleep(0.01)
            embeddings = np.random.rand(len(texts), 768).astype(np.float32)
            from chinese_graphrag.embeddings.base import EmbeddingResult
            return EmbeddingResult(
                embeddings=embeddings,
                texts=texts,
                model_name="mock_model",
                dimensions=768,
                processing_time=0.01
            )
        
        mock_service.embed_texts = mock_embed_texts
        manager.register_service("mock", mock_service, set_as_default=True)
        
        # 測試向量化
        texts = ["測試文本1", "測試文本2"]
        result = await manager.embed_texts(texts)
        
        print(f"  向量化結果: 形狀={result.embeddings.shape}")
        print(f"  處理時間: {result.processing_time:.3f}s")
        
        # 測試快取統計
        cache_stats = await manager.get_cache_stats()
        if cache_stats:
            print(f"  快取統計: 類型={cache_stats.get('cache_type', 'unknown')}")
        
        # 測試使用量統計
        usage_stats = manager.get_usage_stats(time_range_hours=1)
        if usage_stats:
            print(f"  使用量統計: 請求數={usage_stats['summary']['total_requests']}")
    
    print("✓ EmbeddingManager 整合測試完成")


async def main():
    print("🧪 Embedding 效能優化功能簡單測試")
    print("=" * 40)
    
    try:
        # 測試各個模組
        await test_cache()
        print()
        
        test_device_manager()
        print()
        
        test_memory_optimizer()
        print()
        
        test_usage_monitor()
        print()
        
        await test_embedding_manager()
        print()
        
        print("=" * 40)
        print("✅ 所有測試完成！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())