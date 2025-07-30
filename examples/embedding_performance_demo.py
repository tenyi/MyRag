#!/usr/bin/env python3
"""
Embedding 效能優化功能示範

展示快取、GPU 加速、記憶體優化和使用量監控功能的使用方法
"""

import asyncio
import time
from pathlib import Path

from chinese_graphrag.embeddings import (
    # 管理器
    EmbeddingManager,
    
    # 快取
    create_embedding_cache,
    
    # GPU 加速和記憶體優化
    get_device_manager,
    get_memory_optimizer,
    create_batch_processor,
    
    # 監控
    get_usage_monitor,
    record_embedding_usage,
    
    # 服務
    BGEM3EmbeddingService,
    ChineseOptimizedEmbeddingService
)


async def demo_basic_usage():
    """基本使用示範"""
    print("=== 基本使用示範 ===")
    
    # 建立帶有所有優化功能的 EmbeddingManager
    manager = EmbeddingManager(
        enable_cache=True,
        cache_config={
            "memory_cache_mb": 100,
            "disk_cache_mb": 500,
            "cache_dir": "./cache/demo"
        },
        enable_gpu_acceleration=True,
        enable_monitoring=True
    )
    
    # 註冊一個模擬的 embedding 服務
    try:
        # 嘗試使用 BGE-M3 服務
        service = BGEM3EmbeddingService(
            model_name="BAAI/bge-m3",
            device="cpu",  # 使用 CPU 以確保相容性
            max_sequence_length=512
        )
        
        # 註冊服務
        manager.register_service("bge_m3", service, set_as_default=True)
        
        print("✓ 成功註冊 BGE-M3 服務")
        
    except Exception as e:
        print(f"✗ 無法載入 BGE-M3 服務: {e}")
        print("使用模擬服務進行示範...")
        
        # 建立模擬服務
        from unittest.mock import Mock
        import numpy as np
        
        mock_service = Mock()
        mock_service.model_name = "mock_bge_m3"
        mock_service.model_type = "mock"
        mock_service.is_loaded = True
        mock_service.device = "cpu"
        
        async def mock_embed_texts(texts, normalize=True, show_progress=False):
            await asyncio.sleep(0.1)  # 模擬處理時間
            embeddings = np.random.rand(len(texts), 768).astype(np.float32)
            from chinese_graphrag.embeddings.base import EmbeddingResult
            return EmbeddingResult(
                embeddings=embeddings,
                texts=texts,
                model_name="mock_bge_m3",
                dimensions=768,
                processing_time=0.1
            )
        
        mock_service.embed_texts = mock_embed_texts
        manager.register_service("mock_bge_m3", mock_service, set_as_default=True)
        print("✓ 使用模擬服務")
    
    # 測試文本
    test_texts = [
        "這是一個測試文本，用於展示 embedding 功能。",
        "中文 GraphRAG 系統支援多種 embedding 模型。",
        "快取機制可以顯著提升重複查詢的效能。",
        "GPU 加速能夠處理大批量的文本向量化任務。",
        "使用量監控幫助追蹤系統效能和成本。"
    ]
    
    print(f"\n處理 {len(test_texts)} 個測試文本...")
    
    # 第一次調用（應該會快取結果）
    start_time = time.time()
    result1 = await manager.embed_texts(test_texts)
    first_call_time = time.time() - start_time
    
    print(f"✓ 第一次調用完成，耗時: {first_call_time:.3f}s")
    print(f"  - 向量維度: {result1.dimensions}")
    print(f"  - 處理文本數: {len(result1.texts)}")
    
    # 第二次調用（應該命中快取）
    start_time = time.time()
    result2 = await manager.embed_texts(test_texts)
    second_call_time = time.time() - start_time
    
    print(f"✓ 第二次調用完成，耗時: {second_call_time:.3f}s")
    
    # 檢查快取效果
    if second_call_time < first_call_time * 0.5:
        print("✓ 快取生效，第二次調用明顯更快")
    else:
        print("? 快取可能未生效或使用模擬服務")
    
    return manager


async def demo_cache_functionality(manager):
    """快取功能示範"""
    print("\n=== 快取功能示範 ===")
    
    # 取得快取統計
    cache_stats = await manager.get_cache_stats()
    if cache_stats:
        print("快取統計資訊:")
        print(f"  - 快取類型: {cache_stats.get('cache_type', 'unknown')}")
        print(f"  - 命中率: {cache_stats.get('hit_rate', 0):.1%}")
        print(f"  - 總請求數: {cache_stats.get('total_requests', 0)}")
        
        if 'memory_cache' in cache_stats:
            mem_stats = cache_stats['memory_cache']
            print(f"  - 記憶體快取: {mem_stats.get('entry_count', 0)} 個條目")
            print(f"  - 記憶體使用: {mem_stats.get('current_size_mb', 0):.1f}MB")
        
        if 'disk_cache' in cache_stats:
            disk_stats = cache_stats['disk_cache']
            print(f"  - 磁碟快取: {disk_stats.get('entry_count', 0)} 個條目")
            print(f"  - 磁碟使用: {disk_stats.get('current_size_mb', 0):.1f}MB")
    else:
        print("快取未啟用或無統計資訊")
    
    # 快取預熱示範
    preload_texts = [
        "預載入文本 1：人工智慧技術發展迅速。",
        "預載入文本 2：機器學習模型需要大量資料。",
        "預載入文本 3：自然語言處理是 AI 的重要分支。"
    ]
    
    print(f"\n執行快取預熱，預載入 {len(preload_texts)} 個文本...")
    preload_result = await manager.preload_cache(preload_texts, batch_size=2)
    
    if "error" not in preload_result:
        print("✓ 快取預熱完成")
        print(f"  - 處理批次數: {preload_result.get('processed_batches', 0)}")
        print(f"  - 快取條目數: {preload_result.get('cached_entries', 0)}")
        print(f"  - 耗時: {preload_result.get('duration_seconds', 0):.2f}s")
    else:
        print(f"✗ 快取預熱失敗: {preload_result['error']}")


def demo_device_management():
    """裝置管理示範"""
    print("\n=== 裝置管理示範 ===")
    
    device_manager = get_device_manager()
    
    # 顯示可用裝置
    print("可用裝置:")
    for device in device_manager.available_devices:
        print(f"  - {device}")
    
    # 取得最佳裝置
    optimal_device = device_manager.get_optimal_device(
        memory_required_mb=100,
        prefer_gpu=True
    )
    print(f"\n推薦裝置: {optimal_device}")
    
    # 裝置詳細資訊
    device_info = device_manager.get_device_info(optimal_device)
    print(f"裝置資訊:")
    print(f"  - 類型: {device_info.get('type', 'unknown')}")
    
    if 'memory_info' in device_info:
        mem_info = device_info['memory_info']
        print(f"  - 總記憶體: {mem_info.get('total_mb', 0):.0f}MB")
        print(f"  - 可用記憶體: {mem_info.get('available_mb', 0):.0f}MB")
        print(f"  - 使用率: {mem_info.get('usage_percent', 0):.1f}%")
    
    # 所有裝置統計
    all_stats = device_manager.get_all_device_stats()
    print(f"\n系統總覽:")
    print(f"  - 可用裝置數: {len(all_stats['available_devices'])}")
    print(f"  - GPU 數量: {all_stats['gpu_count']}")


async def demo_memory_optimization():
    """記憶體優化示範"""
    print("\n=== 記憶體優化示範 ===")
    
    memory_optimizer = get_memory_optimizer()
    
    # 取得記憶體統計
    stats = memory_optimizer.get_memory_stats()
    print("記憶體統計:")
    print(f"  - 系統總記憶體: {stats.system_total:.0f}MB")
    print(f"  - 系統已用記憶體: {stats.system_used:.0f}MB")
    print(f"  - 系統可用記憶體: {stats.system_available:.0f}MB")
    print(f"  - 程序記憶體使用: {stats.process_used:.0f}MB")
    print(f"  - 系統記憶體使用率: {stats.system_usage_ratio:.1%}")
    
    if stats.gpu_total > 0:
        print(f"  - GPU 總記憶體: {stats.gpu_total:.0f}MB")
        print(f"  - GPU 已用記憶體: {stats.gpu_used:.0f}MB")
        print(f"  - GPU 記憶體使用率: {stats.gpu_usage_ratio:.1%}")
    
    # 執行記憶體清理
    print("\n執行記憶體清理...")
    cleanup_result = await memory_optimizer.cleanup_memory()
    
    print("清理結果:")
    print(f"  - 清理耗時: {cleanup_result['cleanup_time']:.3f}s")
    print(f"  - 執行的清理動作: {len(cleanup_result['actions_taken'])}")
    
    for action in cleanup_result['actions_taken']:
        print(f"    * {action}")
    
    if 'memory_freed' in cleanup_result:
        freed = cleanup_result['memory_freed']
        total_freed = freed['system_mb'] + freed['gpu_mb']
        if total_freed > 0:
            print(f"  - 釋放記憶體: {total_freed:.1f}MB")


def demo_usage_monitoring(manager):
    """使用量監控示範"""
    print("\n=== 使用量監控示範 ===")
    
    # 取得使用量統計
    usage_stats = manager.get_usage_stats(time_range_hours=1)
    
    if usage_stats:
        summary = usage_stats['summary']
        print("使用量統計 (最近 1 小時):")
        print(f"  - 總請求數: {summary['total_requests']}")
        print(f"  - 成功請求數: {summary['successful_requests']}")
        print(f"  - 失敗請求數: {summary['failed_requests']}")
        print(f"  - 成功率: {summary['success_rate']:.1%}")
        print(f"  - 總處理時間: {summary['total_processing_time']:.2f}s")
        print(f"  - 平均處理時間: {summary['average_processing_time']:.3f}s")
        print(f"  - 平均吞吐量: {summary['average_throughput']:.1f} 項目/秒")
        
        # 模型分解
        if usage_stats['model_breakdown']:
            print("\n模型使用分解:")
            for model_name, model_stats in usage_stats['model_breakdown'].items():
                print(f"  - {model_name}:")
                print(f"    * 請求數: {model_stats['requests']}")
                print(f"    * 成功數: {model_stats['successful']}")
                print(f"    * 處理時間: {model_stats['processing_time']:.2f}s")
        
        # 裝置分解
        if usage_stats['device_breakdown']:
            print("\n裝置使用分解:")
            for device, count in usage_stats['device_breakdown'].items():
                print(f"  - {device}: {count} 次")
    else:
        print("無使用量統計資訊")
    
    # 檢查警報
    alerts = manager.get_usage_alerts(unresolved_only=True, limit=5)
    
    if alerts:
        print(f"\n未解決警報 ({len(alerts)} 個):")
        for alert in alerts:
            print(f"  - [{alert['level'].upper()}] {alert['model_name']}: {alert['message']}")
    else:
        print("\n✓ 無未解決警報")


async def demo_comprehensive_stats(manager):
    """綜合統計示範"""
    print("\n=== 綜合統計示範 ===")
    
    stats = manager.get_comprehensive_stats()
    
    # 管理器資訊
    manager_info = stats['manager_info']
    print("管理器資訊:")
    print(f"  - 總服務數: {manager_info['total_services']}")
    print(f"  - 已載入服務數: {manager_info['loaded_services']}")
    print(f"  - 預設服務: {manager_info['default_service']}")
    print(f"  - 啟用快取: {manager_info['enable_cache']}")
    print(f"  - 啟用 GPU 加速: {manager_info['enable_gpu_acceleration']}")
    print(f"  - 啟用監控: {manager_info['enable_monitoring']}")
    
    # 服務統計
    if 'services' in stats:
        print("\n服務統計:")
        for service_name, service_stats in stats['services'].items():
            print(f"  - {service_name}:")
            print(f"    * 模型: {service_stats['model_name']}")
            print(f"    * 總請求數: {service_stats['total_requests']}")
            print(f"    * 平均處理時間: {service_stats['average_processing_time']:.3f}s")
            print(f"    * 成功率: {service_stats['success_rate']:.1%}")
            print(f"    * 記憶體使用: {service_stats['memory_usage_mb']:.1f}MB")


async def main():
    """主函數"""
    print("🚀 Embedding 效能優化功能示範")
    print("=" * 50)
    
    try:
        # 基本使用示範
        manager = await demo_basic_usage()
        
        # 快取功能示範
        await demo_cache_functionality(manager)
        
        # 裝置管理示範
        demo_device_management()
        
        # 記憶體優化示範
        await demo_memory_optimization()
        
        # 使用量監控示範
        demo_usage_monitoring(manager)
        
        # 綜合統計示範
        await demo_comprehensive_stats(manager)
        
        print("\n" + "=" * 50)
        print("✅ 所有示範完成！")
        
        # 清理資源
        if manager.cache:
            print("\n🧹 清理快取...")
            await manager.clear_cache()
        
        if manager.memory_optimizer:
            print("🧹 最終記憶體清理...")
            await manager.cleanup_memory()
        
    except Exception as e:
        print(f"\n❌ 示範過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 執行示範
    asyncio.run(main())