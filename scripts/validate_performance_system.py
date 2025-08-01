#!/usr/bin/env python3
"""
效能優化系統驗證腳本

驗證所有效能優化模組的基本功能
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.chinese_graphrag.performance import (
        OptimizerManager,
        OptimizationConfig,
        BatchOptimizer,
        QueryOptimizer,
        CostOptimizer,
        PerformanceMonitor,
        BenchmarkRunner,
        ConfigLoader,
        load_performance_config
    )
    print("✅ 所有模組匯入成功")
except ImportError as e:
    print(f"❌ 模組匯入失敗: {e}")
    sys.exit(1)


async def test_batch_optimizer():
    """測試批次優化器"""
    print("\n🔄 測試批次優化器...")
    
    try:
        optimizer = BatchOptimizer(
            default_batch_size=8,
            max_batch_size=32,
            parallel_workers=2
        )
        
        # 模擬處理函數
        async def mock_process(item):
            await asyncio.sleep(0.01)
            return f"processed_{item}"
        
        # 測試批次處理
        test_items = list(range(20))
        results = await optimizer.process_batch(test_items, mock_process)
        
        assert len(results) == len(test_items)
        print("✅ 批次優化器測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 批次優化器測試失敗: {e}")
        return False


async def test_query_optimizer():
    """測試查詢優化器"""
    print("\n🔍 測試查詢優化器...")
    
    try:
        optimizer = QueryOptimizer(
            cache_ttl=300,
            max_cache_size=100
        )
        
        # 模擬查詢函數
        call_count = 0
        async def mock_query(query):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return f"result_for_{query}"
        
        query = "test_query"
        
        # 第一次查詢
        result1 = await optimizer.get_cached_result(query)
        if result1 is None:
            result1 = await mock_query(query)
            await optimizer.cache_result(query, result1)
        
        # 第二次查詢（應該從快取取得）
        result2 = await optimizer.get_cached_result(query)
        
        assert result1 == result2
        assert result2 is not None
        print("✅ 查詢優化器測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 查詢優化器測試失敗: {e}")
        return False


async def test_cost_optimizer():
    """測試成本優化器"""
    print("\n💰 測試成本優化器...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            optimizer = CostOptimizer(
                budget_limit=10.0,
                quality_threshold=0.8,
                storage_path=str(Path(temp_dir) / "cost_tracking.json")
            )
            
            # 測試使用追蹤
            await optimizer.track_usage(
                model_name="test-model",
                input_tokens=100,
                output_tokens=50,
                operation_type="test"
            )
            
            # 測試統計
            stats = optimizer.get_usage_stats(60)
            assert "total_cost" in stats
            
            print("✅ 成本優化器測試通過")
            return True
            
    except Exception as e:
        print(f"❌ 成本優化器測試失敗: {e}")
        return False


async def test_performance_monitor():
    """測試效能監控器"""
    print("\n📈 測試效能監控器...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(
                collection_interval=1.0,
                history_size=10,
                storage_path=temp_dir
            )
            
            # 啟動監控
            monitor.start_monitoring()
            
            # 等待收集一些資料
            await asyncio.sleep(2.5)
            
            # 檢查資料
            current_metrics = monitor.get_current_metrics()
            assert current_metrics is not None
            assert current_metrics.cpu_usage >= 0
            
            # 停止監控
            monitor.stop_monitoring()
            
            print("✅ 效能監控器測試通過")
            return True
            
    except Exception as e:
        print(f"❌ 效能監控器測試失敗: {e}")
        return False


async def test_benchmark_runner():
    """測試基準測試執行器"""
    print("\n🏁 測試基準測試執行器...")
    
    try:
        runner = BenchmarkRunner()
        
        # 定義測試函數
        async def test_function():
            await asyncio.sleep(0.01)
            return "test_result"
        
        # 執行基準測試
        result = await runner.run_benchmark(
            test_name="simple_test",
            test_func=test_function,
            test_params={},
            iterations=3
        )
        
        assert result.test_name == "simple_test"
        assert result.total_operations == 3
        assert result.throughput > 0
        
        print("✅ 基準測試執行器測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 基準測試執行器測試失敗: {e}")
        return False


async def test_optimizer_manager():
    """測試優化管理器"""
    print("\n🎯 測試優化管理器...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = OptimizationConfig(
                batch_enabled=True,
                batch_size=8,
                parallel_workers=2,
                query_cache_enabled=True,
                cache_ttl_seconds=300,
                cost_tracking_enabled=True,
                budget_limit_usd=10.0,
                monitoring_enabled=True,
                monitoring_interval=2.0,
                storage_path=temp_dir
            )
            
            async with OptimizerManager(config) as manager:
                # 測試批次處理
                async def mock_process(item):
                    return f"item_{item}"
                
                results = await manager.optimize_batch_processing(
                    items=[1, 2, 3],
                    process_func=mock_process
                )
                assert len(results) == 3
                
                # 測試查詢優化
                async def mock_query(query):
                    return f"result_{query}"
                
                result = await manager.optimize_query(
                    query="test",
                    query_func=mock_query
                )
                assert result == "result_test"
                
                # 測試狀態
                status = manager.get_performance_status()
                assert status["initialized"]
                assert status["running"]
                
                print("✅ 優化管理器測試通過")
                return True
                
    except Exception as e:
        print(f"❌ 優化管理器測試失敗: {e}")
        return False


def test_config_loader():
    """測試配置載入器"""
    print("\n⚙️ 測試配置載入器...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 建立測試配置檔案
            config_content = """
batch_optimization:
  enabled: true
  default_batch_size: 16
  parallel_workers: 2

query_optimization:
  cache_enabled: true
  cache_ttl_seconds: 600

cost_optimization:
  tracking_enabled: true
  budget_limit_usd: 50.0

performance_monitoring:
  enabled: true
  collection_interval: 5.0

storage:
  base_path: "test_logs"
"""
            
            config_path = Path(temp_dir) / "test_config.yaml"
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # 測試載入配置
            loader = ConfigLoader(str(config_path), "development")
            config = loader.load_config()
            
            assert config.optimization.batch_enabled
            assert config.optimization.batch_size == 16
            assert config.optimization.parallel_workers == 2
            
            print("✅ 配置載入器測試通過")
            return True
            
    except Exception as e:
        print(f"❌ 配置載入器測試失敗: {e}")
        return False


async def run_all_tests():
    """執行所有測試"""
    print("🚀 開始效能優化系統驗證")
    print("=" * 50)
    
    tests = [
        test_batch_optimizer,
        test_query_optimizer,
        test_cost_optimizer,
        test_performance_monitor,
        test_benchmark_runner,
        test_optimizer_manager,
    ]
    
    sync_tests = [
        test_config_loader,
    ]
    
    passed = 0
    total = len(tests) + len(sync_tests)
    
    # 執行異步測試
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"❌ 測試執行錯誤: {e}")
    
    # 執行同步測試
    for test in sync_tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 測試執行錯誤: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！效能優化系統驗證成功")
        return True
    else:
        print(f"⚠️  有 {total - passed} 個測試失敗")
        return False


def main():
    """主函數"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  測試被中斷")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 測試執行失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()