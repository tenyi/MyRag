"""
效能優化使用範例

展示如何使用 Chinese GraphRAG 系統的效能優化功能
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from src.chinese_graphrag.performance import (
    OptimizerManager,
    OptimizationConfig,
    BatchOptimizer,
    QueryOptimizer,
    CostOptimizer,
    PerformanceMonitor
)


async def simulate_embedding_task(texts: List[str]) -> List[List[float]]:
    """模擬文本嵌入任務"""
    embeddings = []
    for text in texts:
        # 模擬嵌入計算時間
        await asyncio.sleep(0.01)
        # 生成假的嵌入向量
        embedding = [hash(text + str(i)) % 1000 / 1000.0 for i in range(384)]
        embeddings.append(embedding)
    return embeddings


async def simulate_query_task(query: str) -> Dict[str, Any]:
    """模擬查詢任務"""
    # 模擬查詢處理時間
    await asyncio.sleep(0.1)
    
    return {
        "query": query,
        "results": [
            {"text": f"結果 1 for {query}", "score": 0.95},
            {"text": f"結果 2 for {query}", "score": 0.87},
            {"text": f"結果 3 for {query}", "score": 0.82}
        ],
        "processing_time": 0.1
    }


async def simulate_llm_inference(prompt: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """模擬 LLM 推理任務"""
    # 模擬不同模型的處理時間
    processing_times = {
        "gpt-3.5-turbo": 0.5,
        "gpt-4": 1.0,
        "claude-3-sonnet": 0.8
    }
    
    await asyncio.sleep(processing_times.get(model, 0.5))
    
    return {
        "model": model,
        "prompt": prompt,
        "response": f"這是來自 {model} 的回應：{prompt[:50]}...",
        "input_tokens": len(prompt.split()),
        "output_tokens": 50,
        "processing_time": processing_times.get(model, 0.5)
    }


class PerformanceOptimizationDemo:
    """效能優化示範"""
    
    def __init__(self):
        """初始化示範"""
        # 建立優化配置
        self.config = OptimizationConfig(
            # 批次處理設定
            batch_enabled=True,
            batch_size=32,
            max_batch_size=128,
            parallel_workers=4,
            memory_threshold_mb=1024.0,
            
            # 查詢快取設定
            query_cache_enabled=True,
            cache_ttl_seconds=1800,  # 30分鐘
            cache_max_size=5000,
            preload_enabled=True,
            
            # 成本追蹤設定
            cost_tracking_enabled=True,
            budget_limit_usd=50.0,
            quality_threshold=0.85,
            
            # 效能監控設定
            monitoring_enabled=True,
            monitoring_interval=2.0,
            alert_thresholds={
                "cpu_usage": 75.0,
                "memory_usage": 80.0,
                "error_rate": 3.0
            },
            
            storage_path="logs/performance_demo"
        )
        
        self.optimizer_manager = None
    
    async def initialize(self):
        """初始化優化管理器"""
        print("🚀 初始化效能優化管理器...")
        self.optimizer_manager = OptimizerManager(self.config)
        await self.optimizer_manager.initialize()
        await self.optimizer_manager.start()
        print("✅ 效能優化管理器已啟動")
    
    async def demo_batch_optimization(self):
        """示範批次處理優化"""
        print("\n📦 批次處理優化示範")
        print("=" * 50)
        
        # 準備測試資料
        test_texts = [f"這是測試文本 {i}，用於演示批次處理優化功能。" for i in range(100)]
        
        # 不使用優化的處理
        print("⏱️  執行未優化的處理...")
        start_time = time.time()
        unoptimized_results = []
        for text in test_texts:
            result = await simulate_embedding_task([text])
            unoptimized_results.extend(result)
        unoptimized_time = time.time() - start_time
        
        # 使用批次優化的處理
        print("⚡ 執行批次優化處理...")
        start_time = time.time()
        optimized_results = await self.optimizer_manager.optimize_batch_processing(
            items=test_texts,
            process_func=lambda texts: simulate_embedding_task([texts])
        )
        optimized_time = time.time() - start_time
        
        # 比較結果
        speedup = unoptimized_time / optimized_time if optimized_time > 0 else 0
        print(f"📊 效能比較:")
        print(f"   未優化處理時間: {unoptimized_time:.2f}s")
        print(f"   批次優化時間: {optimized_time:.2f}s")
        print(f"   效能提升: {speedup:.2f}x")
        print(f"   處理項目數量: {len(test_texts)}")
    
    async def demo_query_optimization(self):
        """示範查詢優化"""
        print("\n🔍 查詢優化示範")
        print("=" * 50)
        
        test_queries = [
            "什麼是人工智慧？",
            "機器學習的基本概念",
            "深度學習與傳統機器學習的差異",
            "什麼是人工智慧？",  # 重複查詢，應該命中快取
            "自然語言處理的應用領域"
        ]
        
        total_time = 0
        cache_hits = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"🔎 執行查詢 {i}: {query}")
            
            start_time = time.time()
            result = await self.optimizer_manager.optimize_query(
                query=query,
                query_func=simulate_query_task
            )
            query_time = time.time() - start_time
            total_time += query_time
            
            # 檢查是否為快取命中（快速查詢通常表示快取命中）
            if query_time < 0.05:
                cache_hits += 1
                print(f"   ⚡ 快取命中! 查詢時間: {query_time:.3f}s")
            else:
                print(f"   🔄 執行查詢，查詢時間: {query_time:.3f}s")
            
            print(f"   📝 結果數量: {len(result['results'])}")
        
        print(f"\n📊 查詢統計:")
        print(f"   總查詢數: {len(test_queries)}")
        print(f"   快取命中數: {cache_hits}")
        print(f"   快取命中率: {cache_hits/len(test_queries)*100:.1f}%")
        print(f"   總查詢時間: {total_time:.2f}s")
        print(f"   平均查詢時間: {total_time/len(test_queries):.3f}s")
    
    async def demo_cost_optimization(self):
        """示範成本優化"""
        print("\n💰 成本優化示範")
        print("=" * 50)
        
        test_prompts = [
            "請總結以下文本的主要內容...",
            "分析這段文字的情感傾向...",
            "將以下內容翻譯成英文...",
            "回答關於機器學習的問題...",
            "生成一個創意故事..."
        ]
        
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"]
        
        for prompt in test_prompts:
            print(f"\n📝 處理提示: {prompt[:30]}...")
            
            # 取得模型建議
            recommendation = await self.optimizer_manager.optimize_model_usage(
                model_name="auto",
                input_tokens=len(prompt.split()),
                operation_type="text_generation"
            )
            
            recommended_model = recommendation.get("recommended_model", "gpt-3.5-turbo")
            print(f"   🎯 建議模型: {recommended_model}")
            
            # 執行推理
            result = await simulate_llm_inference(prompt, recommended_model)
            
            # 記錄實際使用情況
            if self.optimizer_manager.cost_optimizer:
                await self.optimizer_manager.cost_optimizer.track_usage(
                    model_name=recommended_model,
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    operation_type="text_generation"
                )
            
            print(f"   ⏱️  處理時間: {result['processing_time']:.2f}s")
            print(f"   📊 Token 使用: {result['input_tokens']} 輸入 + {result['output_tokens']} 輸出")
        
        # 顯示成本統計
        if self.optimizer_manager.cost_optimizer:
            cost_stats = self.optimizer_manager.cost_optimizer.get_usage_stats(60)
            print(f"\n💳 成本統計:")
            print(f"   總成本: ${cost_stats.get('total_cost', 0):.4f}")
            print(f"   模型使用次數: {len(cost_stats.get('model_usage', {}))}")
            
            for model, usage in cost_stats.get('model_usage', {}).items():
                print(f"   {model}: {usage.get('count', 0)} 次使用")
    
    async def demo_performance_monitoring(self):
        """示範效能監控"""
        print("\n📈 效能監控示範")
        print("=" * 50)
        
        # 執行一些工作負載以產生監控資料
        print("🔄 執行工作負載以產生監控資料...")
        
        # 模擬高負載工作
        tasks = []
        for i in range(20):
            task = asyncio.create_task(simulate_llm_inference(f"測試提示 {i}", "gpt-3.5-turbo"))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # 等待監控資料收集
        await asyncio.sleep(3)
        
        # 取得當前效能指標
        current_metrics = self.optimizer_manager.performance_monitor.get_current_metrics()
        if current_metrics:
            print(f"📊 當前效能指標:")
            print(f"   CPU 使用率: {current_metrics.cpu_usage:.1f}%")
            print(f"   記憶體使用: {current_metrics.memory_usage:.1f}MB")
            print(f"   磁碟使用率: {current_metrics.disk_usage:.1f}%")
            
            if current_metrics.gpu_usage > 0:
                print(f"   GPU 使用率: {current_metrics.gpu_usage:.1f}%")
                print(f"   GPU 記憶體: {current_metrics.gpu_memory_usage:.1f}MB")
        
        # 取得效能統計
        performance_stats = self.optimizer_manager.performance_monitor.get_performance_stats(5)
        if performance_stats:
            print(f"\n📈 效能統計 (過去5分鐘):")
            system_stats = performance_stats.get("system", {})
            
            if "cpu" in system_stats:
                cpu_stats = system_stats["cpu"]
                print(f"   CPU - 平均: {cpu_stats.get('avg', 0):.1f}%, "
                      f"最大: {cpu_stats.get('max', 0):.1f}%, "
                      f"P95: {cpu_stats.get('p95', 0):.1f}%")
            
            if "memory" in system_stats:
                memory_stats = system_stats["memory"]
                print(f"   記憶體 - 平均: {memory_stats.get('avg_mb', 0):.1f}MB, "
                      f"最大: {memory_stats.get('max_mb', 0):.1f}MB, "
                      f"P95: {memory_stats.get('p95_mb', 0):.1f}MB")
            
            trends = performance_stats.get("trends", {})
            if trends:
                print(f"   趨勢 - CPU: {trends.get('cpu_trend', 'unknown')}, "
                      f"記憶體: {trends.get('memory_trend', 'unknown')}, "
                      f"整體狀態: {trends.get('overall_status', 'unknown')}")
    
    async def demo_benchmark_testing(self):
        """示範基準測試"""
        print("\n🏁 基準測試示範")
        print("=" * 50)
        
        # 定義不同的測試場景
        test_configs = [
            {
                "name": "快速嵌入",
                "func": lambda: simulate_embedding_task(["短文本"]),
                "params": {}
            },
            {
                "name": "批量嵌入",
                "func": lambda: simulate_embedding_task(["文本1", "文本2", "文本3", "文本4", "文本5"]),
                "params": {}
            },
            {
                "name": "簡單查詢",
                "func": lambda: simulate_query_task("簡單問題"),
                "params": {}
            },
            {
                "name": "複雜查詢",
                "func": lambda: simulate_query_task("這是一個更複雜的查詢，需要更多處理時間和資源"),
                "params": {}
            }
        ]
        
        print("🚀 執行基準測試...")
        results = await self.optimizer_manager.run_performance_benchmark(
            test_configs=test_configs,
            iterations=10
        )
        
        print(f"\n🏆 基準測試結果:")
        for test_name, result in results.items():
            print(f"\n📋 {test_name}:")
            print(f"   吞吐量: {result.throughput:.2f} ops/s")
            print(f"   平均延遲: {result.latency_ms:.2f}ms")
            print(f"   成功率: {result.success_rate:.1%}")
            print(f"   記憶體峰值: {result.memory_peak_mb:.1f}MB")
            
            if result.latency_percentiles:
                print(f"   延遲分佈 - P50: {result.latency_percentiles.get('p50', 0):.2f}ms, "
                      f"P95: {result.latency_percentiles.get('p95', 0):.2f}ms, "
                      f"P99: {result.latency_percentiles.get('p99', 0):.2f}ms")
    
    async def generate_optimization_report(self):
        """生成優化報告"""
        print("\n📋 生成優化報告")
        print("=" * 50)
        
        report = self.optimizer_manager.get_optimization_report(duration_minutes=10)
        
        print(f"📊 優化報告 (過去 {report['period_minutes']} 分鐘):")
        print(f"   生成時間: {report['timestamp']}")
        
        summary = report.get("summary", {})
        print(f"\n📈 優化統計:")
        print(f"   批次優化次數: {summary.get('batch_optimizations', 0)}")
        print(f"   快取命中次數: {summary.get('cache_hits', 0)}")
        print(f"   快取未命中次數: {summary.get('cache_misses', 0)}")
        print(f"   成本節省: ${summary.get('cost_savings', 0):.4f}")
        
        # 儲存詳細報告
        report_path = Path("logs/performance_demo/optimization_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"💾 詳細報告已儲存至: {report_path}")
    
    async def cleanup(self):
        """清理資源"""
        if self.optimizer_manager:
            await self.optimizer_manager.stop()
        print("🧹 資源清理完成")


async def main():
    """主函數"""
    print("🎯 Chinese GraphRAG 效能優化示範")
    print("=" * 60)
    
    demo = PerformanceOptimizationDemo()
    
    try:
        # 初始化
        await demo.initialize()
        
        # 執行各種示範
        await demo.demo_batch_optimization()
        await demo.demo_query_optimization()
        await demo.demo_cost_optimization()
        await demo.demo_performance_monitoring()
        await demo.demo_benchmark_testing()
        
        # 生成報告
        await demo.generate_optimization_report()
        
        print("\n🎉 所有示範完成!")
        
    except Exception as e:
        print(f"❌ 示範執行錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理資源
        await demo.cleanup()


if __name__ == "__main__":
    # 執行示範
    asyncio.run(main())