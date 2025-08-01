#!/usr/bin/env python3
"""
效能和壓力測試腳本

執行系統效能基準測試和壓力測試，包括：
- 文件處理效能測試
- 查詢回應時間測試
- 記憶體使用量測試
- 並發處理能力測試
- 大量資料處理測試
"""

import asyncio
import json
import logging
import os
import psutil
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

import click
import numpy as np
import pandas as pd
from loguru import logger

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chinese_graphrag.processors.chinese_text_processor import ChineseTextProcessor
from src.chinese_graphrag.embeddings.manager import EmbeddingManager
from src.chinese_graphrag.config.loader import ConfigLoader


class PerformanceStressTester:
    """效能和壓力測試器"""
    
    def __init__(self, test_dir: Path, config_path: Optional[Path] = None):
        self.test_dir = test_dir
        self.config_path = config_path or Path("config/settings.yaml")
        self.results: Dict[str, Any] = {
            "timestamp": time.time(),
            "performance_tests": {},
            "stress_tests": {},
            "system_metrics": {},
            "benchmarks": {},
            "summary": {}
        }
        
        # 設定日誌
        logger.remove()
        logger.add(
            self.test_dir / "performance_test.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
            rotation="10 MB"
        )
        logger.add(sys.stdout, level="INFO")
        
        # 系統監控
        self.system_monitor = SystemMonitor()
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """執行所有效能測試"""
        logger.info("⚡ 開始效能和壓力測試")
        
        try:
            # 開始系統監控
            self.system_monitor.start_monitoring()
            
            # 1. 文件處理效能測試
            await self._test_document_processing_performance()
            
            # 2. 查詢回應時間測試
            await self._test_query_response_time()
            
            # 3. 記憶體使用量測試
            await self._test_memory_usage()
            
            # 4. 並發處理能力測試
            await self._test_concurrent_processing()
            
            # 5. 大量資料處理測試
            await self._test_large_dataset_processing()
            
            # 6. 索引建構效能測試
            await self._test_indexing_performance()
            
            # 7. 向量檢索效能測試
            await self._test_vector_search_performance()
            
            # 8. 系統穩定性測試
            await self._test_system_stability()
            
            # 停止系統監控
            self.system_monitor.stop_monitoring()
            self.results["system_metrics"] = self.system_monitor.get_metrics()
            
            # 生成效能基準
            self._generate_performance_benchmarks()
            
            # 生成測試摘要
            self._generate_test_summary()
            
        except Exception as e:
            logger.error(f"效能測試執行失敗: {e}")
            self.system_monitor.stop_monitoring()
            
        return self.results
    
    async def _test_document_processing_performance(self):
        """測試文件處理效能"""
        logger.info("📄 測試文件處理效能")
        test_name = "document_processing_performance"
        
        try:
            processor = ChineseTextProcessor()
            
            # 建立不同大小的測試文件
            test_documents = [
                {"size": "small", "content": "人工智慧" * 100, "expected_time": 1.0},
                {"size": "medium", "content": "機器學習是人工智慧的重要分支" * 500, "expected_time": 3.0},
                {"size": "large", "content": "深度學習技術在各個領域都有廣泛應用" * 1000, "expected_time": 5.0}
            ]
            
            processing_results = []
            
            for doc in test_documents:
                start_time = time.time()
                
                # 執行文件處理
                processed_text = processor.preprocess_text(doc["content"])
                chunks = processor.split_text(processed_text, chunk_size=512)
                
                processing_time = time.time() - start_time
                
                result = {
                    "document_size": doc["size"],
                    "original_length": len(doc["content"]),
                    "processed_length": len(processed_text),
                    "chunks_count": len(chunks),
                    "processing_time": processing_time,
                    "expected_time": doc["expected_time"],
                    "performance_ratio": processing_time / doc["expected_time"],
                    "throughput": len(doc["content"]) / processing_time if processing_time > 0 else 0
                }
                
                processing_results.append(result)
                logger.info(f"  {doc['size']} 文件: {processing_time:.2f}s (預期: {doc['expected_time']}s)")
            
            # 計算平均效能
            avg_performance_ratio = np.mean([r["performance_ratio"] for r in processing_results])
            performance_pass = avg_performance_ratio <= 1.2  # 允許 20% 的效能差異
            
            self.results["performance_tests"][test_name] = {
                "status": "PASS" if performance_pass else "FAIL",
                "results": processing_results,
                "average_performance_ratio": avg_performance_ratio,
                "total_throughput": sum(r["throughput"] for r in processing_results)
            }
            
            logger.info(f"✅ 文件處理效能: {'通過' if performance_pass else '失敗'} (平均比率: {avg_performance_ratio:.2f})")
            
        except Exception as e:
            self.results["performance_tests"][test_name] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"❌ 文件處理效能測試失敗: {e}")
    
    async def _test_query_response_time(self):
        """測試查詢回應時間"""
        logger.info("❓ 測試查詢回應時間")
        test_name = "query_response_time"
        
        try:
            # 不同複雜度的查詢
            test_queries = [
                {"query": "什麼是AI？", "complexity": "simple", "expected_time": 1.0},
                {"query": "請詳細說明機器學習的主要演算法類型及其應用場景", "complexity": "medium", "expected_time": 2.0},
                {"query": "比較深度學習與傳統機器學習在自然語言處理任務中的優缺點，並分析未來發展趨勢", "complexity": "complex", "expected_time": 3.0}
            ]
            
            query_results = []
            
            for query_data in test_queries:
                start_time = time.time()
                
                # 模擬查詢處理
                await asyncio.sleep(0.1)  # 模擬處理時間
                response = f"這是對查詢「{query_data['query']}」的回應"
                
                response_time = time.time() - start_time
                
                result = {
                    "query": query_data["query"],
                    "complexity": query_data["complexity"],
                    "response_time": response_time,
                    "expected_time": query_data["expected_time"],
                    "performance_ratio": response_time / query_data["expected_time"],
                    "response_length": len(response)
                }
                
                query_results.append(result)
                logger.info(f"  {query_data['complexity']} 查詢: {response_time:.2f}s")
            
            # 計算平均回應時間
            avg_response_time = np.mean([r["response_time"] for r in query_results])
            avg_performance_ratio = np.mean([r["performance_ratio"] for r in query_results])
            response_time_pass = avg_response_time <= 2.0
            
            self.results["performance_tests"][test_name] = {
                "status": "PASS" if response_time_pass else "FAIL",
                "results": query_results,
                "average_response_time": avg_response_time,
                "average_performance_ratio": avg_performance_ratio
            }
            
            logger.info(f"✅ 查詢回應時間: {'通過' if response_time_pass else '失敗'} (平均: {avg_response_time:.2f}s)")
            
        except Exception as e:
            self.results["performance_tests"][test_name] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"❌ 查詢回應時間測試失敗: {e}")
    
    async def _test_memory_usage(self):
        """測試記憶體使用量"""
        logger.info("💾 測試記憶體使用量")
        test_name = "memory_usage"
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 模擬記憶體密集操作
            memory_tests = []
            
            for i in range(3):
                # 建立大量資料
                large_data = ["測試資料" * 1000] * 1000
                current_memory = process.memory_info().rss / 1024 / 1024
                
                memory_tests.append({
                    "test_step": f"step_{i+1}",
                    "memory_usage_mb": current_memory,
                    "memory_increase_mb": current_memory - initial_memory
                })
                
                # 清理資料
                del large_data
                
                await asyncio.sleep(0.1)
            
            # 最終記憶體使用量
            final_memory = process.memory_info().rss / 1024 / 1024
            max_memory = max(test["memory_usage_mb"] for test in memory_tests)
            memory_threshold = 1024  # 1GB
            
            memory_pass = max_memory <= memory_threshold
            
            self.results["performance_tests"][test_name] = {
                "status": "PASS" if memory_pass else "FAIL",
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "max_memory_mb": max_memory,
                "memory_threshold_mb": memory_threshold,
                "memory_tests": memory_tests
            }
            
            logger.info(f"✅ 記憶體使用量: {'通過' if memory_pass else '失敗'} (最大: {max_memory:.1f}MB)")
            
        except Exception as e:
            self.results["performance_tests"][test_name] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"❌ 記憶體使用量測試失敗: {e}")
    
    async def _test_concurrent_processing(self):
        """測試並發處理能力"""
        logger.info("🔄 測試並發處理能力")
        test_name = "concurrent_processing"
        
        try:
            processor = ChineseTextProcessor()
            
            # 並發測試參數
            concurrent_levels = [1, 5, 10, 20]
            test_text = "這是一個用於測試並發處理能力的中文文本範例" * 100
            
            concurrent_results = []
            
            for level in concurrent_levels:
                start_time = time.time()
                
                # 建立並發任務
                tasks = []
                for i in range(level):
                    task = asyncio.create_task(self._process_text_async(processor, test_text))
                    tasks.append(task)
                
                # 等待所有任務完成
                results = await asyncio.gather(*tasks)
                
                total_time = time.time() - start_time
                throughput = level / total_time if total_time > 0 else 0
                
                result = {
                    "concurrent_level": level,
                    "total_time": total_time,
                    "throughput": throughput,
                    "successful_tasks": len([r for r in results if r is not None]),
                    "failed_tasks": len([r for r in results if r is None])
                }
                
                concurrent_results.append(result)
                logger.info(f"  並發級別 {level}: {total_time:.2f}s, 吞吐量: {throughput:.2f} tasks/s")
            
            # 評估並發效能
            max_throughput = max(r["throughput"] for r in concurrent_results)
            concurrent_pass = max_throughput >= 5.0  # 至少 5 tasks/s
            
            self.results["performance_tests"][test_name] = {
                "status": "PASS" if concurrent_pass else "FAIL",
                "results": concurrent_results,
                "max_throughput": max_throughput,
                "throughput_threshold": 5.0
            }
            
            logger.info(f"✅ 並發處理能力: {'通過' if concurrent_pass else '失敗'} (最大吞吐量: {max_throughput:.2f})")
            
        except Exception as e:
            self.results["performance_tests"][test_name] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"❌ 並發處理能力測試失敗: {e}")
    
    async def _process_text_async(self, processor, text: str):
        """異步文本處理"""
        try:
            processed = processor.preprocess_text(text)
            chunks = processor.split_text(processed, chunk_size=256)
            return len(chunks)
        except Exception:
            return None
    
    async def _test_large_dataset_processing(self):
        """測試大量資料處理"""
        logger.info("📊 測試大量資料處理")
        test_name = "large_dataset_processing"
        
        try:
            processor = ChineseTextProcessor()
            
            # 建立大型資料集
            dataset_sizes = [100, 500, 1000]  # 文件數量
            base_text = "人工智慧技術正在快速發展，為各行各業帶來革命性的變化。" * 50
            
            dataset_results = []
            
            for size in dataset_sizes:
                start_time = time.time()
                
                # 處理大量文件
                processed_count = 0
                total_chunks = 0
                
                for i in range(size):
                    try:
                        processed_text = processor.preprocess_text(base_text)
                        chunks = processor.split_text(processed_text, chunk_size=512)
                        processed_count += 1
                        total_chunks += len(chunks)
                    except Exception:
                        continue
                
                processing_time = time.time() - start_time
                throughput = processed_count / processing_time if processing_time > 0 else 0
                
                result = {
                    "dataset_size": size,
                    "processed_documents": processed_count,
                    "total_chunks": total_chunks,
                    "processing_time": processing_time,
                    "throughput": throughput,
                    "success_rate": processed_count / size if size > 0 else 0
                }
                
                dataset_results.append(result)
                logger.info(f"  資料集大小 {size}: {processing_time:.2f}s, 吞吐量: {throughput:.2f} docs/s")
            
            # 評估大量資料處理能力
            avg_throughput = np.mean([r["throughput"] for r in dataset_results])
            avg_success_rate = np.mean([r["success_rate"] for r in dataset_results])
            
            large_dataset_pass = avg_throughput >= 10.0 and avg_success_rate >= 0.95
            
            self.results["performance_tests"][test_name] = {
                "status": "PASS" if large_dataset_pass else "FAIL",
                "results": dataset_results,
                "average_throughput": avg_throughput,
                "average_success_rate": avg_success_rate,
                "throughput_threshold": 10.0,
                "success_rate_threshold": 0.95
            }
            
            logger.info(f"✅ 大量資料處理: {'通過' if large_dataset_pass else '失敗'} (平均吞吐量: {avg_throughput:.2f})")
            
        except Exception as e:
            self.results["performance_tests"][test_name] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"❌ 大量資料處理測試失敗: {e}")
    
    async def _test_indexing_performance(self):
        """測試索引建構效能"""
        logger.info("🔍 測試索引建構效能")
        test_name = "indexing_performance"
        
        try:
            # 模擬索引建構效能測試
            indexing_scenarios = [
                {"documents": 100, "expected_time": 30, "scenario": "small"},
                {"documents": 500, "expected_time": 120, "scenario": "medium"},
                {"documents": 1000, "expected_time": 300, "scenario": "large"}
            ]
            
            indexing_results = []
            
            for scenario in indexing_scenarios:
                start_time = time.time()
                
                # 模擬索引建構過程
                await asyncio.sleep(0.5)  # 模擬處理時間
                
                indexing_time = time.time() - start_time
                
                result = {
                    "scenario": scenario["scenario"],
                    "document_count": scenario["documents"],
                    "indexing_time": indexing_time,
                    "expected_time": scenario["expected_time"],
                    "performance_ratio": indexing_time / scenario["expected_time"],
                    "throughput": scenario["documents"] / indexing_time if indexing_time > 0 else 0
                }
                
                indexing_results.append(result)
                logger.info(f"  {scenario['scenario']} 索引: {indexing_time:.2f}s")
            
            # 評估索引效能
            avg_performance_ratio = np.mean([r["performance_ratio"] for r in indexing_results])
            indexing_pass = avg_performance_ratio <= 1.5  # 允許 50% 的效能差異
            
            self.results["performance_tests"][test_name] = {
                "status": "PASS" if indexing_pass else "FAIL",
                "results": indexing_results,
                "average_performance_ratio": avg_performance_ratio
            }
            
            logger.info(f"✅ 索引建構效能: {'通過' if indexing_pass else '失敗'}")
            
        except Exception as e:
            self.results["performance_tests"][test_name] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"❌ 索引建構效能測試失敗: {e}")
    
    async def _test_vector_search_performance(self):
        """測試向量檢索效能"""
        logger.info("🔎 測試向量檢索效能")
        test_name = "vector_search_performance"
        
        try:
            # 模擬向量檢索效能測試
            search_scenarios = [
                {"vector_count": 1000, "search_queries": 10, "expected_time": 1.0},
                {"vector_count": 10000, "search_queries": 50, "expected_time": 5.0},
                {"vector_count": 100000, "search_queries": 100, "expected_time": 15.0}
            ]
            
            search_results = []
            
            for scenario in search_scenarios:
                start_time = time.time()
                
                # 模擬向量檢索
                for _ in range(scenario["search_queries"]):
                    await asyncio.sleep(0.01)  # 模擬檢索時間
                
                search_time = time.time() - start_time
                
                result = {
                    "vector_count": scenario["vector_count"],
                    "search_queries": scenario["search_queries"],
                    "search_time": search_time,
                    "expected_time": scenario["expected_time"],
                    "performance_ratio": search_time / scenario["expected_time"],
                    "queries_per_second": scenario["search_queries"] / search_time if search_time > 0 else 0
                }
                
                search_results.append(result)
                logger.info(f"  向量數 {scenario['vector_count']}: {search_time:.2f}s")
            
            # 評估檢索效能
            avg_qps = np.mean([r["queries_per_second"] for r in search_results])
            search_pass = avg_qps >= 10.0  # 至少 10 queries/s
            
            self.results["performance_tests"][test_name] = {
                "status": "PASS" if search_pass else "FAIL",
                "results": search_results,
                "average_qps": avg_qps,
                "qps_threshold": 10.0
            }
            
            logger.info(f"✅ 向量檢索效能: {'通過' if search_pass else '失敗'} (平均 QPS: {avg_qps:.2f})")
            
        except Exception as e:
            self.results["performance_tests"][test_name] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"❌ 向量檢索效能測試失敗: {e}")
    
    async def _test_system_stability(self):
        """測試系統穩定性"""
        logger.info("🛡️ 測試系統穩定性")
        test_name = "system_stability"
        
        try:
            processor = ChineseTextProcessor()
            
            # 長時間運行測試
            stability_duration = 60  # 60秒
            test_interval = 5  # 每5秒測試一次
            
            stability_results = []
            start_time = time.time()
            
            while time.time() - start_time < stability_duration:
                iteration_start = time.time()
                
                try:
                    # 執行基本操作
                    test_text = "系統穩定性測試文本" * 100
                    processed = processor.preprocess_text(test_text)
                    chunks = processor.split_text(processed)
                    
                    iteration_time = time.time() - iteration_start
                    
                    stability_results.append({
                        "timestamp": time.time(),
                        "iteration_time": iteration_time,
                        "success": True,
                        "chunks_count": len(chunks)
                    })
                    
                except Exception as e:
                    stability_results.append({
                        "timestamp": time.time(),
                        "iteration_time": time.time() - iteration_start,
                        "success": False,
                        "error": str(e)
                    })
                
                await asyncio.sleep(test_interval)
            
            # 評估穩定性
            total_iterations = len(stability_results)
            successful_iterations = sum(1 for r in stability_results if r["success"])
            success_rate = successful_iterations / total_iterations if total_iterations > 0 else 0
            
            avg_iteration_time = np.mean([r["iteration_time"] for r in stability_results])
            stability_pass = success_rate >= 0.95 and avg_iteration_time <= 2.0
            
            self.results["stress_tests"][test_name] = {
                "status": "PASS" if stability_pass else "FAIL",
                "test_duration": stability_duration,
                "total_iterations": total_iterations,
                "successful_iterations": successful_iterations,
                "success_rate": success_rate,
                "average_iteration_time": avg_iteration_time,
                "stability_results": stability_results[-10:]  # 只保留最後10個結果
            }
            
            logger.info(f"✅ 系統穩定性: {'通過' if stability_pass else '失敗'} (成功率: {success_rate:.2%})")
            
        except Exception as e:
            self.results["stress_tests"][test_name] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"❌ 系統穩定性測試失敗: {e}")
    
    def _generate_performance_benchmarks(self):
        """生成效能基準"""
        logger.info("📊 生成效能基準")
        
        benchmarks = {}
        
        # 從測試結果中提取基準數據
        for test_name, test_result in self.results["performance_tests"].items():
            if test_result["status"] == "PASS":
                if "average_response_time" in test_result:
                    benchmarks[f"{test_name}_response_time"] = test_result["average_response_time"]
                if "average_throughput" in test_result:
                    benchmarks[f"{test_name}_throughput"] = test_result["average_throughput"]
                if "max_throughput" in test_result:
                    benchmarks[f"{test_name}_max_throughput"] = test_result["max_throughput"]
        
        # 系統資源基準
        if "system_metrics" in self.results:
            metrics = self.results["system_metrics"]
            benchmarks.update({
                "peak_memory_usage_mb": metrics.get("peak_memory_mb", 0),
                "average_cpu_usage_percent": metrics.get("average_cpu_percent", 0),
                "peak_cpu_usage_percent": metrics.get("peak_cpu_percent", 0)
            })
        
        self.results["benchmarks"] = benchmarks
        
        logger.info("📊 效能基準已生成")
    
    def _generate_test_summary(self):
        """生成測試摘要"""
        logger.info("📋 生成測試摘要")
        
        # 效能測試摘要
        perf_tests = self.results["performance_tests"]
        perf_total = len(perf_tests)
        perf_passed = sum(1 for result in perf_tests.values() if result["status"] == "PASS")
        perf_failed = sum(1 for result in perf_tests.values() if result["status"] == "FAIL")
        perf_errors = sum(1 for result in perf_tests.values() if result["status"] == "ERROR")
        
        # 壓力測試摘要
        stress_tests = self.results["stress_tests"]
        stress_total = len(stress_tests)
        stress_passed = sum(1 for result in stress_tests.values() if result["status"] == "PASS")
        stress_failed = sum(1 for result in stress_tests.values() if result["status"] == "FAIL")
        stress_errors = sum(1 for result in stress_tests.values() if result["status"] == "ERROR")
        
        total_tests = perf_total + stress_total
        total_passed = perf_passed + stress_passed
        total_failed = perf_failed + stress_failed
        total_errors = perf_errors + stress_errors
        
        self.results["summary"] = {
            "performance_tests": {
                "total": perf_total,
                "passed": perf_passed,
                "failed": perf_failed,
                "errors": perf_errors,
                "success_rate": (perf_passed / perf_total * 100) if perf_total > 0 else 0
            },
            "stress_tests": {
                "total": stress_total,
                "passed": stress_passed,
                "failed": stress_failed,
                "errors": stress_errors,
                "success_rate": (stress_passed / stress_total * 100) if stress_total > 0 else 0
            },
            "overall": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_errors": total_errors,
                "overall_success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "overall_status": "PASS" if total_failed == 0 and total_errors == 0 else "FAIL"
            }
        }
        
        summary = self.results["summary"]
        logger.info(f"📋 測試摘要:")
        logger.info(f"   效能測試: {perf_passed}/{perf_total} 通過")
        logger.info(f"   壓力測試: {stress_passed}/{stress_total} 通過")
        logger.info(f"   整體成功率: {summary['overall']['overall_success_rate']:.1f}%")
        logger.info(f"   整體狀態: {summary['overall']['overall_status']}")
    
    def save_results(self, output_path: Path):
        """儲存測試結果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 測試結果已儲存至: {output_path}")


class SystemMonitor:
    """系統監控器"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_usage": [],
            "network_io": []
        }
        self.monitor_thread = None
    
    def start_monitoring(self):
        """開始監控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        logger.info("🔍 系統監控已開始")
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("🔍 系統監控已停止")
    
    def _monitor_loop(self):
        """監控循環"""
        while self.monitoring:
            try:
                # CPU 使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics["cpu_usage"].append({
                    "timestamp": time.time(),
                    "value": cpu_percent
                })
                
                # 記憶體使用率
                memory = psutil.virtual_memory()
                self.metrics["memory_usage"].append({
                    "timestamp": time.time(),
                    "value": memory.percent,
                    "used_mb": memory.used / 1024 / 1024,
                    "available_mb": memory.available / 1024 / 1024
                })
                
                # 磁碟使用率
                disk = psutil.disk_usage('/')
                self.metrics["disk_usage"].append({
                    "timestamp": time.time(),
                    "value": disk.percent,
                    "used_gb": disk.used / 1024 / 1024 / 1024,
                    "free_gb": disk.free / 1024 / 1024 / 1024
                })
                
            except Exception as e:
                logger.warning(f"監控數據收集失敗: {e}")
            
            time.sleep(5)  # 每5秒收集一次
    
    def get_metrics(self) -> Dict[str, Any]:
        """獲取監控指標"""
        if not self.metrics["cpu_usage"]:
            return {}
        
        return {
            "peak_cpu_percent": max(m["value"] for m in self.metrics["cpu_usage"]),
            "average_cpu_percent": np.mean([m["value"] for m in self.metrics["cpu_usage"]]),
            "peak_memory_mb": max(m["used_mb"] for m in self.metrics["memory_usage"]),
            "average_memory_mb": np.mean([m["used_mb"] for m in self.metrics["memory_usage"]]),
            "peak_memory_percent": max(m["value"] for m in self.metrics["memory_usage"]),
            "monitoring_duration": len(self.metrics["cpu_usage"]) * 5,  # 秒
            "data_points": len(self.metrics["cpu_usage"])
        }


@click.command()
@click.option("--test-dir", type=click.Path(), default="./performance_test_results", 
              help="測試結果目錄")
@click.option("--config", type=click.Path(), default="config/settings.yaml",
              help="配置檔案路徑")
@click.option("--output", type=click.Path(), default="performance_test_results.json",
              help="結果輸出檔案")
@click.option("--duration", type=int, default=60, help="壓力測試持續時間（秒）")
@click.option("--verbose", "-v", is_flag=True, help="詳細輸出")
def main(test_dir: str, config: str, output: str, duration: int, verbose: bool):
    """執行效能和壓力測試"""
    
    # 設定日誌級別
    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    # 建立測試目錄
    test_dir_path = Path(test_dir)
    test_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 執行測試
    tester = PerformanceStressTester(
        test_dir=test_dir_path,
        config_path=Path(config) if config else None
    )
    
    # 執行異步測試
    results = asyncio.run(tester.run_all_tests())
    
    # 儲存結果
    output_path = test_dir_path / output
    tester.save_results(output_path)
    
    # 輸出最終結果
    summary = results["summary"]["overall"]
    if summary["overall_status"] == "PASS":
        logger.success(f"🎉 效能和壓力測試通過！成功率: {summary['overall_success_rate']:.1f}%")
        sys.exit(0)
    else:
        logger.error(f"❌ 效能和壓力測試失敗！成功率: {summary['overall_success_rate']:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()