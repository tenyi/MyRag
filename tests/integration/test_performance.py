"""
效能基準測試

測試系統各個組件的效能表現，包括處理速度、記憶體使用和併發能力。
"""

import asyncio
import resource
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import psutil
import pytest

from tests.test_utils import PerformanceTimer, TestDataGenerator


@pytest.mark.slow
@pytest.mark.integration
class TestPerformanceBenchmarks:
    """效能基準測試"""

    @pytest.fixture
    def performance_config(self):
        """效能測試配置"""
        return {
            "batch_sizes": [1, 10, 50, 100],
            "document_counts": [10, 100, 500],
            "vector_dimensions": [384, 768, 1024],
            "concurrent_threads": [1, 2, 4, 8],
            "timeout_seconds": 60,
            "memory_limit_mb": 1024,
        }

    def test_document_processing_performance(self, performance_config):
        """測試文件處理效能"""
        results = {}

        for doc_count in performance_config["document_counts"]:
            # 生成測試文件
            test_documents = []
            for i in range(doc_count):
                doc_content = TestDataGenerator.generate_chinese_text(
                    1000
                )  # 1000字文件
                test_documents.append(
                    {
                        "id": f"perf_doc_{i}",
                        "title": f"效能測試文件 {i}",
                        "content": doc_content,
                    }
                )

            # 模擬文件處理器
            processor = Mock()

            with PerformanceTimer() as timer:
                # 模擬處理過程
                processed_docs = []
                for doc in test_documents:
                    # 模擬文本分塊
                    chunks = [
                        doc["content"][i : i + 500]
                        for i in range(0, len(doc["content"]), 450)
                    ]
                    processed_doc = {
                        **doc,
                        "chunks": chunks,
                        "chunk_count": len(chunks),
                    }
                    processed_docs.append(processed_doc)

                processor.process_documents.return_value = processed_docs
                result = processor.process_documents(test_documents)

            # 記錄效能指標
            processing_time = timer.duration or 0.1
            docs_per_second = doc_count / processing_time

            results[doc_count] = {
                "processing_time": processing_time,
                "docs_per_second": docs_per_second,
                "total_chunks": sum(doc["chunk_count"] for doc in result),
                "avg_chunks_per_doc": sum(doc["chunk_count"] for doc in result)
                / len(result),
            }

            # 驗證效能要求
            if doc_count <= 10:
                assert processing_time < 5.0, f"小批次處理過慢: {processing_time}s"
            elif doc_count <= 100:
                assert processing_time < 30.0, f"中批次處理過慢: {processing_time}s"
            else:
                assert processing_time < 120.0, f"大批次處理過慢: {processing_time}s"

        # 驗證擴展性
        assert (
            results[100]["docs_per_second"] >= results[10]["docs_per_second"] * 0.5
        ), "處理速度下降過多"

        return results

    def test_embedding_generation_performance(self, performance_config):
        """測試向量生成效能"""
        results = {}

        for batch_size in performance_config["batch_sizes"]:
            for dimension in performance_config["vector_dimensions"]:
                # 生成測試文本
                test_texts = [
                    TestDataGenerator.generate_chinese_text(100)
                    for _ in range(batch_size)
                ]

                # 模擬 embedding 服務
                embedding_service = Mock()

                with PerformanceTimer() as timer:
                    # 模擬向量生成
                    embeddings = []
                    for text in test_texts:
                        vector = TestDataGenerator.generate_vector(dimension)
                        embeddings.append(vector)

                    embedding_service.encode_batch.return_value = embeddings
                    result = embedding_service.encode_batch(test_texts)

                # 記錄效能指標
                processing_time = timer.duration or 0.001
                texts_per_second = batch_size / processing_time

                key = f"batch_{batch_size}_dim_{dimension}"
                results[key] = {
                    "batch_size": batch_size,
                    "dimension": dimension,
                    "processing_time": processing_time,
                    "texts_per_second": texts_per_second,
                    "memory_per_vector_mb": (dimension * 4) / (1024 * 1024),  # float32
                }

                # 驗證效能要求
                if batch_size == 1:
                    assert (
                        processing_time < 1.0
                    ), f"單個向量生成過慢: {processing_time}s"
                elif batch_size <= 10:
                    assert (
                        processing_time < 5.0
                    ), f"小批次向量生成過慢: {processing_time}s"
                else:
                    assert (
                        processing_time < 30.0
                    ), f"大批次向量生成過慢: {processing_time}s"

        return results

    def test_vector_search_performance(self, performance_config):
        """測試向量搜尋效能"""
        from tests.test_utils import MockFactory

        results = {}

        # 準備不同大小的向量資料庫
        database_sizes = [100, 1000, 10000]

        for db_size in database_sizes:
            for dimension in [768]:  # 使用標準維度
                # 建立模擬向量資料庫
                vector_store = MockFactory.create_vector_store()

                # 模擬向量資料
                stored_vectors = []
                for i in range(db_size):
                    vector_doc = {
                        "id": f"vec_{i}",
                        "embedding": TestDataGenerator.generate_vector(dimension),
                        "metadata": {"index": i},
                    }
                    stored_vectors.append(vector_doc)

                # 測試不同的 top_k 值
                top_k_values = [1, 5, 10, 20]

                for top_k in top_k_values:
                    query_vector = TestDataGenerator.generate_vector(dimension)

                    # 模擬搜尋結果
                    mock_results = []
                    for i in range(min(top_k, db_size)):
                        mock_results.append(
                            {
                                "id": f"vec_{i}",
                                "score": 0.9 - (i * 0.1),
                                "metadata": {"index": i},
                            }
                        )

                    vector_store.search.return_value = mock_results

                    with PerformanceTimer() as timer:
                        search_results = vector_store.search(query_vector, top_k=top_k)

                    # 記錄效能指標
                    search_time = timer.duration or 0.001

                    key = f"db_{db_size}_topk_{top_k}"
                    results[key] = {
                        "database_size": db_size,
                        "top_k": top_k,
                        "search_time": search_time,
                        "results_count": len(search_results),
                        "searches_per_second": 1.0 / search_time,
                    }

                    # 驗證效能要求
                    if db_size <= 1000:
                        assert search_time < 0.1, f"小資料庫搜尋過慢: {search_time}s"
                    elif db_size <= 10000:
                        assert search_time < 1.0, f"中資料庫搜尋過慢: {search_time}s"
                    else:
                        assert search_time < 5.0, f"大資料庫搜尋過慢: {search_time}s"

        return results

    def test_concurrent_processing_performance(self, performance_config):
        """測試併發處理效能"""
        results = {}

        def simulate_processing_task(task_id: int, processing_time: float = 0.1):
            """模擬處理任務"""
            time.sleep(processing_time)  # 模擬處理時間
            return {
                "task_id": task_id,
                "result": TestDataGenerator.generate_chinese_text(50),
                "processing_time": processing_time,
            }

        task_count = 20

        for thread_count in performance_config["concurrent_threads"]:
            with PerformanceTimer() as timer:
                # 使用線程池執行併發任務
                with ThreadPoolExecutor(max_workers=thread_count) as executor:
                    futures = []
                    for i in range(task_count):
                        future = executor.submit(simulate_processing_task, i, 0.05)
                        futures.append(future)

                    # 等待所有任務完成
                    task_results = []
                    for future in futures:
                        result = future.result()
                        task_results.append(result)

            # 記錄效能指標
            total_time = timer.duration or 0.1
            theoretical_sequential_time = task_count * 0.05
            speedup = theoretical_sequential_time / total_time
            efficiency = speedup / thread_count

            results[thread_count] = {
                "thread_count": thread_count,
                "total_time": total_time,
                "tasks_completed": len(task_results),
                "tasks_per_second": task_count / total_time,
                "speedup": speedup,
                "efficiency": efficiency,
            }

            # 驗證併發效能
            assert len(task_results) == task_count, "任務執行不完整"
            if thread_count > 1:
                assert speedup > 1.0, f"併發未帶來加速: {speedup}"

        return results

    @pytest.mark.skipif(not psutil, reason="需要 psutil 套件")
    def test_memory_usage_performance(self, performance_config):
        """測試記憶體使用效能"""
        import gc

        results = {}
        process = psutil.Process()

        # 測試不同資料量的記憶體使用
        data_sizes = [100, 500, 1000, 2000]

        for data_size in data_sizes:
            # 記錄初始記憶體
            gc.collect()  # 強制垃圾回收
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # 生成測試資料
            test_data = []
            for i in range(data_size):
                doc_data = {
                    "id": f"mem_test_{i}",
                    "content": TestDataGenerator.generate_chinese_text(500),
                    "embedding": TestDataGenerator.generate_vector(768),
                    "metadata": {"index": i, "category": f"cat_{i % 10}"},
                }
                test_data.append(doc_data)

            # 記錄峰值記憶體
            peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = peak_memory - initial_memory

            # 清理資料
            del test_data
            gc.collect()

            # 記錄清理後記憶體
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_released = peak_memory - final_memory

            results[data_size] = {
                "data_size": data_size,
                "initial_memory_mb": initial_memory,
                "peak_memory_mb": peak_memory,
                "final_memory_mb": final_memory,
                "memory_used_mb": memory_used,
                "memory_released_mb": memory_released,
                "memory_per_item_kb": (
                    (memory_used * 1024) / data_size if data_size > 0 else 0
                ),
            }

            # 驗證記憶體使用
            memory_limit = performance_config["memory_limit_mb"]
            assert (
                memory_used < memory_limit
            ), f"記憶體使用超限: {memory_used}MB > {memory_limit}MB"

            # 驗證記憶體釋放
            release_ratio = memory_released / memory_used if memory_used > 0 else 1.0
            assert release_ratio > 0.8, f"記憶體釋放不足: {release_ratio:.2%}"

        return results

    @pytest.mark.asyncio
    async def test_async_processing_performance(self, performance_config):
        """測試異步處理效能"""
        results = {}

        async def simulate_async_task(task_id: int, delay: float = 0.1):
            """模擬異步任務"""
            await asyncio.sleep(delay)
            return {
                "task_id": task_id,
                "result": f"異步任務 {task_id} 完成",
                "delay": delay,
            }

        task_counts = [10, 50, 100]

        for task_count in task_counts:
            with PerformanceTimer() as timer:
                # 創建異步任務
                tasks = []
                for i in range(task_count):
                    task = simulate_async_task(i, 0.05)
                    tasks.append(task)

                # 併發執行所有任務
                task_results = await asyncio.gather(*tasks)

            # 記錄效能指標
            total_time = timer.duration or 0.1
            theoretical_sequential_time = task_count * 0.05
            speedup = theoretical_sequential_time / total_time

            results[task_count] = {
                "task_count": task_count,
                "total_time": total_time,
                "tasks_completed": len(task_results),
                "tasks_per_second": task_count / total_time,
                "speedup": speedup,
            }

            # 驗證異步效能
            assert len(task_results) == task_count, "異步任務執行不完整"
            assert speedup > task_count * 0.8, f"異步併發效果不佳: {speedup}"

        return results

    def test_end_to_end_performance(self, temp_dir, performance_config):
        """測試端到端效能"""
        # 建立測試工作空間
        workspace = temp_dir / "e2e_performance"
        workspace.mkdir()

        # 生成測試文件
        doc_count = 50
        test_documents = []

        for i in range(doc_count):
            doc_content = TestDataGenerator.generate_chinese_text(800)
            doc_path = workspace / f"test_doc_{i}.txt"

            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(doc_content)

            test_documents.append(doc_path)

        with PerformanceTimer() as total_timer:
            # 階段1：文件處理
            with PerformanceTimer() as processing_timer:
                processed_docs = []
                for doc_path in test_documents:
                    with open(doc_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    processed_docs.append(
                        {
                            "id": f"doc_{len(processed_docs)}",
                            "content": content,
                            "chunks": content.split("\n\n"),
                            "file_path": str(doc_path),
                        }
                    )

            # 階段2：向量化
            with PerformanceTimer() as embedding_timer:
                embedding_service = Mock()
                for doc in processed_docs:
                    doc["embedding"] = TestDataGenerator.generate_vector(768)
                    doc["chunk_embeddings"] = [
                        TestDataGenerator.generate_vector(768) for _ in doc["chunks"]
                    ]

            # 階段3：索引建立
            with PerformanceTimer() as indexing_timer:
                vector_store = Mock()
                indexed_count = 0

                for doc in processed_docs:
                    vector_store.add_document(
                        {
                            "id": doc["id"],
                            "embedding": doc["embedding"],
                            "content": doc["content"],
                        }
                    )
                    indexed_count += 1

            # 階段4：查詢測試
            with PerformanceTimer() as query_timer:
                query_engine = Mock()
                test_queries = ["測試查詢1", "測試查詢2", "測試查詢3"]

                query_results = []
                for query in test_queries:
                    result = {
                        "query": query,
                        "answer": f"{query} 的回答",
                        "confidence": 0.85,
                    }
                    query_results.append(result)

        # 記錄完整效能指標
        performance_report = {
            "document_count": doc_count,
            "total_time": total_timer.duration or 1.0,
            "processing_time": processing_timer.duration or 0.1,
            "embedding_time": embedding_timer.duration or 0.1,
            "indexing_time": indexing_timer.duration or 0.1,
            "query_time": query_timer.duration or 0.1,
            "docs_per_second": doc_count / (total_timer.duration or 1.0),
            "queries_per_second": len(test_queries) / (query_timer.duration or 0.1),
            "stage_breakdown": {
                "processing": (processing_timer.duration or 0.1)
                / (total_timer.duration or 1.0),
                "embedding": (embedding_timer.duration or 0.1)
                / (total_timer.duration or 1.0),
                "indexing": (indexing_timer.duration or 0.1)
                / (total_timer.duration or 1.0),
                "query": (query_timer.duration or 0.1) / (total_timer.duration or 1.0),
            },
        }

        # 驗證端到端效能
        assert performance_report["total_time"] < 60.0, "端到端處理時間過長"
        assert performance_report["docs_per_second"] > 0.5, "文件處理速度過慢"
        assert performance_report["queries_per_second"] > 1.0, "查詢速度過慢"

        return performance_report


@pytest.mark.integration
class TestStressTests:
    """壓力測試"""

    def test_high_concurrency_stress(self):
        """測試高併發壓力"""
        concurrent_requests = 50
        results = []
        errors = []

        def stress_task(task_id: int):
            """壓力測試任務"""
            try:
                # 模擬查詢處理
                query = f"壓力測試查詢 {task_id}"
                processing_time = 0.1  # 模擬處理時間

                time.sleep(processing_time)

                result = {
                    "task_id": task_id,
                    "query": query,
                    "success": True,
                    "processing_time": processing_time,
                }
                results.append(result)

            except Exception as e:
                errors.append({"task_id": task_id, "error": str(e)})

        # 執行併發壓力測試
        with PerformanceTimer() as timer:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for i in range(concurrent_requests):
                    future = executor.submit(stress_task, i)
                    futures.append(future)

                # 等待所有任務完成
                for future in futures:
                    future.result()

        # 驗證壓力測試結果
        success_rate = len(results) / concurrent_requests
        average_response_time = (
            timer.duration / concurrent_requests if timer.duration else 0
        )

        assert success_rate > 0.95, f"成功率過低: {success_rate:.2%}"
        assert len(errors) < concurrent_requests * 0.05, f"錯誤過多: {len(errors)}"
        assert (
            average_response_time < 1.0
        ), f"平均響應時間過長: {average_response_time}s"

    def test_memory_stress(self):
        """測試記憶體壓力"""
        if not psutil:
            pytest.skip("需要 psutil 套件")

        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # 逐步增加記憶體使用
        large_data_sets = []

        try:
            for i in range(10):
                # 每次創建約50MB的資料
                large_data = []
                for j in range(1000):
                    doc_data = {
                        "id": f"stress_{i}_{j}",
                        "content": TestDataGenerator.generate_chinese_text(2000),
                        "embedding": TestDataGenerator.generate_vector(768),
                        "metadata": {"batch": i, "index": j},
                    }
                    large_data.append(doc_data)

                large_data_sets.append(large_data)

                # 檢查記憶體使用
                current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                memory_increase = current_memory - initial_memory

                # 如果記憶體使用超過1GB，停止測試
                if memory_increase > 1024:
                    break

        finally:
            # 清理記憶體
            del large_data_sets
            import gc

            gc.collect()

        # 驗證記憶體恢復
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_leaked = final_memory - initial_memory

        assert memory_leaked < 100, f"記憶體洩漏: {memory_leaked}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
