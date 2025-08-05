"""
Embedding 模型效能評估模組

提供多維度的 embedding 模型效能評估功能
包括速度、品質、記憶體使用等指標
"""

import asyncio
import gc
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil 未安裝，記憶體監控功能將不可用")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import EmbeddingResult, EmbeddingService
from .manager import EmbeddingManager


@dataclass
class PerformanceMetrics:
    """效能評估指標"""

    model_name: str
    test_name: str

    # 速度指標
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    throughput_texts_per_second: float = 0.0

    # 記憶體指標
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_increase_mb: float = 0.0

    # GPU 指標（如果可用）
    gpu_memory_before_mb: float = 0.0
    gpu_memory_after_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_utilization_percent: float = 0.0

    # 品質指標
    embedding_dimension: int = 0
    embedding_norm_mean: float = 0.0
    embedding_norm_std: float = 0.0

    # 其他指標
    batch_size: int = 0
    text_count: int = 0
    success_rate: float = 1.0
    error_count: int = 0

    # 元數據
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    device: str = "unknown"
    test_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """基準測試配置"""

    # 測試文本配置
    text_lengths: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64])
    text_counts: List[int] = field(default_factory=lambda: [10, 50, 100, 500])

    # 測試配置
    warmup_iterations: int = 3
    test_iterations: int = 5
    memory_monitoring_interval: float = 0.1

    # 中文測試文本
    sample_texts: List[str] = field(
        default_factory=lambda: [
            "這是一個中文測試文本，用於評估 embedding 模型的效能。",
            "人工智慧技術在自然語言處理領域取得了顯著進展，特別是在文本理解和生成方面。",
            "深度學習模型如 BERT、GPT 等在各種 NLP 任務中表現出色，為實際應用提供了強大的技術支撐。",
            "知識圖譜結合檢索增強生成技術，能夠提供更準確、更可靠的問答系統，這對企業級應用具有重要意義。",
            "隨著計算能力的提升和演算法的優化，大型語言模型在處理複雜語言任務時展現出了前所未有的能力，但同時也帶來了計算成本和環境影響的挑戰。",
        ]
    )


class EmbeddingEvaluator:
    """Embedding 模型效能評估器

    提供全面的 embedding 模型效能評估功能
    """

    def __init__(
        self, config: Optional[BenchmarkConfig] = None, output_dir: Optional[str] = None
    ):
        """初始化評估器

        Args:
            config: 基準測試配置
            output_dir: 結果輸出目錄
        """
        self.config = config or BenchmarkConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)

        self.results: List[PerformanceMetrics] = []

        logger.info(f"初始化 Embedding 評估器，輸出目錄: {self.output_dir}")

    def _get_memory_usage(self) -> float:
        """取得當前記憶體使用量（MB）"""
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _get_gpu_memory_usage(self) -> Tuple[float, float]:
        """取得 GPU 記憶體使用量和利用率

        Returns:
            Tuple[float, float]: (memory_mb, utilization_percent)
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0, 0.0

        try:
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

            # 嘗試取得 GPU 利用率（需要 nvidia-ml-py）
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return memory_mb, float(utilization.gpu)
            except (ImportError, Exception):
                return memory_mb, 0.0

        except Exception as e:
            logger.debug(f"無法取得 GPU 資訊: {e}")
            return 0.0, 0.0

    def _generate_test_texts(self, count: int, length: int) -> List[str]:
        """生成測試文本

        Args:
            count: 文本數量
            length: 目標文本長度

        Returns:
            List[str]: 測試文本列表
        """
        texts = []
        base_texts = self.config.sample_texts

        for i in range(count):
            # 循環使用基礎文本
            base_text = base_texts[i % len(base_texts)]

            # 調整文本長度
            if len(base_text) >= length:
                text = base_text[:length]
            else:
                # 重複文本直到達到目標長度
                repeat_count = (length // len(base_text)) + 1
                text = (base_text + " ") * repeat_count
                text = text[:length]

            texts.append(text.strip())

        return texts

    async def evaluate_service(
        self, service: EmbeddingService, test_name: str = "default"
    ) -> List[PerformanceMetrics]:
        """評估單一 embedding 服務

        Args:
            service: embedding 服務
            test_name: 測試名稱

        Returns:
            List[PerformanceMetrics]: 評估結果列表
        """
        logger.info(f"開始評估 embedding 服務: {service.model_name}")

        if not service.is_loaded:
            await service.load_model()

        service_results = []

        # 暖身運行
        logger.info("執行暖身運行...")
        warmup_texts = self._generate_test_texts(10, 100)
        for _ in range(self.config.warmup_iterations):
            try:
                await service.embed_texts(warmup_texts)
            except Exception as e:
                logger.warning(f"暖身運行失敗: {e}")

        # 清理記憶體
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 執行各種測試場景
        test_scenarios = [
            ("batch_size", self._test_batch_sizes),
            ("text_length", self._test_text_lengths),
            ("text_count", self._test_text_counts),
        ]

        for scenario_name, test_func in test_scenarios:
            logger.info(f"執行 {scenario_name} 測試...")
            try:
                scenario_results = await test_func(
                    service, f"{test_name}_{scenario_name}"
                )
                service_results.extend(scenario_results)
            except Exception as e:
                logger.error(f"{scenario_name} 測試失敗: {e}")

        self.results.extend(service_results)
        logger.info(
            f"完成 {service.model_name} 評估，共 {len(service_results)} 個測試結果"
        )

        return service_results

    async def _test_batch_sizes(
        self, service: EmbeddingService, test_name: str
    ) -> List[PerformanceMetrics]:
        """測試不同批次大小的效能"""
        results = []

        for batch_size in self.config.batch_sizes:
            if batch_size > service.max_batch_size:
                continue

            logger.debug(f"測試批次大小: {batch_size}")

            # 生成測試文本
            texts = self._generate_test_texts(batch_size, 200)

            metrics = await self._run_performance_test(
                service=service,
                texts=texts,
                test_name=f"{test_name}_batch_{batch_size}",
                test_config={"batch_size": batch_size, "text_length": 200},
            )

            results.append(metrics)

        return results

    async def _test_text_lengths(
        self, service: EmbeddingService, test_name: str
    ) -> List[PerformanceMetrics]:
        """測試不同文本長度的效能"""
        results = []

        for text_length in self.config.text_lengths:
            if text_length > service.max_sequence_length:
                continue

            logger.debug(f"測試文本長度: {text_length}")

            # 生成測試文本
            texts = self._generate_test_texts(16, text_length)

            metrics = await self._run_performance_test(
                service=service,
                texts=texts,
                test_name=f"{test_name}_length_{text_length}",
                test_config={"batch_size": 16, "text_length": text_length},
            )

            results.append(metrics)

        return results

    async def _test_text_counts(
        self, service: EmbeddingService, test_name: str
    ) -> List[PerformanceMetrics]:
        """測試不同文本數量的效能"""
        results = []

        for text_count in self.config.text_counts:
            logger.debug(f"測試文本數量: {text_count}")

            # 生成測試文本
            texts = self._generate_test_texts(text_count, 200)

            metrics = await self._run_performance_test(
                service=service,
                texts=texts,
                test_name=f"{test_name}_count_{text_count}",
                test_config={
                    "batch_size": min(32, text_count),
                    "text_count": text_count,
                },
            )

            results.append(metrics)

        return results

    async def _run_performance_test(
        self,
        service: EmbeddingService,
        texts: List[str],
        test_name: str,
        test_config: Dict[str, Any],
    ) -> PerformanceMetrics:
        """執行單一效能測試

        Args:
            service: embedding 服務
            texts: 測試文本
            test_name: 測試名稱
            test_config: 測試配置

        Returns:
            PerformanceMetrics: 效能指標
        """
        metrics = PerformanceMetrics(
            model_name=service.model_name,
            test_name=test_name,
            text_count=len(texts),
            batch_size=test_config.get("batch_size", len(texts)),
            device=service.device or "unknown",
            test_config=test_config,
        )

        # 記錄初始狀態
        metrics.memory_before_mb = self._get_memory_usage()
        metrics.gpu_memory_before_mb, _ = self._get_gpu_memory_usage()

        processing_times = []
        error_count = 0
        embeddings_list = []

        # 執行多次測試取平均值
        for iteration in range(self.config.test_iterations):
            try:
                start_time = time.time()

                result = await service.embed_texts(texts, normalize=True)

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                if iteration == 0:  # 只在第一次記錄 embedding 資訊
                    embeddings_list.append(result.embeddings)
                    metrics.embedding_dimension = result.dimensions

            except Exception as e:
                error_count += 1
                logger.warning(f"測試迭代 {iteration + 1} 失敗: {e}")

        # 記錄最終狀態
        metrics.memory_after_mb = self._get_memory_usage()
        metrics.gpu_memory_after_mb, metrics.gpu_utilization_percent = (
            self._get_gpu_memory_usage()
        )

        # 計算統計指標
        if processing_times:
            metrics.total_processing_time = sum(processing_times)
            metrics.average_processing_time = np.mean(processing_times)
            metrics.throughput_texts_per_second = (
                len(texts) / metrics.average_processing_time
            )

        metrics.memory_increase_mb = metrics.memory_after_mb - metrics.memory_before_mb
        metrics.error_count = error_count
        metrics.success_rate = (
            self.config.test_iterations - error_count
        ) / self.config.test_iterations

        # 計算 embedding 品質指標
        if embeddings_list:
            embeddings = embeddings_list[0]
            norms = np.linalg.norm(embeddings, axis=1)
            metrics.embedding_norm_mean = float(np.mean(norms))
            metrics.embedding_norm_std = float(np.std(norms))

        return metrics

    async def compare_services(
        self, services: List[EmbeddingService], test_name: str = "comparison"
    ) -> Dict[str, List[PerformanceMetrics]]:
        """比較多個 embedding 服務的效能

        Args:
            services: embedding 服務列表
            test_name: 測試名稱

        Returns:
            Dict[str, List[PerformanceMetrics]]: 各服務的評估結果
        """
        logger.info(f"開始比較 {len(services)} 個 embedding 服務")

        comparison_results = {}

        for service in services:
            try:
                service_results = await self.evaluate_service(service, test_name)
                comparison_results[service.model_name] = service_results
            except Exception as e:
                logger.error(f"評估服務 {service.model_name} 失敗: {e}")
                comparison_results[service.model_name] = []

        return comparison_results

    def generate_report(
        self,
        results: Optional[List[PerformanceMetrics]] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成效能評估報告

        Args:
            results: 評估結果，如果為 None 則使用所有結果
            output_file: 輸出檔案名稱

        Returns:
            Dict[str, Any]: 報告內容
        """
        if results is None:
            results = self.results

        if not results:
            logger.warning("沒有評估結果可生成報告")
            return {}

        # 按模型分組結果
        results_by_model = {}
        for result in results:
            model_name = result.model_name
            if model_name not in results_by_model:
                results_by_model[model_name] = []
            results_by_model[model_name].append(result)

        # 生成報告
        report = {
            "evaluation_summary": {
                "total_tests": len(results),
                "models_tested": len(results_by_model),
                "evaluation_time": datetime.now().isoformat(),
            },
            "model_performance": {},
        }

        for model_name, model_results in results_by_model.items():
            # 計算模型的平均效能
            avg_processing_time = np.mean(
                [r.average_processing_time for r in model_results]
            )
            avg_throughput = np.mean(
                [r.throughput_texts_per_second for r in model_results]
            )
            avg_memory_increase = np.mean([r.memory_increase_mb for r in model_results])
            success_rate = np.mean([r.success_rate for r in model_results])

            model_summary = {
                "model_name": model_name,
                "test_count": len(model_results),
                "average_processing_time": round(avg_processing_time, 4),
                "average_throughput": round(avg_throughput, 2),
                "average_memory_increase_mb": round(avg_memory_increase, 2),
                "success_rate": round(success_rate, 3),
                "embedding_dimension": (
                    model_results[0].embedding_dimension if model_results else 0
                ),
                "device": model_results[0].device if model_results else "unknown",
            }

            # 詳細測試結果
            detailed_results = []
            for result in model_results:
                detailed_results.append(
                    {
                        "test_name": result.test_name,
                        "processing_time": round(result.average_processing_time, 4),
                        "throughput": round(result.throughput_texts_per_second, 2),
                        "memory_increase_mb": round(result.memory_increase_mb, 2),
                        "text_count": result.text_count,
                        "batch_size": result.batch_size,
                        "success_rate": round(result.success_rate, 3),
                        "test_config": result.test_config,
                    }
                )

            report["model_performance"][model_name] = {
                "summary": model_summary,
                "detailed_results": detailed_results,
            }

        # 儲存報告
        if output_file:
            output_path = self.output_dir / output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                self.output_dir / f"embedding_evaluation_report_{timestamp}.json"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"評估報告已儲存至: {output_path}")

        return report

    def get_best_model(
        self,
        metric: str = "throughput",
        results: Optional[List[PerformanceMetrics]] = None,
    ) -> Optional[str]:
        """根據指定指標找出最佳模型

        Args:
            metric: 評估指標 ('throughput', 'processing_time', 'memory', 'success_rate')
            results: 評估結果

        Returns:
            Optional[str]: 最佳模型名稱
        """
        if results is None:
            results = self.results

        if not results:
            return None

        # 按模型分組並計算平均值
        model_scores = {}

        for result in results:
            model_name = result.model_name
            if model_name not in model_scores:
                model_scores[model_name] = []

            if metric == "throughput":
                model_scores[model_name].append(result.throughput_texts_per_second)
            elif metric == "processing_time":
                model_scores[model_name].append(result.average_processing_time)
            elif metric == "memory":
                model_scores[model_name].append(result.memory_increase_mb)
            elif metric == "success_rate":
                model_scores[model_name].append(result.success_rate)
            else:
                raise ValueError(f"不支援的評估指標: {metric}")

        # 計算平均分數
        avg_scores = {model: np.mean(scores) for model, scores in model_scores.items()}

        # 根據指標選擇最佳模型
        if metric in ["throughput", "success_rate"]:
            # 越高越好
            best_model = max(avg_scores.items(), key=lambda x: x[1])[0]
        else:
            # 越低越好
            best_model = min(avg_scores.items(), key=lambda x: x[1])[0]

        logger.info(f"根據 {metric} 指標，最佳模型是: {best_model}")

        return best_model


class ChineseEmbeddingEvaluator(EmbeddingEvaluator):
    """中文 Embedding 專用評估器

    專門針對中文文本和 embedding 的品質評估
    """

    def __init__(
        self, config: Optional[BenchmarkConfig] = None, output_dir: Optional[str] = None
    ):
        """初始化中文 Embedding 評估器"""
        # 使用中文特化的測試配置
        if config is None:
            config = BenchmarkConfig(
                sample_texts=[
                    "這是一個中文測試文本，用於評估 embedding 模型的效能。",
                    "人工智慧技術在自然語言處理領域取得了顯著進展，特別是在文本理解和生成方面。",
                    "深度學習模型如 BERT、GPT 等在各種 NLP 任務中表現出色，為實際應用提供了強大的技術支撐。",
                    "知識圖譜結合檢索增強生成技術，能夠提供更準確、更可靠的問答系統，這對企業級應用具有重要意義。",
                    "隨著計算能力的提升和演算法的優化，大型語言模型在處理複雜語言任務時展現出了前所未有的能力。",
                    "中文自然語言處理面臨著分詞、語義理解、實體識別等多重挑戰，需要專門的技術解決方案。",
                    "在企業級應用中，文件管理和知識檢索系統需要高效準確的文本向量化技術支撐。",
                    "機器學習模型的訓練需要大量高品質的中文語料庫，這對模型效能至關重要。",
                    "跨語言的語義理解和翻譯技術正在快速發展，為全球化應用提供了新的可能性。",
                    "資訊檢索系統的核心在於準確理解用戶意圖並匹配最相關的內容，這需要先進的語義匹配技術。",
                ]
            )

        super().__init__(config, output_dir)

        # 中文特定的評估指標
        self.chinese_metrics = [
            "chinese_character_ratio",
            "semantic_similarity_coherence",
            "cross_domain_consistency",
            "length_robustness",
        ]

    async def evaluate_chinese_quality(
        self, service: EmbeddingService, test_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """評估中文 embedding 品質

        Args:
            service: embedding 服務
            test_texts: 測試文本，如果為 None 則使用預設文本

        Returns:
            Dict[str, Any]: 中文品質評估結果
        """
        if test_texts is None:
            test_texts = self.config.sample_texts

        logger.info(f"開始中文品質評估: {service.model_name}")

        try:
            # 確保服務已載入
            if not service.is_loaded:
                await service.load_model()

            # 執行向量化
            result = await service.embed_texts(test_texts, normalize=True)
            embeddings = result.embeddings

            # 計算中文特定指標
            chinese_metrics = await self._calculate_chinese_metrics(
                test_texts, embeddings, service
            )

            # 計算語義一致性
            semantic_metrics = await self._calculate_semantic_consistency(
                test_texts, embeddings, service
            )

            # 計算跨領域穩定性
            domain_metrics = await self._calculate_domain_stability(service)

            # 綜合評估
            overall_score = self._calculate_chinese_overall_score(
                chinese_metrics, semantic_metrics, domain_metrics
            )

            evaluation_result = {
                "model_name": service.model_name,
                "overall_score": overall_score,
                "chinese_metrics": chinese_metrics,
                "semantic_metrics": semantic_metrics,
                "domain_metrics": domain_metrics,
                "evaluation_time": datetime.now().isoformat(),
                "test_text_count": len(test_texts),
                "embedding_dimension": result.dimensions,
            }

            logger.info(
                f"中文品質評估完成: {service.model_name}, 總分: {overall_score:.3f}"
            )

            return evaluation_result

        except Exception as e:
            logger.error(f"中文品質評估失敗: {e}")
            return {
                "model_name": service.model_name,
                "error": str(e),
                "evaluation_time": datetime.now().isoformat(),
            }

    async def _calculate_chinese_metrics(
        self, texts: List[str], embeddings: np.ndarray, service: EmbeddingService
    ) -> Dict[str, float]:
        """計算中文特定指標"""
        metrics = {}

        # 1. 中文字符比例分析
        chinese_ratios = []
        for text in texts:
            chinese_chars = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
            total_chars = len(text)
            ratio = chinese_chars / total_chars if total_chars > 0 else 0
            chinese_ratios.append(ratio)

        metrics["chinese_char_ratio_mean"] = float(np.mean(chinese_ratios))
        metrics["chinese_char_ratio_std"] = float(np.std(chinese_ratios))

        # 2. 向量品質指標
        norms = np.linalg.norm(embeddings, axis=1)
        metrics["embedding_norm_mean"] = float(np.mean(norms))
        metrics["embedding_norm_std"] = float(np.std(norms))

        # 3. 向量分佈均勻性
        # 計算向量在高維空間中的分佈均勻性
        if len(embeddings) > 1:
            pairwise_distances = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    pairwise_distances.append(dist)

            metrics["pairwise_distance_mean"] = float(np.mean(pairwise_distances))
            metrics["pairwise_distance_std"] = float(np.std(pairwise_distances))

        return metrics

    async def _calculate_semantic_consistency(
        self, texts: List[str], embeddings: np.ndarray, service: EmbeddingService
    ) -> Dict[str, float]:
        """計算語義一致性指標"""
        metrics = {}

        # 1. 相似文本的向量相似度
        # 創建一些語義相似的文本對進行測試
        similar_pairs = [
            ("人工智慧技術發展迅速", "AI技術進步很快"),
            ("機器學習模型效能優秀", "ML模型表現良好"),
            ("自然語言處理很重要", "NLP技術非常關鍵"),
            ("深度學習應用廣泛", "深度學習用途很多"),
        ]

        similarity_scores = []

        for text1, text2 in similar_pairs:
            try:
                # 計算相似文本對的向量相似度
                result1 = await service.embed_texts([text1])
                result2 = await service.embed_texts([text2])

                vec1 = result1.embeddings[0]
                vec2 = result2.embeddings[0]

                # 餘弦相似度
                similarity = np.dot(vec1, vec2) / (
                    np.linalg.norm(vec1) * np.linalg.norm(vec2)
                )
                similarity_scores.append(float(similarity))

            except Exception as e:
                logger.warning(f"計算語義相似度失敗: {e}")

        if similarity_scores:
            metrics["semantic_similarity_mean"] = float(np.mean(similarity_scores))
            metrics["semantic_similarity_std"] = float(np.std(similarity_scores))

        # 2. 長短文本一致性
        # 測試相同內容的長短版本是否有一致的向量表示
        short_long_pairs = [
            ("AI很重要", "人工智慧技術在現代社會中扮演著非常重要的角色"),
            ("學習機器學習", "學習機器學習技術對於數據科學家來說是必不可少的技能"),
            ("文本處理", "自然語言文本處理是計算機科學中的一個重要分支領域"),
        ]

        length_consistency_scores = []

        for short_text, long_text in short_long_pairs:
            try:
                result_short = await service.embed_texts([short_text])
                result_long = await service.embed_texts([long_text])

                vec_short = result_short.embeddings[0]
                vec_long = result_long.embeddings[0]

                similarity = np.dot(vec_short, vec_long) / (
                    np.linalg.norm(vec_short) * np.linalg.norm(vec_long)
                )
                length_consistency_scores.append(float(similarity))

            except Exception as e:
                logger.warning(f"計算長度一致性失敗: {e}")

        if length_consistency_scores:
            metrics["length_consistency_mean"] = float(
                np.mean(length_consistency_scores)
            )
            metrics["length_consistency_std"] = float(np.std(length_consistency_scores))

        return metrics

    async def _calculate_domain_stability(
        self, service: EmbeddingService
    ) -> Dict[str, float]:
        """計算跨領域穩定性指標"""
        metrics = {}

        # 不同領域的測試文本
        domain_texts = {
            "technology": [
                "人工智慧技術發展迅速，深度學習模型在各個領域都有應用。",
                "機器學習演算法能夠從大量數據中學習模式和規律。",
                "自然語言處理技術使計算機能夠理解和生成人類語言。",
            ],
            "business": [
                "企業數位轉型需要採用先進的技術解決方案。",
                "市場競爭激烈，公司需要不斷創新以保持優勢。",
                "客戶滿意度是衡量企業成功的重要指標之一。",
            ],
            "education": [
                "教育改革應該注重培養學生的創新思維能力。",
                "線上學習平台為學生提供了更多學習機會。",
                "教師的專業發展對提高教學品質至關重要。",
            ],
        }

        domain_embeddings = {}

        # 為每個領域計算向量
        for domain, texts in domain_texts.items():
            try:
                result = await service.embed_texts(texts)
                domain_embeddings[domain] = result.embeddings
            except Exception as e:
                logger.warning(f"計算領域 {domain} 向量失敗: {e}")

        # 計算領域內一致性和領域間差異性
        if len(domain_embeddings) >= 2:
            intra_domain_similarities = []
            inter_domain_similarities = []

            # 領域內相似度
            for domain, embeddings in domain_embeddings.items():
                if len(embeddings) > 1:
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            sim = np.dot(embeddings[i], embeddings[j])
                            intra_domain_similarities.append(float(sim))

            # 領域間相似度
            domains = list(domain_embeddings.keys())
            for i in range(len(domains)):
                for j in range(i + 1, len(domains)):
                    domain1_embs = domain_embeddings[domains[i]]
                    domain2_embs = domain_embeddings[domains[j]]

                    for emb1 in domain1_embs:
                        for emb2 in domain2_embs:
                            sim = np.dot(emb1, emb2)
                            inter_domain_similarities.append(float(sim))

            if intra_domain_similarities:
                metrics["intra_domain_similarity_mean"] = float(
                    np.mean(intra_domain_similarities)
                )
                metrics["intra_domain_similarity_std"] = float(
                    np.std(intra_domain_similarities)
                )

            if inter_domain_similarities:
                metrics["inter_domain_similarity_mean"] = float(
                    np.mean(inter_domain_similarities)
                )
                metrics["inter_domain_similarity_std"] = float(
                    np.std(inter_domain_similarities)
                )

            # 領域區分度（領域內相似度應該高於領域間相似度）
            if intra_domain_similarities and inter_domain_similarities:
                domain_separation = np.mean(intra_domain_similarities) - np.mean(
                    inter_domain_similarities
                )
                metrics["domain_separation_score"] = float(domain_separation)

        return metrics

    def _calculate_chinese_overall_score(
        self,
        chinese_metrics: Dict[str, float],
        semantic_metrics: Dict[str, float],
        domain_metrics: Dict[str, float],
    ) -> float:
        """計算中文 embedding 的綜合評分"""

        score_components = []

        # 1. 中文字符處理能力 (0-1)
        chinese_ratio = chinese_metrics.get("chinese_char_ratio_mean", 0)
        chinese_score = min(chinese_ratio / 0.8, 1.0)  # 期望中文比例至少80%
        score_components.append(("chinese_handling", chinese_score, 0.2))

        # 2. 向量品質 (0-1)
        norm_mean = chinese_metrics.get("embedding_norm_mean", 0)
        norm_std = chinese_metrics.get("embedding_norm_std", 1)

        # 理想的向量範數應該接近1（正規化後），標準差應該較小
        norm_quality = max(0, 1 - abs(norm_mean - 1.0)) * max(0, 1 - norm_std)
        score_components.append(("vector_quality", norm_quality, 0.25))

        # 3. 語義一致性 (0-1)
        semantic_sim = semantic_metrics.get("semantic_similarity_mean", 0)
        # 語義相似的文本應該有較高的相似度（期望 > 0.7）
        semantic_score = min(max(semantic_sim, 0), 1.0)
        score_components.append(("semantic_consistency", semantic_score, 0.25))

        # 4. 長度穩定性 (0-1)
        length_consistency = semantic_metrics.get("length_consistency_mean", 0)
        length_score = min(max(length_consistency, 0), 1.0)
        score_components.append(("length_stability", length_score, 0.15))

        # 5. 領域區分能力 (0-1)
        domain_separation = domain_metrics.get("domain_separation_score", 0)
        # 正的領域區分分數表示好的區分能力
        domain_score = min(max(domain_separation, 0) / 0.2, 1.0)  # 正規化到0-1
        score_components.append(("domain_separation", domain_score, 0.15))

        # 計算加權總分
        total_score = sum(score * weight for _, score, weight in score_components)

        return round(total_score, 3)


async def quick_benchmark(
    services: List[EmbeddingService], output_dir: str = "benchmark_results"
) -> Dict[str, Any]:
    """快速基準測試的便利函數

    Args:
        services: embedding 服務列表
        output_dir: 輸出目錄

    Returns:
        Dict[str, Any]: 基準測試報告
    """
    # 使用較小的測試配置以加快速度
    quick_config = BenchmarkConfig(
        text_lengths=[100, 500],
        batch_sizes=[1, 16],
        text_counts=[10, 50],
        warmup_iterations=1,
        test_iterations=3,
    )

    evaluator = EmbeddingEvaluator(config=quick_config, output_dir=output_dir)

    # 執行比較測試
    results = await evaluator.compare_services(services, "quick_benchmark")

    # 生成報告
    report = evaluator.generate_report(output_file="quick_benchmark_report.json")

    return report


async def chinese_quality_benchmark(
    services: List[EmbeddingService], output_dir: str = "chinese_benchmark_results"
) -> Dict[str, Any]:
    """中文品質基準測試的便利函數

    Args:
        services: embedding 服務列表
        output_dir: 輸出目錄

    Returns:
        Dict[str, Any]: 中文基準測試報告
    """
    evaluator = ChineseEmbeddingEvaluator(output_dir=output_dir)

    results = {}

    # 對每個服務進行中文品質評估
    for service in services:
        try:
            result = await evaluator.evaluate_chinese_quality(service)
            results[service.model_name] = result
        except Exception as e:
            logger.error(f"評估服務 {service.model_name} 失敗: {e}")
            results[service.model_name] = {"error": str(e)}

    # 生成綜合報告
    report = {
        "evaluation_type": "chinese_quality_benchmark",
        "evaluation_time": datetime.now().isoformat(),
        "services_evaluated": len(services),
        "results": results,
    }

    # 找出最佳服務
    valid_results = {k: v for k, v in results.items() if "overall_score" in v}
    if valid_results:
        best_service = max(valid_results.items(), key=lambda x: x[1]["overall_score"])
        report["best_service"] = {
            "name": best_service[0],
            "score": best_service[1]["overall_score"],
        }

    # 儲存報告
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"chinese_quality_benchmark_{timestamp}.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"中文品質基準測試報告已儲存至: {report_file}")

    return report
