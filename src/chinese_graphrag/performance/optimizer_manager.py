"""
效能優化管理器

整合所有效能優化模組，提供統一的效能優化介面
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from .batch_optimizer import BatchOptimizer
from .query_optimizer import QueryOptimizer
from .cost_optimizer import CostOptimizer
from .performance_monitor import PerformanceMonitor, BenchmarkRunner


@dataclass
class OptimizationConfig:
    """效能優化配置"""
    
    # 批次處理優化
    batch_enabled: bool = True
    batch_size: int = 32
    max_batch_size: int = 128
    parallel_workers: int = 4
    memory_threshold_mb: float = 1024.0
    
    # 查詢優化
    query_cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000
    preload_enabled: bool = True
    
    # 成本優化
    cost_tracking_enabled: bool = True
    budget_limit_usd: Optional[float] = None
    quality_threshold: float = 0.8
    
    # 效能監控
    monitoring_enabled: bool = True
    monitoring_interval: float = 5.0
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "error_rate": 5.0
    })
    
    # 儲存設定
    storage_path: str = "logs/performance"
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "batch": {
                "enabled": self.batch_enabled,
                "batch_size": self.batch_size,
                "max_batch_size": self.max_batch_size,
                "parallel_workers": self.parallel_workers,
                "memory_threshold_mb": self.memory_threshold_mb
            },
            "query": {
                "cache_enabled": self.query_cache_enabled,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "cache_max_size": self.cache_max_size,
                "preload_enabled": self.preload_enabled
            },
            "cost": {
                "tracking_enabled": self.cost_tracking_enabled,
                "budget_limit_usd": self.budget_limit_usd,
                "quality_threshold": self.quality_threshold
            },
            "monitoring": {
                "enabled": self.monitoring_enabled,
                "interval": self.monitoring_interval,
                "alert_thresholds": self.alert_thresholds
            },
            "storage": {
                "path": self.storage_path
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """從字典建立配置"""
        config = cls()
        
        if "batch" in data:
            batch_config = data["batch"]
            config.batch_enabled = batch_config.get("enabled", config.batch_enabled)
            config.batch_size = batch_config.get("batch_size", config.batch_size)
            config.max_batch_size = batch_config.get("max_batch_size", config.max_batch_size)
            config.parallel_workers = batch_config.get("parallel_workers", config.parallel_workers)
            config.memory_threshold_mb = batch_config.get("memory_threshold_mb", config.memory_threshold_mb)
        
        if "query" in data:
            query_config = data["query"]
            config.query_cache_enabled = query_config.get("cache_enabled", config.query_cache_enabled)
            config.cache_ttl_seconds = query_config.get("cache_ttl_seconds", config.cache_ttl_seconds)
            config.cache_max_size = query_config.get("cache_max_size", config.cache_max_size)
            config.preload_enabled = query_config.get("preload_enabled", config.preload_enabled)
        
        if "cost" in data:
            cost_config = data["cost"]
            config.cost_tracking_enabled = cost_config.get("tracking_enabled", config.cost_tracking_enabled)
            config.budget_limit_usd = cost_config.get("budget_limit_usd", config.budget_limit_usd)
            config.quality_threshold = cost_config.get("quality_threshold", config.quality_threshold)
        
        if "monitoring" in data:
            monitoring_config = data["monitoring"]
            config.monitoring_enabled = monitoring_config.get("enabled", config.monitoring_enabled)
            config.monitoring_interval = monitoring_config.get("interval", config.monitoring_interval)
            config.alert_thresholds = monitoring_config.get("alert_thresholds", config.alert_thresholds)
        
        if "storage" in data:
            storage_config = data["storage"]
            config.storage_path = storage_config.get("path", config.storage_path)
        
        return config


class OptimizerManager:
    """效能優化管理器"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """初始化效能優化管理器
        
        Args:
            config: 優化配置
        """
        self.config = config or OptimizationConfig()
        
        # 初始化各個優化器
        self.batch_optimizer: Optional[BatchOptimizer] = None
        self.query_optimizer: Optional[QueryOptimizer] = None
        self.cost_optimizer: Optional[CostOptimizer] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.benchmark_runner: Optional[BenchmarkRunner] = None
        
        # 狀態追蹤
        self._initialized = False
        self._running = False
        
        # 統計資料
        self._optimization_stats = {
            "batch_optimizations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cost_savings": 0.0,
            "performance_improvements": []
        }
        
        logger.info("效能優化管理器初始化完成")
    
    async def initialize(self):
        """初始化所有優化器"""
        if self._initialized:
            logger.warning("效能優化管理器已初始化")
            return
        
        try:
            # 建立儲存目錄
            storage_path = Path(self.config.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # 初始化批次優化器
            if self.config.batch_enabled:
                from .batch_optimizer import BatchProcessingConfig
                batch_config = BatchProcessingConfig(
                    initial_batch_size=self.config.batch_size,
                    max_batch_size=self.config.max_batch_size,
                    max_workers=self.config.parallel_workers,
                    memory_limit_mb=int(self.config.memory_threshold_mb)
                )
                self.batch_optimizer = BatchOptimizer(batch_config)
                logger.info("批次優化器已初始化")
            
            # 初始化查詢優化器
            if self.config.query_cache_enabled:
                from .query_optimizer import QueryCacheConfig
                query_config = QueryCacheConfig(
                    memory_cache_ttl=self.config.cache_ttl_seconds,
                    memory_cache_size=self.config.cache_max_size,
                    enable_preloading=self.config.preload_enabled
                )
                self.query_optimizer = QueryOptimizer(query_config)
                logger.info("查詢優化器已初始化")
            
            # 初始化成本優化器
            if self.config.cost_tracking_enabled:
                from .cost_optimizer import ModelUsageTracker
                usage_tracker = ModelUsageTracker()
                self.cost_optimizer = CostOptimizer(usage_tracker)
                logger.info("成本優化器已初始化")
            
            # 初始化效能監控器
            if self.config.monitoring_enabled:
                self.performance_monitor = PerformanceMonitor(
                    collection_interval=self.config.monitoring_interval,
                    storage_path=str(storage_path)
                )
                
                # 設定警報閾值
                for metric, threshold in self.config.alert_thresholds.items():
                    self.performance_monitor.set_alert_threshold(
                        metric, threshold, self._handle_alert
                    )
                
                # 初始化基準測試執行器
                self.benchmark_runner = BenchmarkRunner(self.performance_monitor)
                
                logger.info("效能監控器已初始化")
            
            self._initialized = True
            logger.info("所有效能優化器初始化完成")
            
        except Exception as e:
            logger.error(f"效能優化器初始化失敗: {e}")
            raise
    
    async def start(self):
        """啟動效能優化"""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            logger.warning("效能優化已在執行中")
            return
        
        try:
            # 啟動效能監控
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
            
            self._running = True
            logger.info("效能優化已啟動")
            
        except Exception as e:
            logger.error(f"效能優化啟動失敗: {e}")
            raise
    
    async def stop(self):
        """停止效能優化"""
        if not self._running:
            return
        
        try:
            # 停止效能監控
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            self._running = False
            logger.info("效能優化已停止")
            
        except Exception as e:
            logger.error(f"效能優化停止失敗: {e}")
    
    async def _handle_alert(self, metric_name: str, value: float, threshold: float):
        """處理效能警報"""
        logger.warning(f"效能警報: {metric_name} = {value:.2f} (閾值: {threshold:.2f})")
        
        # 根據不同指標採取相應措施
        if metric_name == "memory_usage" and self.batch_optimizer:
            # 記憶體使用過高，減少批次大小
            with self.batch_optimizer._batch_size_lock:
                self.batch_optimizer.current_batch_size = max(
                    self.batch_optimizer.config.min_batch_size,
                    int(self.batch_optimizer.current_batch_size * 0.8)
                )
            logger.info(f"已調整批次大小以降低記憶體使用: {self.batch_optimizer.current_batch_size}")
        
        elif metric_name == "cpu_usage" and self.batch_optimizer:
            # CPU 使用過高，減少並行工作者
            current_workers = self.batch_optimizer.parallel_workers
            new_workers = max(1, int(current_workers * 0.8))
            self.batch_optimizer.config.max_workers = new_workers
            self.batch_optimizer.parallel_workers = new_workers
            logger.info(f"已調整並行工作者數量: {current_workers} -> {new_workers}")
    
    # 批次處理優化介面
    async def optimize_batch_processing(self, 
                                      items: List[Any],
                                      process_func: callable,
                                      **kwargs) -> List[Any]:
        """優化批次處理
        
        Args:
            items: 待處理項目
            process_func: 處理函數
            **kwargs: 額外參數
        
        Returns:
            處理結果
        """
        if not self.batch_optimizer:
            logger.warning("批次優化器未啟用，使用預設處理")
            return [await process_func(item, **kwargs) for item in items]
        
        # 包裝處理函數以適配批次處理
        async def batch_process_func(batch_items):
            """將單項處理函數包裝為批次處理函數"""
            results = []
            for item in batch_items:
                try:
                    if asyncio.iscoroutinefunction(process_func):
                        result = await process_func(item, **kwargs)
                    else:
                        result = process_func(item, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"處理項目失敗: {e}")
                    results.append(None)  # 或者根據需要處理錯誤
            return results
        
        # 使用異步批次處理
        result = await self.batch_optimizer.process_in_batches_async(
            items=items, 
            process_func=batch_process_func,
            **kwargs
        )
        self._optimization_stats["batch_optimizations"] += 1
        
        return result
    
    # 查詢優化介面
    async def optimize_query(self, 
                           query: str,
                           query_func: callable,
                           **kwargs) -> Any:
        """優化查詢處理
        
        Args:
            query: 查詢內容
            query_func: 查詢函數
            **kwargs: 額外參數
        
        Returns:
            查詢結果
        """
        if not self.query_optimizer:
            logger.warning("查詢優化器未啟用，使用預設處理")
            return await query_func(query, **kwargs)
        
        # 使用查詢優化器的 optimize_query 方法
        result = await self.query_optimizer.optimize_query(
            query=query,
            query_func=query_func,
            context=kwargs
        )
        
        # 更新統計資料
        if hasattr(self.query_optimizer, 'stats'):
            if self.query_optimizer.stats.cache_hits > 0:
                self._optimization_stats["cache_hits"] += 1
            else:
                self._optimization_stats["cache_misses"] += 1
        
        return result
    
    # 成本優化介面
    async def optimize_model_usage(self,
                                 model_name: str,
                                 input_tokens: int,
                                 operation_type: str = "inference") -> Dict[str, Any]:
        """優化模型使用
        
        Args:
            model_name: 模型名稱
            input_tokens: 輸入 token 數量
            operation_type: 操作類型
        
        Returns:
            優化建議
        """
        if not self.cost_optimizer:
            return {"recommendation": "cost_optimizer_disabled"}
        
        # 記錄使用情況 - 使用正確的方法名稱和參數
        if hasattr(self.cost_optimizer, 'usage_tracker') and self.cost_optimizer.usage_tracker:
            from .cost_optimizer import ModelType, TaskType
            task_type = TaskType.EMBEDDING if operation_type == "embedding" else TaskType.QUERY
            
            self.cost_optimizer.usage_tracker.record_usage(
                model_name=model_name,
                model_type=ModelType.LLM,  # 預設類型
                task_type=task_type,
                input_tokens=input_tokens,
                output_tokens=0,  # 將在實際使用後更新
                cost=0.0,  # 將根據實際使用計算
                latency_ms=0.0,  # 將在實際使用後更新
                quality_score=0.0,  # 將在實際使用後更新
                success=True
            )
        
        # 取得優化建議 - 使用實際存在的方法
        from .cost_optimizer import TaskType
        task_type = TaskType.EMBEDDING if operation_type == "embedding" else TaskType.QUERY
        
        try:
            optimal_model = self.cost_optimizer.select_optimal_model(
                task_type=task_type,
                input_size=input_tokens
            )
            recommendation = {
                "recommended_model": optimal_model,
                "reason": "cost_optimization"
            }
        except Exception as e:
            logger.warning(f"成本優化建議生成失敗: {e}")
            recommendation = {
                "recommended_model": model_name,
                "reason": "fallback_to_original"
            }
        
        return recommendation
    
    # 效能監控介面
    def get_performance_status(self) -> Dict[str, Any]:
        """取得效能狀態"""
        status = {
            "initialized": self._initialized,
            "running": self._running,
            "optimizers": {
                "batch_optimizer": self.batch_optimizer is not None,
                "query_optimizer": self.query_optimizer is not None,
                "cost_optimizer": self.cost_optimizer is not None,
                "performance_monitor": self.performance_monitor is not None
            },
            "statistics": self._optimization_stats.copy()
        }
        
        # 添加當前效能指標
        if self.performance_monitor:
            current_metrics = self.performance_monitor.get_current_metrics()
            if current_metrics:
                status["current_metrics"] = current_metrics.to_dict()
        
        return status
    
    def get_optimization_report(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """取得優化報告
        
        Args:
            duration_minutes: 報告時間範圍（分鐘）
        
        Returns:
            優化報告
        """
        report = {
            "period_minutes": duration_minutes,
            "timestamp": datetime.now().isoformat(),
            "summary": self._optimization_stats.copy()
        }
        
        # 添加效能統計
        if self.performance_monitor:
            performance_stats = self.performance_monitor.get_performance_stats(duration_minutes)
            report["performance"] = performance_stats
        
        # 添加成本統計
        if self.cost_optimizer:
            if hasattr(self.cost_optimizer, 'usage_tracker') and self.cost_optimizer.usage_tracker:
                cost_stats = self.cost_optimizer.usage_tracker.get_usage_stats(
                    period="today"
                )
            else:
                cost_stats = self.cost_optimizer.get_cost_analysis("today")
            report["cost"] = cost_stats
        
        # 添加快取統計
        if self.query_optimizer:
            cache_stats = self.query_optimizer.get_cache_stats()
            report["cache"] = cache_stats
        
        return report
    
    # 基準測試介面
    async def run_performance_benchmark(self,
                                      test_configs: List[Dict[str, Any]],
                                      iterations: int = 10) -> Dict[str, Any]:
        """執行效能基準測試
        
        Args:
            test_configs: 測試配置
            iterations: 迭代次數
        
        Returns:
            測試結果
        """
        if not self.benchmark_runner:
            raise RuntimeError("基準測試執行器未初始化")
        
        results = await self.benchmark_runner.run_comparative_benchmark(test_configs, iterations)
        
        # 記錄效能改進
        for test_name, result in results.items():
            self._optimization_stats["performance_improvements"].append({
                "test_name": test_name,
                "timestamp": datetime.now().isoformat(),
                "throughput": result.throughput,
                "latency_ms": result.latency_ms,
                "success_rate": result.success_rate
            })
        
        return results
    
    # 配置管理
    def save_config(self, config_path: str):
        """儲存配置
        
        Args:
            config_path: 配置檔案路徑
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"配置已儲存: {config_file}")
    
    @classmethod
    def load_config(cls, config_path: str) -> 'OptimizerManager':
        """載入配置
        
        Args:
            config_path: 配置檔案路徑
        
        Returns:
            優化管理器實例
        """
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"配置檔案不存在: {config_file}，使用預設配置")
            return cls()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        config = OptimizationConfig.from_dict(config_data)
        logger.info(f"配置已載入: {config_file}")
        
        return cls(config)
    

        
    async def __aenter__(self):
        """異步上下文管理器進入"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器退出"""
        await self.stop()