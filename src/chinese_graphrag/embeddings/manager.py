"""
Embedding 服務管理器

負責管理多個 embedding 模型，提供統一的介面和智慧路由功能
整合快取、GPU 加速、記憶體優化和使用量監控功能
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Type
import time
import numpy as np

from loguru import logger

from .base import (
    EmbeddingService, 
    EmbeddingResult, 
    EmbeddingModelType,
    ModelMetrics,
    EmbeddingServiceError
)
from .cache import MultiLevelCache, create_embedding_cache
from .gpu_acceleration import (
    get_device_manager, 
    get_memory_optimizer, 
    create_batch_processor
)
from .monitoring import get_usage_monitor, record_embedding_usage


class EmbeddingManager:
    """Embedding 服務管理器
    
    管理多個 embedding 模型，提供模型選擇、負載均衡和降級處理
    整合快取、GPU 加速、記憶體優化和使用量監控功能
    """
    
    def __init__(
        self,
        default_model: Optional[str] = None,
        enable_fallback: bool = True,
        max_concurrent_requests: int = 10,
        enable_cache: bool = True,
        cache_config: Optional[Dict[str, Any]] = None,
        enable_gpu_acceleration: bool = True,
        enable_monitoring: bool = True
    ):
        """初始化 Embedding 管理器
        
        Args:
            default_model: 預設使用的模型名稱
            enable_fallback: 是否啟用降級處理
            max_concurrent_requests: 最大並發請求數
            enable_cache: 是否啟用快取
            cache_config: 快取配置
            enable_gpu_acceleration: 是否啟用 GPU 加速
            enable_monitoring: 是否啟用使用量監控
        """
        self.services: Dict[str, EmbeddingService] = {}
        self.default_model = default_model
        self.enable_fallback = enable_fallback
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._last_health_check = {}
        
        # 快取系統
        self.enable_cache = enable_cache
        if enable_cache:
            cache_config = cache_config or {}
            self.cache = create_embedding_cache("multi_level", **cache_config)
            logger.info("啟用 Embedding 快取系統")
        else:
            self.cache = None
        
        # GPU 加速和記憶體優化
        self.enable_gpu_acceleration = enable_gpu_acceleration
        if enable_gpu_acceleration:
            self.device_manager = get_device_manager()
            self.memory_optimizer = get_memory_optimizer()
            self.batch_processor = create_batch_processor()
            logger.info("啟用 GPU 加速和記憶體優化")
        else:
            self.device_manager = None
            self.memory_optimizer = None
            self.batch_processor = None
        
        # 使用量監控
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.usage_monitor = get_usage_monitor()
            logger.info("啟用使用量監控")
        else:
            self.usage_monitor = None
        
        logger.info(f"初始化 Embedding 管理器，預設模型: {default_model}")
    
    def register_service(
        self, 
        name: str, 
        service: EmbeddingService,
        set_as_default: bool = False
    ) -> None:
        """註冊 embedding 服務
        
        Args:
            name: 服務名稱
            service: embedding 服務實例
            set_as_default: 是否設為預設服務
        """
        self.services[name] = service
        logger.info(f"註冊 Embedding 服務: {name} ({service.model_type.value})")
        
        if set_as_default or self.default_model is None:
            self.default_model = name
            logger.info(f"設定預設 Embedding 服務: {name}")
    
    def unregister_service(self, name: str) -> None:
        """取消註冊 embedding 服務
        
        Args:
            name: 服務名稱
        """
        if name in self.services:
            del self.services[name]
            logger.info(f"取消註冊 Embedding 服務: {name}")
            
            # 如果移除的是預設服務，重新選擇預設服務
            if self.default_model == name:
                if self.services:
                    self.default_model = next(iter(self.services.keys()))
                    logger.info(f"重新設定預設 Embedding 服務: {self.default_model}")
                else:
                    self.default_model = None
                    logger.warning("沒有可用的 Embedding 服務")
    
    def get_service(self, name: Optional[str] = None) -> EmbeddingService:
        """取得 embedding 服務
        
        Args:
            name: 服務名稱，如果為 None 則使用預設服務
            
        Returns:
            EmbeddingService: embedding 服務實例
            
        Raises:
            ValueError: 當服務不存在時
        """
        service_name = name or self.default_model
        
        if not service_name:
            raise ValueError("沒有指定服務名稱且沒有預設服務")
        
        if service_name not in self.services:
            raise ValueError(f"Embedding 服務 '{service_name}' 不存在")
        
        return self.services[service_name]
    
    def list_services(self) -> List[Dict[str, Any]]:
        """列出所有已註冊的服務
        
        Returns:
            List[Dict[str, Any]]: 服務資訊列表
        """
        services_info = []
        for name, service in self.services.items():
            info = service.get_model_info()
            info['service_name'] = name
            info['is_default'] = (name == self.default_model)
            services_info.append(info)
        
        return services_info
    
    async def load_all_models(self) -> Dict[str, bool]:
        """載入所有註冊的模型
        
        Returns:
            Dict[str, bool]: 各模型載入結果
        """
        results = {}
        
        # 並行載入所有模型
        tasks = []
        for name, service in self.services.items():
            if not service.is_loaded:
                task = self._load_service_with_name(name, service)
                tasks.append(task)
        
        if tasks:
            load_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (name, result) in enumerate(zip(self.services.keys(), load_results)):
                if isinstance(result, Exception):
                    logger.error(f"載入模型 {name} 失敗: {result}")
                    results[name] = False
                else:
                    results[name] = result
        
        # 對於已載入的模型，直接標記為成功
        for name, service in self.services.items():
            if service.is_loaded and name not in results:
                results[name] = True
        
        loaded_count = sum(1 for success in results.values() if success)
        logger.info(f"模型載入完成: {loaded_count}/{len(results)} 成功")
        
        return results
    
    async def _load_service_with_name(self, name: str, service: EmbeddingService) -> bool:
        """載入指定服務（內部方法）"""
        try:
            await service.load_model()
            logger.info(f"成功載入模型: {name}")
            return True
        except Exception as e:
            logger.error(f"載入模型 {name} 失敗: {e}")
            return False
    
    async def unload_all_models(self) -> None:
        """卸載所有模型"""
        tasks = []
        for name, service in self.services.items():
            if service.is_loaded:
                task = self._unload_service_with_name(name, service)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("所有模型已卸載")
    
    async def _unload_service_with_name(self, name: str, service: EmbeddingService) -> None:
        """卸載指定服務（內部方法）"""
        try:
            await service.unload_model()
            logger.info(f"成功卸載模型: {name}")
        except Exception as e:
            logger.error(f"卸載模型 {name} 失敗: {e}")
    
    async def embed_texts(
        self,
        texts: List[str],
        service_name: Optional[str] = None,
        normalize: bool = True,
        show_progress: bool = False,
        fallback_on_error: bool = None,
        use_cache: bool = True,
        use_gpu_optimization: bool = True
    ) -> EmbeddingResult:
        """批次文本向量化
        
        Args:
            texts: 文本列表
            service_name: 使用的服務名稱
            normalize: 是否正規化向量
            show_progress: 是否顯示進度
            fallback_on_error: 是否在錯誤時降級，None 時使用管理器設定
            use_cache: 是否使用快取
            use_gpu_optimization: 是否使用 GPU 優化
            
        Returns:
            EmbeddingResult: 向量化結果
        """
        if fallback_on_error is None:
            fallback_on_error = self.enable_fallback
        
        # 選擇服務
        primary_service_name = service_name or self.default_model
        service = self.get_service(primary_service_name)
        
        # 檢查快取
        cache_key = None
        if self.enable_cache and use_cache and self.cache:
            cache_key = self.cache._generate_cache_key(
                texts, service.model_name, normalize
            )
            
            cached_entry = await self.cache.get(cache_key)
            if cached_entry:
                logger.debug(f"快取命中: {len(texts)} 個文本")
                
                # 記錄使用量（快取命中）
                if self.enable_monitoring:
                    record_embedding_usage(
                        model_name=service.model_name,
                        operation="embed_texts_cached",
                        input_count=len(texts),
                        processing_time=0.001,  # 快取存取時間很短
                        device="cache",
                        success=True,
                        cache_hit=True
                    )
                
                return EmbeddingResult(
                    embeddings=cached_entry.embeddings,
                    texts=cached_entry.texts,
                    model_name=cached_entry.model_name,
                    dimensions=cached_entry.embeddings.shape[1],
                    processing_time=0.001,
                    metadata={"cache_hit": True, **cached_entry.metadata}
                )
        
        # 記憶體優化
        if self.enable_gpu_acceleration and use_gpu_optimization and self.memory_optimizer:
            # 檢查記憶體使用情況
            if self.memory_optimizer._should_cleanup():
                logger.info("執行記憶體清理")
                await self.memory_optimizer.cleanup_memory()
        
        # 執行向量化
        start_time = time.time()
        success = False
        result = None
        error = None
        
        async with self._semaphore:
            try:
                # GPU 優化批次處理
                if (self.enable_gpu_acceleration and use_gpu_optimization and 
                    self.batch_processor and len(texts) > 32):
                    
                    # 使用批次處理器
                    device = self.device_manager.get_optimal_device(
                        memory_required_mb=len(texts) * 10  # 估算記憶體需求
                    )
                    
                    async def process_batch(batch_texts):
                        return await service.embed_texts(batch_texts, normalize, False)
                    
                    batch_results = await self.batch_processor.process_in_batches(
                        items=[texts],  # 包裝成批次
                        process_func=process_batch,
                        device=device,
                        estimated_memory_per_item=len(texts) * 10,
                        show_progress=show_progress
                    )
                    
                    if batch_results:
                        result = batch_results[0]
                        success = True
                else:
                    # 直接處理
                    result = await service.embed_texts(texts, normalize, show_progress)
                    success = True
                
                processing_time = time.time() - start_time
                
                # 儲存到快取
                if (self.enable_cache and use_cache and self.cache and 
                    cache_key and result and success):
                    
                    from .cache import CacheEntry
                    cache_entry = CacheEntry(
                        key=cache_key,
                        embeddings=result.embeddings,
                        texts=result.texts,
                        model_name=result.model_name,
                        timestamp=time.time()
                    )
                    
                    await self.cache.put(cache_key, cache_entry)
                    logger.debug(f"結果已快取: {len(texts)} 個文本")
                
                # 記錄使用量
                if self.enable_monitoring:
                    record_embedding_usage(
                        model_name=service.model_name,
                        operation="embed_texts",
                        input_count=len(texts),
                        processing_time=processing_time,
                        memory_used=getattr(result, 'memory_used', 0.0),
                        device=getattr(service, 'device', 'unknown'),
                        success=success,
                        output_dimensions=result.dimensions if result else 0,
                        cache_hit=False
                    )
                
                return result
                
            except Exception as e:
                error = e
                processing_time = time.time() - start_time
                logger.error(f"使用服務 {primary_service_name} 向量化失敗: {e}")
                
                # 記錄失敗的使用量
                if self.enable_monitoring:
                    record_embedding_usage(
                        model_name=service.model_name,
                        operation="embed_texts",
                        input_count=len(texts),
                        processing_time=processing_time,
                        device=getattr(service, 'device', 'unknown'),
                        success=False,
                        error_message=str(e)
                    )
                
                if not fallback_on_error:
                    raise
                
                # 嘗試降級到其他可用服務
                for name, fallback_service in self.services.items():
                    if name != primary_service_name and fallback_service.is_loaded:
                        try:
                            logger.info(f"降級使用服務: {name}")
                            fallback_start = time.time()
                            
                            result = await fallback_service.embed_texts(texts, normalize, show_progress)
                            fallback_time = time.time() - fallback_start
                            
                            # 記錄降級成功的使用量
                            if self.enable_monitoring:
                                record_embedding_usage(
                                    model_name=fallback_service.model_name,
                                    operation="embed_texts_fallback",
                                    input_count=len(texts),
                                    processing_time=fallback_time,
                                    device=getattr(fallback_service, 'device', 'unknown'),
                                    success=True,
                                    fallback_from=service.model_name
                                )
                            
                            return result
                            
                        except Exception as fallback_error:
                            logger.error(f"降級服務 {name} 也失敗: {fallback_error}")
                            
                            # 記錄降級失敗的使用量
                            if self.enable_monitoring:
                                record_embedding_usage(
                                    model_name=fallback_service.model_name,
                                    operation="embed_texts_fallback",
                                    input_count=len(texts),
                                    processing_time=time.time() - fallback_start,
                                    device=getattr(fallback_service, 'device', 'unknown'),
                                    success=False,
                                    error_message=str(fallback_error),
                                    fallback_from=service.model_name
                                )
                            continue
                
                # 所有服務都失敗
                raise EmbeddingServiceError(f"所有 embedding 服務都失敗，最後錯誤: {error}")
    
    async def embed_single_text(
        self,
        text: str,
        service_name: Optional[str] = None,
        normalize: bool = True,
        fallback_on_error: bool = None
    ) -> np.ndarray:
        """單一文本向量化
        
        Args:
            text: 單一文本
            service_name: 使用的服務名稱
            normalize: 是否正規化向量
            fallback_on_error: 是否在錯誤時降級
            
        Returns:
            np.ndarray: 向量結果
        """
        result = await self.embed_texts(
            [text], 
            service_name=service_name,
            normalize=normalize,
            fallback_on_error=fallback_on_error
        )
        return result.embeddings[0]
    
    async def compute_similarity(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]],
        service_name: Optional[str] = None,
        method: str = "cosine"
    ) -> Union[float, np.ndarray]:
        """計算文本相似度
        
        Args:
            texts1: 第一組文本
            texts2: 第二組文本
            service_name: 使用的服務名稱
            method: 相似度計算方法
            
        Returns:
            相似度分數或分數矩陣
        """
        service = self.get_service(service_name)
        return await service.compute_similarity(texts1, texts2, method)
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """檢查所有服務的健康狀態
        
        Returns:
            Dict[str, Dict[str, Any]]: 各服務的健康狀態
        """
        results = {}
        
        # 並行檢查所有服務
        tasks = []
        service_names = []
        
        for name, service in self.services.items():
            tasks.append(service.health_check())
            service_names.append(name)
        
        if tasks:
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for name, result in zip(service_names, health_results):
                if isinstance(result, Exception):
                    results[name] = {
                        "status": "error",
                        "error": str(result),
                        "model_name": self.services[name].model_name
                    }
                else:
                    results[name] = result
                
                # 更新最後檢查時間
                self._last_health_check[name] = datetime.now()
        
        return results
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """取得所有服務的效能指標摘要
        
        Returns:
            Dict[str, Any]: 效能指標摘要
        """
        summary = {
            "total_services": len(self.services),
            "loaded_services": sum(1 for s in self.services.values() if s.is_loaded),
            "default_service": self.default_model,
            "services": {}
        }
        
        for name, service in self.services.items():
            metrics = service.get_metrics()
            summary["services"][name] = {
                "model_name": metrics.model_name,
                "total_requests": metrics.total_requests,
                "average_processing_time": round(metrics.average_processing_time, 4),
                "success_rate": round(metrics.success_rate, 3),
                "error_count": metrics.error_count,
                "memory_usage_mb": round(metrics.memory_usage_mb, 1),
                "is_loaded": service.is_loaded
            }
        
        return summary
    
    def reset_all_metrics(self) -> None:
        """重置所有服務的效能指標"""
        for service in self.services.values():
            service.reset_metrics()
        logger.info("重置所有服務的效能指標")
    
    async def smart_route_request(
        self,
        texts: List[str],
        strategy: str = "fastest",
        **kwargs
    ) -> EmbeddingResult:
        """智慧路由請求到最適合的服務
        
        Args:
            texts: 文本列表
            strategy: 路由策略 ('fastest', 'least_loaded', 'round_robin')
            **kwargs: 其他參數傳遞給 embed_texts
            
        Returns:
            EmbeddingResult: 向量化結果
        """
        if not self.services:
            raise EmbeddingServiceError("沒有可用的 embedding 服務")
        
        # 只考慮已載入的服務
        available_services = {
            name: service for name, service in self.services.items() 
            if service.is_loaded
        }
        
        if not available_services:
            raise EmbeddingServiceError("沒有已載入的 embedding 服務")
        
        # 根據策略選擇服務
        if strategy == "fastest":
            # 選擇平均處理時間最短的服務
            best_service = min(
                available_services.items(),
                key=lambda x: x[1].metrics.average_processing_time or float('inf')
            )[0]
        elif strategy == "least_loaded":
            # 選擇請求數最少的服務
            best_service = min(
                available_services.items(),
                key=lambda x: x[1].metrics.total_requests
            )[0]
        elif strategy == "round_robin":
            # 輪詢選擇
            services_list = list(available_services.keys())
            # 簡單的輪詢實作（基於總請求數）
            total_requests = sum(s.metrics.total_requests for s in available_services.values())
            best_service = services_list[total_requests % len(services_list)]
        else:
            raise ValueError(f"不支援的路由策略: {strategy}")
        
        logger.debug(f"智慧路由選擇服務: {best_service} (策略: {strategy})")
        
        return await self.embed_texts(texts, service_name=best_service, **kwargs)
    
    async def __aenter__(self):
        """異步上下文管理器入口"""
        await self.load_all_models()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器出口"""
        await self.unload_all_models()
    
    # 新增的效能優化和管理方法
    
    async def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """取得快取統計資訊"""
        if self.cache:
            return await self.cache.get_stats()
        return None
    
    async def clear_cache(self) -> bool:
        """清空快取"""
        if self.cache:
            await self.cache.clear()
            logger.info("Embedding 快取已清空")
            return True
        return False
    
    async def preload_cache(
        self,
        texts: List[str],
        service_name: Optional[str] = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """快取預熱"""
        if not self.cache:
            return {"error": "快取未啟用"}
        
        service = self.get_service(service_name)
        return await self.cache.preload_cache(service, texts, batch_size)
    
    def get_device_stats(self) -> Optional[Dict[str, Any]]:
        """取得裝置統計資訊"""
        if self.device_manager:
            return self.device_manager.get_all_device_stats()
        return None
    
    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """取得記憶體統計資訊"""
        if self.memory_optimizer:
            stats = self.memory_optimizer.get_memory_stats()
            return {
                "system_total_mb": stats.system_total,
                "system_used_mb": stats.system_used,
                "system_available_mb": stats.system_available,
                "process_used_mb": stats.process_used,
                "gpu_total_mb": stats.gpu_total,
                "gpu_used_mb": stats.gpu_used,
                "system_usage_ratio": stats.system_usage_ratio,
                "gpu_usage_ratio": stats.gpu_usage_ratio
            }
        return None
    
    async def cleanup_memory(self, aggressive: bool = False) -> Optional[Dict[str, Any]]:
        """清理記憶體"""
        if self.memory_optimizer:
            return await self.memory_optimizer.cleanup_memory(aggressive)
        return None
    
    def get_batch_stats(self) -> Optional[Dict[str, Any]]:
        """取得批次處理統計資訊"""
        if self.batch_processor:
            return self.batch_processor.get_batch_stats()
        return None
    
    def get_usage_stats(
        self,
        time_range_hours: int = 24,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """取得使用量統計資訊"""
        if self.usage_monitor:
            return self.usage_monitor.get_usage_summary(time_range_hours, model_name)
        return None
    
    def get_usage_alerts(
        self,
        level: Optional[str] = None,
        model_name: Optional[str] = None,
        unresolved_only: bool = True,
        limit: int = 50
    ) -> Optional[List[Dict[str, Any]]]:
        """取得使用量警報"""
        if not self.usage_monitor:
            return None
        
        from .monitoring import AlertLevel
        alert_level = None
        if level:
            try:
                alert_level = AlertLevel(level.lower())
            except ValueError:
                logger.warning(f"無效的警報等級: {level}")
        
        alerts = self.usage_monitor.get_alerts(
            level=alert_level,
            model_name=model_name,
            unresolved_only=unresolved_only,
            limit=limit
        )
        
        # 轉換為字典格式
        return [
            {
                "timestamp": alert.timestamp,
                "level": alert.level.value,
                "model_name": alert.model_name,
                "message": alert.message,
                "details": alert.details,
                "resolved": alert.resolved,
                "resolved_timestamp": alert.resolved_timestamp
            }
            for alert in alerts
        ]
    
    async def export_usage_report(
        self,
        output_file: str,
        time_range_hours: int = 168,  # 一週
        format: str = "json"
    ) -> bool:
        """匯出使用量報告"""
        if self.usage_monitor:
            return self.usage_monitor.export_report(output_file, time_range_hours, format)
        return False
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """取得綜合統計資訊"""
        stats = {
            "manager_info": {
                "total_services": len(self.services),
                "loaded_services": sum(1 for s in self.services.values() if s.is_loaded),
                "default_service": self.default_model,
                "enable_fallback": self.enable_fallback,
                "enable_cache": self.enable_cache,
                "enable_gpu_acceleration": self.enable_gpu_acceleration,
                "enable_monitoring": self.enable_monitoring
            },
            "services": self.get_metrics_summary()["services"]
        }
        
        # 添加快取統計
        if self.enable_cache:
            cache_stats = asyncio.create_task(self.get_cache_stats())
            stats["cache_stats"] = "pending"  # 異步獲取
        
        # 添加裝置統計
        if self.enable_gpu_acceleration:
            stats["device_stats"] = self.get_device_stats()
            stats["memory_stats"] = self.get_memory_stats()
            stats["batch_stats"] = self.get_batch_stats()
        
        # 添加使用量統計
        if self.enable_monitoring:
            stats["usage_stats"] = self.get_usage_stats(24)  # 最近 24 小時
            stats["recent_alerts"] = self.get_usage_alerts(limit=10)
        
        return stats