"""
GPU 加速和記憶體優化模組

提供 GPU 資源管理、記憶體優化策略和批次處理優化功能
支援 CUDA、MPS (Apple Silicon) 和多 GPU 環境
"""

import asyncio
import gc
import os
import psutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import threading
from contextlib import contextmanager

import numpy as np
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # 使用 debug 級別避免過多警告
    logger.debug("PyTorch 未安裝，GPU 加速將不可用")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    # 使用 debug 級別避免過多警告
    logger.debug("GPUtil 未安裝，GPU 監控功能受限")


@dataclass
class GPUInfo:
    """GPU 資訊資料結構"""
    device_id: int
    name: str
    memory_total: int  # MB
    memory_used: int   # MB
    memory_free: int   # MB
    utilization: float  # 0-100%
    temperature: Optional[float] = None  # 攝氏度
    power_usage: Optional[float] = None  # 瓦特
    
    @property
    def memory_usage_ratio(self) -> float:
        """記憶體使用率"""
        return self.memory_used / self.memory_total if self.memory_total > 0 else 0.0
    
    @property
    def is_available(self) -> bool:
        """是否可用（記憶體使用率 < 90%）"""
        return self.memory_usage_ratio < 0.9


@dataclass
class MemoryStats:
    """記憶體統計資訊"""
    system_total: int     # 系統總記憶體 (MB)
    system_used: int      # 系統已用記憶體 (MB)
    system_available: int # 系統可用記憶體 (MB)
    process_used: int     # 當前程序記憶體 (MB)
    gpu_total: int = 0    # GPU 總記憶體 (MB)
    gpu_used: int = 0     # GPU 已用記憶體 (MB)
    
    @property
    def system_usage_ratio(self) -> float:
        """系統記憶體使用率"""
        return self.system_used / self.system_total if self.system_total > 0 else 0.0
    
    @property
    def gpu_usage_ratio(self) -> float:
        """GPU 記憶體使用率"""
        return self.gpu_used / self.gpu_total if self.gpu_total > 0 else 0.0


class DeviceManager:
    """裝置管理器
    
    負責 GPU 裝置的偵測、選擇和資源管理
    """
    
    def __init__(self):
        self.available_devices: List[str] = []
        self.gpu_info: Dict[int, GPUInfo] = {}
        self._lock = threading.Lock()
        
        # 初始化裝置資訊
        self._detect_devices()
        
        logger.info(f"偵測到可用裝置: {self.available_devices}")
    
    def _detect_devices(self) -> None:
        """偵測可用裝置"""
        self.available_devices = ["cpu"]
        
        if not TORCH_AVAILABLE:
            return
        
        # 檢查 CUDA
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            for i in range(cuda_count):
                device_name = f"cuda:{i}"
                self.available_devices.append(device_name)
                logger.info(f"偵測到 CUDA 裝置: {device_name} - {torch.cuda.get_device_name(i)}")
        
        # 檢查 MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.available_devices.append("mps")
            logger.info("偵測到 MPS 裝置 (Apple Silicon)")
        
        # 更新 GPU 資訊
        self._update_gpu_info()
    
    def _update_gpu_info(self) -> None:
        """更新 GPU 資訊"""
        if not TORCH_AVAILABLE:
            return
        
        # CUDA 裝置資訊
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # 基本資訊
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory // 1024 // 1024  # 轉換為 MB
                    
                    # 記憶體使用情況
                    torch.cuda.set_device(i)
                    memory_allocated = torch.cuda.memory_allocated(i) // 1024 // 1024
                    memory_reserved = torch.cuda.memory_reserved(i) // 1024 // 1024
                    memory_free = memory_total - memory_reserved
                    
                    # 使用率（如果有 GPUtil）
                    utilization = 0.0
                    temperature = None
                    power_usage = None
                    
                    if GPUTIL_AVAILABLE:
                        try:
                            gpu = GPUtil.getGPUs()[i]
                            utilization = gpu.load * 100
                            temperature = gpu.temperature
                            power_usage = getattr(gpu, 'powerDraw', None)
                        except (IndexError, AttributeError):
                            pass
                    
                    self.gpu_info[i] = GPUInfo(
                        device_id=i,
                        name=props.name,
                        memory_total=memory_total,
                        memory_used=memory_reserved,
                        memory_free=memory_free,
                        utilization=utilization,
                        temperature=temperature,
                        power_usage=power_usage
                    )
                    
                except Exception as e:
                    logger.warning(f"無法取得 GPU {i} 資訊: {e}")
    
    def get_optimal_device(
        self,
        memory_required_mb: int = 0,
        prefer_gpu: bool = True
    ) -> str:
        """取得最佳裝置
        
        Args:
            memory_required_mb: 所需記憶體 (MB)
            prefer_gpu: 是否優先使用 GPU
            
        Returns:
            str: 最佳裝置名稱
        """
        with self._lock:
            self._update_gpu_info()
            
            if not prefer_gpu or not TORCH_AVAILABLE:
                return "cpu"
            
            # 尋找最佳 GPU
            best_device = "cpu"
            best_score = -1
            
            for device in self.available_devices:
                if device == "cpu":
                    continue
                
                if device.startswith("cuda:"):
                    gpu_id = int(device.split(":")[1])
                    if gpu_id in self.gpu_info:
                        gpu = self.gpu_info[gpu_id]
                        
                        # 檢查記憶體需求
                        if memory_required_mb > 0 and gpu.memory_free < memory_required_mb:
                            continue
                        
                        # 計算分數（考慮記憶體和使用率）
                        memory_score = gpu.memory_free / gpu.memory_total
                        utilization_score = (100 - gpu.utilization) / 100
                        score = memory_score * 0.6 + utilization_score * 0.4
                        
                        if score > best_score:
                            best_score = score
                            best_device = device
                
                elif device == "mps":
                    # MPS 裝置簡單檢查
                    if best_device == "cpu":
                        best_device = "mps"
            
            logger.debug(f"選擇裝置: {best_device} (分數: {best_score:.3f})")
            return best_device
    
    def get_device_info(self, device: str) -> Dict[str, Any]:
        """取得裝置詳細資訊"""
        if device == "cpu":
            return {
                "device": "cpu",
                "type": "CPU",
                "memory_info": self._get_cpu_memory_info()
            }
        
        if device.startswith("cuda:"):
            gpu_id = int(device.split(":")[1])
            if gpu_id in self.gpu_info:
                gpu = self.gpu_info[gpu_id]
                return {
                    "device": device,
                    "type": "CUDA",
                    "gpu_info": gpu,
                    "cuda_version": torch.version.cuda if TORCH_AVAILABLE else None
                }
        
        if device == "mps":
            return {
                "device": "mps",
                "type": "MPS",
                "available": torch.backends.mps.is_available() if TORCH_AVAILABLE else False
            }
        
        return {"device": device, "type": "Unknown"}
    
    def _get_cpu_memory_info(self) -> Dict[str, Any]:
        """取得 CPU 記憶體資訊"""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total // 1024 // 1024,
                "available_mb": memory.available // 1024 // 1024,
                "used_mb": memory.used // 1024 // 1024,
                "usage_percent": memory.percent
            }
        except Exception as e:
            logger.warning(f"無法取得 CPU 記憶體資訊: {e}")
            return {}
    
    def get_all_device_stats(self) -> Dict[str, Any]:
        """取得所有裝置統計資訊"""
        with self._lock:
            self._update_gpu_info()
            
            stats = {
                "available_devices": self.available_devices,
                "gpu_count": len(self.gpu_info),
                "devices": {}
            }
            
            for device in self.available_devices:
                stats["devices"][device] = self.get_device_info(device)
            
            return stats


class MemoryOptimizer:
    """記憶體優化器
    
    提供記憶體監控、清理和優化策略
    """
    
    def __init__(
        self,
        memory_threshold: float = 0.85,  # 記憶體使用率閾值
        cleanup_interval: int = 300,     # 清理間隔（秒）
        enable_auto_cleanup: bool = True
    ):
        """初始化記憶體優化器
        
        Args:
            memory_threshold: 記憶體使用率閾值
            cleanup_interval: 自動清理間隔
            enable_auto_cleanup: 是否啟用自動清理
        """
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        self.enable_auto_cleanup = enable_auto_cleanup
        
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_cleanup = False
        
        if enable_auto_cleanup:
            self._start_auto_cleanup()
        
        logger.info(f"初始化記憶體優化器，閾值: {memory_threshold:.1%}")
    
    def _start_auto_cleanup(self) -> None:
        """啟動自動清理任務"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._auto_cleanup_loop())
    
    async def _auto_cleanup_loop(self) -> None:
        """自動清理循環"""
        while not self._stop_cleanup:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                if self._should_cleanup():
                    logger.info("觸發自動記憶體清理")
                    await self.cleanup_memory()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自動清理失敗: {e}")
    
    def _should_cleanup(self) -> bool:
        """判斷是否需要清理"""
        stats = self.get_memory_stats()
        return (stats.system_usage_ratio > self.memory_threshold or
                stats.gpu_usage_ratio > self.memory_threshold)
    
    def get_memory_stats(self) -> MemoryStats:
        """取得記憶體統計資訊"""
        # 系統記憶體
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        stats = MemoryStats(
            system_total=memory.total // 1024 // 1024,
            system_used=memory.used // 1024 // 1024,
            system_available=memory.available // 1024 // 1024,
            process_used=process_memory.rss // 1024 // 1024
        )
        
        # GPU 記憶體
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    stats.gpu_total += props.total_memory // 1024 // 1024
                    stats.gpu_used += torch.cuda.memory_allocated(i) // 1024 // 1024
            except Exception as e:
                logger.warning(f"無法取得 GPU 記憶體資訊: {e}")
        
        return stats
    
    async def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """清理記憶體
        
        Args:
            aggressive: 是否進行積極清理
            
        Returns:
            Dict[str, Any]: 清理結果統計
        """
        start_time = time.time()
        before_stats = self.get_memory_stats()
        
        cleanup_results = {
            "before_cleanup": before_stats,
            "actions_taken": [],
            "cleanup_time": 0.0
        }
        
        try:
            # Python 垃圾回收
            collected = gc.collect()
            if collected > 0:
                cleanup_results["actions_taken"].append(f"Python GC: {collected} 個物件")
            
            # PyTorch 記憶體清理
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    # 清空 CUDA 快取
                    torch.cuda.empty_cache()
                    cleanup_results["actions_taken"].append("CUDA 快取清理")
                    
                    if aggressive:
                        # 積極清理：重置記憶體統計
                        torch.cuda.reset_peak_memory_stats()
                        cleanup_results["actions_taken"].append("CUDA 記憶體統計重置")
                
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # MPS 記憶體清理
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                        cleanup_results["actions_taken"].append("MPS 快取清理")
            
            # 強制垃圾回收（積極模式）
            if aggressive:
                for _ in range(3):
                    gc.collect()
                cleanup_results["actions_taken"].append("積極垃圾回收")
            
            # 等待一小段時間讓清理生效
            await asyncio.sleep(0.1)
            
            after_stats = self.get_memory_stats()
            cleanup_results["after_cleanup"] = after_stats
            cleanup_results["cleanup_time"] = time.time() - start_time
            
            # 計算清理效果
            system_freed = before_stats.system_used - after_stats.system_used
            gpu_freed = before_stats.gpu_used - after_stats.gpu_used
            
            cleanup_results["memory_freed"] = {
                "system_mb": system_freed,
                "gpu_mb": gpu_freed,
                "total_mb": system_freed + gpu_freed
            }
            
            logger.info(f"記憶體清理完成，釋放: 系統 {system_freed}MB, GPU {gpu_freed}MB, "
                       f"耗時: {cleanup_results['cleanup_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"記憶體清理失敗: {e}")
            cleanup_results["error"] = str(e)
        
        return cleanup_results
    
    @contextmanager
    def memory_monitor(self, operation_name: str = "操作"):
        """記憶體監控上下文管理器"""
        start_stats = self.get_memory_stats()
        start_time = time.time()
        
        logger.debug(f"{operation_name} 開始 - 記憶體使用: "
                    f"系統 {start_stats.system_used}MB, GPU {start_stats.gpu_used}MB")
        
        try:
            yield start_stats
        finally:
            end_stats = self.get_memory_stats()
            duration = time.time() - start_time
            
            system_delta = end_stats.system_used - start_stats.system_used
            gpu_delta = end_stats.gpu_used - start_stats.gpu_used
            
            logger.debug(f"{operation_name} 完成 - 記憶體變化: "
                        f"系統 {system_delta:+d}MB, GPU {gpu_delta:+d}MB, "
                        f"耗時: {duration:.2f}s")
    
    def stop_auto_cleanup(self) -> None:
        """停止自動清理"""
        self._stop_cleanup = True
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        logger.info("記憶體自動清理已停止")


class BatchProcessor:
    """批次處理優化器
    
    根據可用記憶體動態調整批次大小，優化處理效率
    """
    
    def __init__(
        self,
        device_manager: DeviceManager,
        memory_optimizer: MemoryOptimizer,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 256
    ):
        """初始化批次處理器
        
        Args:
            device_manager: 裝置管理器
            memory_optimizer: 記憶體優化器
            initial_batch_size: 初始批次大小
            min_batch_size: 最小批次大小
            max_batch_size: 最大批次大小
        """
        self.device_manager = device_manager
        self.memory_optimizer = memory_optimizer
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        # 動態批次大小追蹤
        self.current_batch_size = initial_batch_size
        self.batch_history: List[Tuple[int, float, bool]] = []  # (batch_size, processing_time, success)
        
        logger.info(f"初始化批次處理器，初始批次大小: {initial_batch_size}")
    
    def calculate_optimal_batch_size(
        self,
        device: str,
        estimated_memory_per_item: int = 10  # MB per item
    ) -> int:
        """計算最佳批次大小
        
        Args:
            device: 目標裝置
            estimated_memory_per_item: 每個項目預估記憶體使用量 (MB)
            
        Returns:
            int: 最佳批次大小
        """
        # 取得裝置資訊
        device_info = self.device_manager.get_device_info(device)
        memory_stats = self.memory_optimizer.get_memory_stats()
        
        # 計算可用記憶體
        if device == "cpu":
            available_memory = memory_stats.system_available
        elif device.startswith("cuda:"):
            gpu_id = int(device.split(":")[1])
            if gpu_id in self.device_manager.gpu_info:
                available_memory = self.device_manager.gpu_info[gpu_id].memory_free
            else:
                available_memory = 1000  # 預設值
        else:
            available_memory = 2000  # MPS 或其他裝置的預設值
        
        # 保留 20% 記憶體作為緩衝
        usable_memory = int(available_memory * 0.8)
        
        # 基於記憶體計算批次大小
        memory_based_batch_size = max(
            self.min_batch_size,
            min(self.max_batch_size, usable_memory // estimated_memory_per_item)
        )
        
        # 基於歷史效能調整
        performance_adjusted_size = self._adjust_based_on_history(memory_based_batch_size)
        
        # 最終批次大小
        optimal_size = max(
            self.min_batch_size,
            min(self.max_batch_size, performance_adjusted_size)
        )
        
        logger.debug(f"計算最佳批次大小: {optimal_size} "
                    f"(記憶體限制: {memory_based_batch_size}, "
                    f"效能調整: {performance_adjusted_size})")
        
        return optimal_size
    
    def _adjust_based_on_history(self, base_size: int) -> int:
        """基於歷史效能調整批次大小"""
        if len(self.batch_history) < 3:
            return base_size
        
        # 分析最近的批次處理結果
        recent_history = self.batch_history[-10:]  # 最近 10 次
        
        # 計算成功率
        success_rate = sum(1 for _, _, success in recent_history if success) / len(recent_history)
        
        # 如果成功率低，減小批次大小
        if success_rate < 0.8:
            adjustment_factor = 0.8
        # 如果成功率高且處理時間合理，可以嘗試增大批次大小
        elif success_rate > 0.95:
            avg_time = sum(time for _, time, success in recent_history if success) / max(1, sum(1 for _, _, success in recent_history if success))
            if avg_time < 10.0:  # 如果平均處理時間小於 10 秒
                adjustment_factor = 1.2
            else:
                adjustment_factor = 1.0
        else:
            adjustment_factor = 1.0
        
        adjusted_size = int(base_size * adjustment_factor)
        return max(self.min_batch_size, min(self.max_batch_size, adjusted_size))
    
    def record_batch_result(
        self,
        batch_size: int,
        processing_time: float,
        success: bool
    ) -> None:
        """記錄批次處理結果"""
        self.batch_history.append((batch_size, processing_time, success))
        
        # 保持歷史記錄在合理範圍內
        if len(self.batch_history) > 100:
            self.batch_history = self.batch_history[-50:]
        
        # 更新當前批次大小
        if success:
            self.current_batch_size = batch_size
        
        logger.debug(f"記錄批次結果: 大小={batch_size}, 時間={processing_time:.2f}s, "
                    f"成功={success}")
    
    async def process_in_batches(
        self,
        items: List[Any],
        process_func: Callable,
        device: str,
        estimated_memory_per_item: int = 10,
        show_progress: bool = False
    ) -> List[Any]:
        """批次處理資料
        
        Args:
            items: 要處理的項目列表
            process_func: 處理函數
            device: 目標裝置
            estimated_memory_per_item: 每個項目預估記憶體使用量
            show_progress: 是否顯示進度
            
        Returns:
            List[Any]: 處理結果列表
        """
        if not items:
            return []
        
        results = []
        total_items = len(items)
        processed_items = 0
        
        logger.info(f"開始批次處理 {total_items} 個項目")
        
        while processed_items < total_items:
            # 計算當前批次大小
            remaining_items = total_items - processed_items
            batch_size = min(
                remaining_items,
                self.calculate_optimal_batch_size(device, estimated_memory_per_item)
            )
            
            # 取得當前批次
            batch_start = processed_items
            batch_end = processed_items + batch_size
            current_batch = items[batch_start:batch_end]
            
            # 處理批次
            start_time = time.time()
            success = False
            
            try:
                with self.memory_optimizer.memory_monitor(f"批次處理 {batch_start}-{batch_end}"):
                    batch_result = await process_func(current_batch)
                    results.extend(batch_result)
                    success = True
                    processed_items += batch_size
                
                processing_time = time.time() - start_time
                self.record_batch_result(batch_size, processing_time, success)
                
                if show_progress:
                    progress = processed_items / total_items * 100
                    logger.info(f"批次處理進度: {progress:.1f}% "
                               f"({processed_items}/{total_items})")
                
            except Exception as e:
                processing_time = time.time() - start_time
                self.record_batch_result(batch_size, processing_time, success)
                
                logger.error(f"批次處理失敗: {e}")
                
                # 如果批次大小大於最小值，嘗試減小批次大小重試
                if batch_size > self.min_batch_size:
                    logger.info(f"嘗試減小批次大小重試: {batch_size} -> {batch_size // 2}")
                    self.current_batch_size = max(self.min_batch_size, batch_size // 2)
                    continue
                else:
                    # 無法進一步減小批次大小，跳過當前批次
                    logger.error(f"跳過失敗的批次: {batch_start}-{batch_end}")
                    processed_items += batch_size
            
            # 記憶體清理（如果需要）
            if self.memory_optimizer._should_cleanup():
                await self.memory_optimizer.cleanup_memory()
        
        logger.info(f"批次處理完成，處理了 {len(results)} 個結果")
        return results
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """取得批次處理統計資訊"""
        if not self.batch_history:
            return {
                "total_batches": 0,
                "current_batch_size": self.current_batch_size
            }
        
        successful_batches = [(size, time) for size, time, success in self.batch_history if success]
        failed_batches = [(size, time) for size, time, success in self.batch_history if not success]
        
        stats = {
            "total_batches": len(self.batch_history),
            "successful_batches": len(successful_batches),
            "failed_batches": len(failed_batches),
            "success_rate": len(successful_batches) / len(self.batch_history),
            "current_batch_size": self.current_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size
        }
        
        if successful_batches:
            batch_sizes = [size for size, _ in successful_batches]
            processing_times = [time for _, time in successful_batches]
            
            stats.update({
                "avg_batch_size": sum(batch_sizes) / len(batch_sizes),
                "avg_processing_time": sum(processing_times) / len(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times)
            })
        
        return stats


# 全域實例（單例模式）
_device_manager: Optional[DeviceManager] = None
_memory_optimizer: Optional[MemoryOptimizer] = None


def get_device_manager() -> DeviceManager:
    """取得全域裝置管理器實例"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_memory_optimizer() -> MemoryOptimizer:
    """取得全域記憶體優化器實例"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def create_batch_processor(**kwargs) -> BatchProcessor:
    """建立批次處理器"""
    device_manager = get_device_manager()
    memory_optimizer = get_memory_optimizer()
    
    return BatchProcessor(
        device_manager=device_manager,
        memory_optimizer=memory_optimizer,
        **kwargs
    )