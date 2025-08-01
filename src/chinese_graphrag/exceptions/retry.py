"""
重試機制實作

提供靈活的重試策略和裝飾器，支援同步和非同步操作。
"""

import asyncio
import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from datetime import datetime, timedelta

from .base import ChineseGraphRAGError, NetworkError, ResourceError


logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class BackoffStrategy(Enum):
    """退避策略枚舉"""
    FIXED = "fixed"                 # 固定延遲
    LINEAR = "linear"               # 線性增長
    EXPONENTIAL = "exponential"     # 指數退避
    JITTERED = "jittered"          # 帶隨機抖動的指數退避


class RetryCondition(Enum):
    """重試條件枚舉"""
    ON_EXCEPTION = "on_exception"   # 基於例外類型
    ON_RESULT = "on_result"         # 基於返回結果
    CUSTOM = "custom"               # 自訂條件


class RetryPolicy(ABC):
    """重試策略抽象基類"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            NetworkError,
            ResourceError,
        ]
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """計算延遲時間"""
        pass
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """判斷是否應該重試"""
        if attempt >= self.max_attempts:
            return False
        
        return any(
            isinstance(exception, exc_type) 
            for exc_type in self.retryable_exceptions
        )
    
    def on_retry(self, attempt: int, exception: Exception, delay: float):
        """重試前的回調"""
        logger.warning(
            f"第 {attempt} 次重試，延遲 {delay:.2f} 秒。錯誤: {exception}"
        )


class FixedDelayPolicy(RetryPolicy):
    """固定延遲重試策略"""
    
    def __init__(self, delay: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
    
    def get_delay(self, attempt: int) -> float:
        return self.delay


class LinearBackoffPolicy(RetryPolicy):
    """線性退避重試策略"""
    
    def __init__(self, base_delay: float = 1.0, increment: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.base_delay = base_delay
        self.increment = increment
    
    def get_delay(self, attempt: int) -> float:
        return self.base_delay + (attempt - 1) * self.increment


class ExponentialBackoffPolicy(RetryPolicy):
    """指數退避重試策略"""
    
    def __init__(
        self,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # 添加 ±25% 的隨機抖動
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class CircuitBreakerPolicy(RetryPolicy):
    """熔斷器重試策略"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        current_time = datetime.now()
        
        # 檢查熔斷器狀態
        if self.is_open:
            if (current_time - self.last_failure_time).total_seconds() > self.recovery_timeout:
                # 嘗試半開狀態
                self.is_open = False
                self.failure_count = 0
                logger.info("熔斷器進入半開狀態")
            else:
                logger.warning("熔斷器處於開啟狀態，跳過重試")
                return False
        
        # 檢查是否應該重試
        if not super().should_retry(attempt, exception):
            return False
        
        # 更新失敗計數
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # 檢查是否需要開啟熔斷器
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"熔斷器開啟，失敗次數: {self.failure_count}")
            return False
        
        return True
    
    def get_delay(self, attempt: int) -> float:
        return 1.0  # 簡單的固定延遲


def retry_with_policy(policy: RetryPolicy):
    """重試裝飾器（同步版本）"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            last_exception = None
            
            while attempt <= policy.max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not policy.should_retry(attempt, e):
                        break
                    
                    if attempt < policy.max_attempts:
                        delay = policy.get_delay(attempt)
                        policy.on_retry(attempt, e, delay)
                        time.sleep(delay)
                    
                    attempt += 1
            
            # 所有重試都失敗了
            logger.error(f"重試 {policy.max_attempts} 次後仍然失敗: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator


def async_retry_with_policy(policy: RetryPolicy):
    """非同步重試裝飾器"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 1
            last_exception = None
            
            while attempt <= policy.max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not policy.should_retry(attempt, e):
                        break
                    
                    if attempt < policy.max_attempts:
                        delay = policy.get_delay(attempt)
                        policy.on_retry(attempt, e, delay)
                        await asyncio.sleep(delay)
                    
                    attempt += 1
            
            # 所有重試都失敗了
            logger.error(f"非同步重試 {policy.max_attempts} 次後仍然失敗: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator


class RetryManager:
    """重試管理器"""
    
    def __init__(self):
        self.policies: Dict[str, RetryPolicy] = {}
        self.stats: Dict[str, Dict[str, Any]] = {}
    
    def register_policy(self, name: str, policy: RetryPolicy):
        """註冊重試策略"""
        self.policies[name] = policy
        self.stats[name] = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "last_used": None
        }
        logger.info(f"註冊重試策略: {name}")
    
    def get_policy(self, name: str) -> Optional[RetryPolicy]:
        """獲取重試策略"""
        return self.policies.get(name)
    
    def execute_with_retry(
        self,
        func: Callable,
        policy_name: str = "default",
        *args,
        **kwargs
    ) -> Any:
        """使用指定策略執行函數"""
        policy = self.get_policy(policy_name)
        if not policy:
            logger.warning(f"未找到重試策略: {policy_name}，使用預設策略")
            policy = ExponentialBackoffPolicy()
        
        # 更新統計
        stats = self.stats.setdefault(policy_name, {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "last_used": None
        })
        
        attempt = 1
        last_exception = None
        
        while attempt <= policy.max_attempts:
            try:
                stats["total_attempts"] += 1
                stats["last_used"] = datetime.now()
                
                result = func(*args, **kwargs)
                
                if attempt > 1:
                    stats["successful_retries"] += 1
                    logger.info(f"重試成功，第 {attempt} 次嘗試")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if not policy.should_retry(attempt, e):
                    break
                
                if attempt < policy.max_attempts:
                    delay = policy.get_delay(attempt)
                    policy.on_retry(attempt, e, delay)
                    time.sleep(delay)
                
                attempt += 1
        
        # 重試失敗
        stats["failed_retries"] += 1
        logger.error(f"重試策略 {policy_name} 執行失敗: {last_exception}")
        raise last_exception
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取重試統計資訊"""
        return {
            "policies": list(self.policies.keys()),
            "stats": dict(self.stats)
        }


# 全域重試管理器
_retry_manager = None


def get_retry_manager() -> RetryManager:
    """獲取全域重試管理器"""
    global _retry_manager
    
    if _retry_manager is None:
        _retry_manager = RetryManager()
        
        # 註冊預設策略
        _retry_manager.register_policy("default", ExponentialBackoffPolicy())
        _retry_manager.register_policy("fast", FixedDelayPolicy(delay=0.1, max_attempts=2))
        _retry_manager.register_policy("slow", LinearBackoffPolicy(base_delay=2.0, increment=1.0))
        _retry_manager.register_policy("circuit_breaker", CircuitBreakerPolicy())
    
    return _retry_manager


# 便利裝飾器
def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Optional[List[Type[Exception]]] = None
):
    """簡單的重試裝飾器"""
    policy = ExponentialBackoffPolicy(
        base_delay=delay,
        backoff_factor=backoff_factor,
        max_attempts=max_attempts,
        retryable_exceptions=exceptions
    )
    return retry_with_policy(policy)


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Optional[List[Type[Exception]]] = None
):
    """簡單的非同步重試裝飾器"""
    policy = ExponentialBackoffPolicy(
        base_delay=delay,
        backoff_factor=backoff_factor,
        max_attempts=max_attempts,
        retryable_exceptions=exceptions
    )
    return async_retry_with_policy(policy)