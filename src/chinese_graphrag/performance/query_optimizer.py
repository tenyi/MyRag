"""
查詢效能優化器

提供查詢快取、索引優化、預載入和智慧路由功能
"""

import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class QueryCacheConfig:
    """查詢快取配置"""

    # 記憶體快取設定
    memory_cache_size: int = 1000  # 最大快取條目數
    memory_cache_ttl: int = 3600  # 快取存活時間（秒）

    # Redis 快取設定
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 7200

    # 快取策略
    cache_similarity_threshold: float = 0.95  # 相似查詢快取閾值
    enable_semantic_cache: bool = True
    max_cache_key_length: int = 200

    # 預載入設定
    enable_preloading: bool = True
    preload_popular_queries: bool = True
    preload_batch_size: int = 10


@dataclass
class QueryStats:
    """查詢統計資料"""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    total_query_time: float = 0.0
    avg_query_time: float = 0.0
    min_query_time: float = float("inf")
    max_query_time: float = 0.0

    cache_hit_time: float = 0.0
    cache_miss_time: float = 0.0

    popular_queries: Dict[str, int] = field(default_factory=dict)
    query_patterns: Dict[str, int] = field(default_factory=dict)

    def update_query_result(self, query: str, processing_time: float, cache_hit: bool):
        """更新查詢結果統計"""
        self.total_queries += 1
        self.total_query_time += processing_time

        if cache_hit:
            self.cache_hits += 1
            self.cache_hit_time += processing_time
        else:
            self.cache_misses += 1
            self.cache_miss_time += processing_time

        # 更新統計
        self.avg_query_time = self.total_query_time / self.total_queries
        self.min_query_time = min(self.min_query_time, processing_time)
        self.max_query_time = max(self.max_query_time, processing_time)

        # 記錄熱門查詢
        query_key = self._normalize_query(query)
        self.popular_queries[query_key] = self.popular_queries.get(query_key, 0) + 1

        # 分析查詢模式
        pattern = self._extract_query_pattern(query)
        self.query_patterns[pattern] = self.query_patterns.get(pattern, 0) + 1

    def _normalize_query(self, query: str) -> str:
        """標準化查詢字串"""
        return query.lower().strip()[:100]  # 限制長度

    def _extract_query_pattern(self, query: str) -> str:
        """提取查詢模式"""
        # 簡單的模式提取：基於查詢長度和關鍵詞
        length_category = (
            "short" if len(query) < 50 else "medium" if len(query) < 200 else "long"
        )

        # 檢查常見關鍵詞
        keywords = []
        if "什麼" in query or "什么" in query:
            keywords.append("what")
        if "如何" in query or "怎麼" in query or "怎么" in query:
            keywords.append("how")
        if "為什麼" in query or "为什么" in query:
            keywords.append("why")
        if "哪裡" in query or "哪里" in query or "在哪" in query:
            keywords.append("where")

        pattern = f"{length_category}_{'+'.join(keywords) if keywords else 'general'}"
        return pattern

    @property
    def cache_hit_rate(self) -> float:
        """快取命中率"""
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """取得統計摘要"""
        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": self.cache_hit_rate,
            "avg_query_time": self.avg_query_time,
            "avg_cache_hit_time": (
                self.cache_hit_time / self.cache_hits if self.cache_hits > 0 else 0
            ),
            "avg_cache_miss_time": (
                self.cache_miss_time / self.cache_misses if self.cache_misses > 0 else 0
            ),
            "top_queries": dict(
                sorted(self.popular_queries.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ),
            "query_patterns": dict(
                sorted(self.query_patterns.items(), key=lambda x: x[1], reverse=True)
            ),
        }


class QueryCache:
    """查詢快取管理器"""

    def __init__(self, config: Optional[QueryCacheConfig] = None):
        """初始化查詢快取

        Args:
            config: 快取配置
        """
        self.config = config or QueryCacheConfig()

        # 記憶體快取
        self._memory_cache: OrderedDict = OrderedDict()
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.RLock()

        # Redis 快取
        self._redis_client: Optional[Any] = None
        if self.config.enable_redis and REDIS_AVAILABLE:
            self._initialize_redis()

        # 語義快取（用於相似查詢）
        self._semantic_cache: Dict[str, Tuple[np.ndarray, Any, float]] = (
            {}
        )  # query -> (embedding, result, timestamp)

        logger.info(
            f"查詢快取初始化完成，記憶體快取大小: {self.config.memory_cache_size}"
        )

    def _initialize_redis(self):
        """初始化 Redis 連線"""
        try:
            import redis

            self._redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
            )
            # 測試連線
            self._redis_client.ping()
            logger.info("Redis 快取連線成功")
        except Exception as e:
            logger.warning(f"Redis 快取連線失敗: {e}")
            self._redis_client = None

    def _generate_cache_key(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成快取鍵值

        Args:
            query: 查詢字串
            context: 查詢上下文

        Returns:
            快取鍵值
        """
        # 標準化查詢
        normalized_query = query.lower().strip()

        # 包含上下文資訊
        cache_data = {"query": normalized_query, "context": context or {}}

        # 生成 hash
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.md5(cache_str.encode("utf-8")).hexdigest()

        return cache_key[: self.config.max_cache_key_length]

    async def get(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """從快取取得查詢結果

        Args:
            query: 查詢字串
            context: 查詢上下文

        Returns:
            快取的結果，如果沒有則返回 None
        """
        cache_key = self._generate_cache_key(query, context)

        # 檢查記憶體快取
        with self._cache_lock:
            if cache_key in self._memory_cache:
                # 檢查是否過期
                if (
                    time.time() - self._cache_timestamps.get(cache_key, 0)
                    < self.config.memory_cache_ttl
                ):
                    # 移到最前面（LRU）
                    result = self._memory_cache.pop(cache_key)
                    self._memory_cache[cache_key] = result
                    logger.debug(f"記憶體快取命中: {cache_key[:16]}...")
                    return result
                else:
                    # 過期，移除
                    del self._memory_cache[cache_key]
                    del self._cache_timestamps[cache_key]

        # 檢查 Redis 快取
        if self._redis_client:
            try:
                cached_data = self._redis_client.get(f"query_cache:{cache_key}")
                if cached_data:
                    result = json.loads(cached_data)
                    # 同時更新記憶體快取
                    await self.put(query, result, context)
                    logger.debug(f"Redis 快取命中: {cache_key[:16]}...")
                    return result
            except Exception as e:
                logger.warning(f"Redis 快取讀取失敗: {e}")

        # 檢查語義快取
        if self.config.enable_semantic_cache:
            semantic_result = await self._check_semantic_cache(query, context)
            if semantic_result:
                logger.debug(f"語義快取命中: {cache_key[:16]}...")
                return semantic_result

        return None

    async def put(
        self, query: str, result: Any, context: Optional[Dict[str, Any]] = None
    ):
        """將查詢結果存入快取

        Args:
            query: 查詢字串
            result: 查詢結果
            context: 查詢上下文
        """
        cache_key = self._generate_cache_key(query, context)
        current_time = time.time()

        # 存入記憶體快取
        with self._cache_lock:
            # 如果快取已滿，移除最舊的項目
            if len(self._memory_cache) >= self.config.memory_cache_size:
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]
                del self._cache_timestamps[oldest_key]

            self._memory_cache[cache_key] = result
            self._cache_timestamps[cache_key] = current_time

        # 存入 Redis 快取
        if self._redis_client:
            try:
                cached_data = json.dumps(result, ensure_ascii=False, default=str)
                self._redis_client.setex(
                    f"query_cache:{cache_key}", self.config.redis_ttl, cached_data
                )
            except Exception as e:
                logger.warning(f"Redis 快取寫入失敗: {e}")

        logger.debug(f"查詢結果已快取: {cache_key[:16]}...")

    async def _check_semantic_cache(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """檢查語義快取（相似查詢）

        Args:
            query: 查詢字串
            context: 查詢上下文

        Returns:
            相似查詢的結果，如果沒有則返回 None
        """
        if not self._semantic_cache:
            return None

        try:
            # 這裡需要查詢的 embedding，實際實作時需要整合 embedding 服務
            # 為了簡化，這裡使用簡單的字串相似度
            query_normalized = query.lower().strip()

            for cached_query, (
                cached_embedding,
                cached_result,
                timestamp,
            ) in self._semantic_cache.items():
                # 檢查是否過期
                if time.time() - timestamp > self.config.memory_cache_ttl:
                    continue

                # 簡單的字串相似度檢查（實際應該使用 embedding 相似度）
                similarity = self._calculate_string_similarity(
                    query_normalized, cached_query
                )

                if similarity >= self.config.cache_similarity_threshold:
                    return cached_result

        except Exception as e:
            logger.warning(f"語義快取檢查失敗: {e}")

        return None

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """計算字串相似度（簡化版本）"""
        if str1 == str2:
            return 1.0

        # 使用 Jaccard 相似度
        set1 = set(str1)
        set2 = set(str2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def clear_cache(self, cache_type: str = "all"):
        """清理快取

        Args:
            cache_type: 快取類型 ("memory", "redis", "semantic", "all")
        """
        if cache_type in ("memory", "all"):
            with self._cache_lock:
                self._memory_cache.clear()
                self._cache_timestamps.clear()
            logger.info("記憶體快取已清理")

        if cache_type in ("redis", "all") and self._redis_client:
            try:
                # 刪除所有查詢快取
                keys = self._redis_client.keys("query_cache:*")
                if keys:
                    self._redis_client.delete(*keys)
                logger.info("Redis 快取已清理")
            except Exception as e:
                logger.warning(f"Redis 快取清理失敗: {e}")

        if cache_type in ("semantic", "all"):
            self._semantic_cache.clear()
            logger.info("語義快取已清理")

    def get_cache_stats(self) -> Dict[str, Any]:
        """取得快取統計資料"""
        with self._cache_lock:
            memory_cache_size = len(self._memory_cache)

        redis_cache_size = 0
        if self._redis_client:
            try:
                redis_cache_size = len(self._redis_client.keys("query_cache:*"))
            except Exception:
                pass

        return {
            "memory_cache_size": memory_cache_size,
            "memory_cache_limit": self.config.memory_cache_size,
            "redis_cache_size": redis_cache_size,
            "semantic_cache_size": len(self._semantic_cache),
            "redis_enabled": self._redis_client is not None,
        }


class QueryOptimizer:
    """查詢優化器

    提供查詢快取、索引優化、預載入和智慧路由功能
    """

    def __init__(self, config: Optional[QueryCacheConfig] = None):
        """初始化查詢優化器

        Args:
            config: 快取配置
        """
        self.config = config or QueryCacheConfig()
        self.cache = QueryCache(config)
        self.stats = QueryStats()

        # 查詢路由
        self._query_routers: Dict[str, Callable] = {}

        # 索引優化
        self._index_cache: Dict[str, Any] = {}
        self._preloaded_data: Dict[str, Any] = {}

        logger.info("查詢優化器初始化完成")

    def register_query_router(self, pattern: str, handler: Callable):
        """註冊查詢路由器

        Args:
            pattern: 查詢模式
            handler: 處理函數
        """
        self._query_routers[pattern] = handler
        logger.info(f"註冊查詢路由器: {pattern}")

    async def optimize_query(
        self,
        query: str,
        query_func: Callable,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Any:
        """優化查詢執行

        Args:
            query: 查詢字串
            query_func: 查詢執行函數
            context: 查詢上下文
            use_cache: 是否使用快取

        Returns:
            查詢結果
        """
        start_time = time.time()
        cache_hit = False

        try:
            # 檢查快取
            if use_cache:
                cached_result = await self.cache.get(query, context)
                if cached_result is not None:
                    cache_hit = True
                    processing_time = time.time() - start_time
                    self.stats.update_query_result(query, processing_time, cache_hit)
                    return cached_result

            # 查詢路由
            optimized_query_func = self._route_query(query, query_func)

            # 執行查詢 - 適應不同的函數簽名
            import inspect

            sig = inspect.signature(optimized_query_func)

            # 根據參數數量調用函數
            if len(sig.parameters) == 1:
                # 只接受 query 參數
                if asyncio.iscoroutinefunction(optimized_query_func):
                    result = await optimized_query_func(query)
                else:
                    result = optimized_query_func(query)
            else:
                # 接受 query 和 context 參數
                if asyncio.iscoroutinefunction(optimized_query_func):
                    result = await optimized_query_func(query, context)
                else:
                    result = optimized_query_func(query, context)

            # 存入快取
            if use_cache:
                await self.cache.put(query, result, context)

            processing_time = time.time() - start_time
            self.stats.update_query_result(query, processing_time, cache_hit)

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats.update_query_result(query, processing_time, cache_hit)
            logger.error(f"查詢優化失敗: {e}")
            raise

    def _route_query(self, query: str, default_func: Callable) -> Callable:
        """路由查詢到最適合的處理器

        Args:
            query: 查詢字串
            default_func: 預設處理函數

        Returns:
            最適合的處理函數
        """
        # 分析查詢模式
        pattern = self.stats._extract_query_pattern(query)

        # 查找匹配的路由器
        for router_pattern, handler in self._query_routers.items():
            if router_pattern in pattern or pattern in router_pattern:
                logger.debug(f"查詢路由: {pattern} -> {router_pattern}")
                return handler

        return default_func

    async def preload_popular_queries(self, query_func: Callable, limit: int = 50):
        """預載入熱門查詢

        Args:
            query_func: 查詢執行函數
            limit: 預載入查詢數量限制
        """
        if not self.config.enable_preloading:
            return

        # 取得熱門查詢
        popular_queries = sorted(
            self.stats.popular_queries.items(), key=lambda x: x[1], reverse=True
        )[:limit]

        logger.info(f"開始預載入 {len(popular_queries)} 個熱門查詢")

        preload_tasks = []
        for query, count in popular_queries:
            # 檢查是否已快取
            if await self.cache.get(query) is None:
                task = self._preload_query(query, query_func)
                preload_tasks.append(task)

        # 批次執行預載入
        if preload_tasks:
            await asyncio.gather(*preload_tasks, return_exceptions=True)

        logger.info(f"預載入完成，處理了 {len(preload_tasks)} 個查詢")

    async def _preload_query(self, query: str, query_func: Callable):
        """預載入單個查詢"""
        try:
            result = await query_func(query, {})
            await self.cache.put(query, result)
            logger.debug(f"預載入查詢: {query[:50]}...")
        except Exception as e:
            logger.warning(f"預載入查詢失敗 {query[:50]}...: {e}")

    def optimize_index_access(self, index_key: str, data: Any):
        """優化索引存取

        Args:
            index_key: 索引鍵值
            data: 索引資料
        """
        self._index_cache[index_key] = {
            "data": data,
            "timestamp": time.time(),
            "access_count": 0,
        }

        logger.debug(f"索引已快取: {index_key}")

    def get_index_data(self, index_key: str) -> Optional[Any]:
        """取得索引資料

        Args:
            index_key: 索引鍵值

        Returns:
            索引資料，如果沒有則返回 None
        """
        if index_key in self._index_cache:
            cache_entry = self._index_cache[index_key]
            cache_entry["access_count"] += 1
            return cache_entry["data"]

        return None

    def get_optimization_stats(self) -> Dict[str, Any]:
        """取得優化統計資料"""
        stats_summary = self.stats.get_summary()
        cache_stats = self.cache.get_cache_stats()

        return {
            "query_stats": stats_summary,
            "cache_stats": cache_stats,
            "index_cache_size": len(self._index_cache),
            "registered_routers": len(self._query_routers),
        }

    def clear_all_caches(self):
        """清理所有快取"""
        self.cache.clear_cache("all")
        self._index_cache.clear()
        self._preloaded_data.clear()
        logger.info("所有快取已清理")

    def reset_stats(self):
        """重設統計資料"""
        self.stats = QueryStats()
        logger.info("查詢統計資料已重設")
