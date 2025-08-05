"""
Embedding 向量快取機制

提供多層次的向量快取功能，包括記憶體快取、磁碟快取和分散式快取
支援 LRU 淘汰策略和快取預熱功能
"""

import asyncio
import hashlib
import json
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # 使用 debug 級別避免過多警告
    logger.debug("redis 套件未安裝，分散式快取將不可用")


@dataclass
class CacheEntry:
    """快取條目資料結構"""

    key: str
    embeddings: np.ndarray
    texts: List[str]
    model_name: str
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_access == 0.0:
            self.last_access = self.timestamp

    @property
    def size_bytes(self) -> int:
        """計算條目大小（位元組）"""
        embedding_size = self.embeddings.nbytes
        text_size = sum(len(text.encode("utf-8")) for text in self.texts)
        metadata_size = len(
            json.dumps(self.metadata, ensure_ascii=False).encode("utf-8")
        )
        return embedding_size + text_size + metadata_size + 200  # 額外開銷

    @property
    def age_seconds(self) -> float:
        """條目年齡（秒）"""
        return time.time() - self.timestamp

    def update_access(self):
        """更新存取資訊"""
        self.access_count += 1
        self.last_access = time.time()


class CacheStrategy(ABC):
    """快取策略抽象基類"""

    @abstractmethod
    def should_evict(self, entry: CacheEntry, cache_stats: Dict[str, Any]) -> bool:
        """判斷是否應該淘汰條目"""
        pass

    @abstractmethod
    def get_eviction_priority(self, entry: CacheEntry) -> float:
        """取得淘汰優先級（數值越大越優先淘汰）"""
        pass


class LRUStrategy(CacheStrategy):
    """LRU (Least Recently Used) 淘汰策略"""

    def should_evict(self, entry: CacheEntry, cache_stats: Dict[str, Any]) -> bool:
        """基於記憶體使用率和條目年齡判斷是否淘汰"""
        memory_usage_ratio = cache_stats.get("memory_usage_ratio", 0.0)

        # 記憶體使用率超過 80% 時開始淘汰
        if memory_usage_ratio > 0.8:
            # 超過 1 小時未存取的條目優先淘汰
            return (time.time() - entry.last_access) > 3600

        # 記憶體使用率超過 90% 時積極淘汰
        if memory_usage_ratio > 0.9:
            return (time.time() - entry.last_access) > 1800  # 30 分鐘

        return False

    def get_eviction_priority(self, entry: CacheEntry) -> float:
        """LRU 優先級：最近存取時間越久遠，優先級越高"""
        return time.time() - entry.last_access


class LFUStrategy(CacheStrategy):
    """LFU (Least Frequently Used) 淘汰策略"""

    def should_evict(self, entry: CacheEntry, cache_stats: Dict[str, Any]) -> bool:
        """基於存取頻率判斷是否淘汰"""
        memory_usage_ratio = cache_stats.get("memory_usage_ratio", 0.0)
        avg_access_count = cache_stats.get("avg_access_count", 1.0)

        if memory_usage_ratio > 0.8:
            # 存取次數低於平均值的條目優先淘汰
            return entry.access_count < avg_access_count * 0.5

        return False

    def get_eviction_priority(self, entry: CacheEntry) -> float:
        """LFU 優先級：存取次數越少，優先級越高"""
        # 考慮時間衰減因子
        time_factor = max(0.1, 1.0 - entry.age_seconds / 86400)  # 24小時衰減
        adjusted_count = entry.access_count * time_factor
        return 1.0 / (adjusted_count + 1)  # 避免除零


class EmbeddingCache(ABC):
    """Embedding 快取抽象基類"""

    def __init__(
        self,
        max_size_mb: float = 1024,
        strategy: CacheStrategy = None,
        enable_compression: bool = True,
    ):
        """初始化快取

        Args:
            max_size_mb: 最大快取大小（MB）
            strategy: 淘汰策略
            enable_compression: 是否啟用壓縮
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy or LRUStrategy()
        self.enable_compression = enable_compression
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size_bytes": 0,
            "entry_count": 0,
        }
        self._lock = threading.RLock()

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """取得快取條目"""
        pass

    @abstractmethod
    async def put(self, key: str, entry: CacheEntry) -> bool:
        """存入快取條目"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """刪除快取條目"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """清空快取"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """取得快取統計資訊"""
        pass

    def _generate_cache_key(
        self, texts: List[str], model_name: str, normalize: bool = True, **kwargs
    ) -> str:
        """生成快取鍵值

        Args:
            texts: 文本列表
            model_name: 模型名稱
            normalize: 是否正規化
            **kwargs: 其他參數

        Returns:
            str: 快取鍵值
        """
        # 建立唯一標識
        content = {
            "texts": texts,
            "model_name": model_name,
            "normalize": normalize,
            **kwargs,
        }

        # 序列化並計算雜湊
        content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        hash_obj = hashlib.sha256(content_str.encode("utf-8"))
        return hash_obj.hexdigest()

    def _compress_embeddings(self, embeddings: np.ndarray) -> bytes:
        """壓縮 embedding 資料"""
        if not self.enable_compression:
            return pickle.dumps(embeddings)

        # 使用 numpy 的壓縮功能
        import gzip

        pickled_data = pickle.dumps(embeddings)
        return gzip.compress(pickled_data)

    def _decompress_embeddings(self, compressed_data: bytes) -> np.ndarray:
        """解壓縮 embedding 資料"""
        if not self.enable_compression:
            return pickle.loads(compressed_data)

        import gzip

        pickled_data = gzip.decompress(compressed_data)
        return pickle.loads(pickled_data)

    @property
    def hit_rate(self) -> float:
        """快取命中率"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total_requests if total_requests > 0 else 0.0

    @property
    def memory_usage_ratio(self) -> float:
        """記憶體使用率"""
        return self.stats["total_size_bytes"] / self.max_size_bytes


class MemoryCache(EmbeddingCache):
    """記憶體快取實作"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        logger.info(
            f"初始化記憶體快取，最大大小: {self.max_size_bytes / 1024 / 1024:.3f} MB"
        )

    async def get(self, key: str) -> Optional[CacheEntry]:
        """取得快取條目"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.update_access()

                # 移到最後（LRU 更新）
                self._cache.move_to_end(key)

                self.stats["hits"] += 1
                logger.debug(f"快取命中: {key[:16]}...")
                return entry
            else:
                self.stats["misses"] += 1
                logger.debug(f"快取未命中: {key[:16]}...")
                return None

    async def put(self, key: str, entry: CacheEntry) -> bool:
        """存入快取條目"""
        with self._lock:
            try:
                # 檢查是否需要淘汰舊條目
                await self._evict_if_needed(entry.size_bytes)

                # 存入新條目
                self._cache[key] = entry
                self.stats["total_size_bytes"] += entry.size_bytes
                self.stats["entry_count"] += 1

                logger.debug(f"快取存入: {key[:16]}..., 大小: {entry.size_bytes} bytes")
                return True

            except Exception as e:
                logger.error(f"快取存入失敗: {e}")
                return False

    async def delete(self, key: str) -> bool:
        """刪除快取條目"""
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self.stats["total_size_bytes"] -= entry.size_bytes
                self.stats["entry_count"] -= 1
                logger.debug(f"快取刪除: {key[:16]}...")
                return True
            return False

    async def remove(self, key: str) -> bool:
        """移除快取條目（delete 的別名）"""
        return await self.delete(key)

    async def clear(self) -> None:
        """清空快取"""
        with self._lock:
            self._cache.clear()
            self.stats["total_size_bytes"] = 0
            self.stats["entry_count"] = 0
            logger.info("記憶體快取已清空")

    async def get_stats(self) -> Dict[str, Any]:
        """取得快取統計資訊"""
        with self._lock:
            cache_stats = self.stats.copy()
            cache_stats.update(
                {
                    "hit_rate": self.hit_rate,
                    "memory_usage_ratio": self.memory_usage_ratio,
                    "max_size_mb": self.max_size_bytes // 1024 // 1024,
                    "current_size_mb": self.stats["total_size_bytes"] // 1024 // 1024,
                    "memory_usage_mb": max(
                        0.001, self.stats["total_size_bytes"] / 1024 / 1024
                    ),
                    "avg_access_count": self._calculate_avg_access_count(),
                    "cache_type": "memory",
                }
            )
            return cache_stats

    def _calculate_avg_access_count(self) -> float:
        """計算平均存取次數"""
        if not self._cache:
            return 0.0

        total_access = sum(entry.access_count for entry in self._cache.values())
        return total_access / len(self._cache)

    async def _evict_if_needed(self, new_entry_size: int) -> None:
        """根據需要淘汰條目"""
        # 檢查是否需要淘汰
        projected_size = self.stats["total_size_bytes"] + new_entry_size

        if projected_size <= self.max_size_bytes:
            return

        # 取得快取統計資訊用於淘汰決策
        cache_stats = await self.get_stats()

        # 收集需要淘汰的條目
        eviction_candidates = []

        for key, entry in self._cache.items():
            if self.strategy.should_evict(entry, cache_stats):
                priority = self.strategy.get_eviction_priority(entry)
                eviction_candidates.append((priority, key, entry))

        # 按優先級排序（優先級高的先淘汰）
        eviction_candidates.sort(key=lambda x: x[0], reverse=True)

        # 淘汰條目直到有足夠空間
        bytes_to_free = projected_size - self.max_size_bytes
        freed_bytes = 0

        for priority, key, entry in eviction_candidates:
            if freed_bytes >= bytes_to_free:
                break

            await self.delete(key)
            freed_bytes += entry.size_bytes
            self.stats["evictions"] += 1

            logger.debug(f"淘汰快取條目: {key[:16]}..., 釋放: {entry.size_bytes} bytes")

        # 如果還是空間不足，強制淘汰最舊的條目
        while (
            self.stats["total_size_bytes"] + new_entry_size > self.max_size_bytes
            and self._cache
        ):
            oldest_key = next(iter(self._cache))
            oldest_entry = self._cache[oldest_key]
            await self.delete(oldest_key)
            self.stats["evictions"] += 1
            logger.warning(f"強制淘汰最舊條目: {oldest_key[:16]}...")


class DiskCache(EmbeddingCache):
    """磁碟快取實作"""

    def __init__(self, cache_dir: Union[str, Path] = "./cache/embeddings", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 索引檔案
        self.index_file = self.cache_dir / "cache_index.json"
        self._index: Dict[str, Dict[str, Any]] = {}

        # 載入現有索引（延遲到第一次使用時）
        self._index_loaded = False
        self._index_dirty = False

        logger.info(f"初始化磁碟快取，目錄: {self.cache_dir}")

    async def _load_index(self) -> None:
        """載入快取索引"""
        try:
            if self.index_file.exists():
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self._index = json.load(f)

                # 更新統計資訊
                self.stats["entry_count"] = len(self._index)
                self.stats["total_size_bytes"] = sum(
                    item.get("size_bytes", 0) for item in self._index.values()
                )

                logger.info(f"載入磁碟快取索引: {len(self._index)} 個條目")
        except Exception as e:
            logger.error(f"載入磁碟快取索引失敗: {e}")
            self._index = {}
        finally:
            self._index_loaded = True

    async def _save_index(self) -> None:
        """儲存快取索引"""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"儲存磁碟快取索引失敗: {e}")

    def _get_cache_file_path(self, key: str) -> Path:
        """取得快取檔案路徑"""
        # 使用前兩個字符作為子目錄，避免單一目錄檔案過多
        subdir = key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{key}.pkl"

    async def get(self, key: str) -> Optional[CacheEntry]:
        """取得快取條目"""
        # 確保索引已載入
        if not self._index_loaded:
            await self._load_index()

        with self._lock:
            if key not in self._index:
                self.stats["misses"] += 1
                return None

            try:
                cache_file = self._get_cache_file_path(key)

                if not cache_file.exists():
                    # 索引存在但檔案不存在，清理索引
                    del self._index[key]
                    # 延遲保存索引，避免在同步上下文中創建任務
                    self._index_dirty = True
                    self.stats["misses"] += 1
                    return None

                # 載入快取條目
                with open(cache_file, "rb") as f:
                    compressed_data = f.read()

                # 解壓縮 embeddings
                cache_data = pickle.loads(compressed_data)
                cache_data["embeddings"] = self._decompress_embeddings(
                    cache_data["embeddings_compressed"]
                )
                del cache_data["embeddings_compressed"]

                entry = CacheEntry(**cache_data)
                entry.update_access()

                # 更新索引中的存取資訊
                self._index[key].update(
                    {
                        "access_count": entry.access_count,
                        "last_access": entry.last_access,
                    }
                )
                # 延遲保存索引，避免在同步上下文中創建任務
                self._index_dirty = True

                self.stats["hits"] += 1
                logger.debug(f"磁碟快取命中: {key[:16]}...")

                # 如果索引髒了，保存它
                if self._index_dirty:
                    await self._save_index()
                    self._index_dirty = False

                return entry

            except Exception as e:
                logger.error(f"讀取磁碟快取失敗: {e}")
                self.stats["misses"] += 1
                return None

    async def put(self, key: str, entry: CacheEntry) -> bool:
        """存入快取條目"""
        # 確保索引已載入
        if not self._index_loaded:
            await self._load_index()

        with self._lock:
            try:
                # 檢查是否需要淘汰
                await self._evict_if_needed(entry.size_bytes)

                cache_file = self._get_cache_file_path(key)

                # 準備儲存資料
                cache_data = asdict(entry)
                cache_data["embeddings_compressed"] = self._compress_embeddings(
                    entry.embeddings
                )
                del cache_data["embeddings"]

                # 儲存到檔案
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)

                # 更新索引
                self._index[key] = {
                    "timestamp": entry.timestamp,
                    "access_count": entry.access_count,
                    "last_access": entry.last_access,
                    "size_bytes": entry.size_bytes,
                    "model_name": entry.model_name,
                    "text_count": len(entry.texts),
                }

                await self._save_index()

                # 更新統計
                self.stats["total_size_bytes"] += entry.size_bytes
                self.stats["entry_count"] += 1

                logger.debug(
                    f"磁碟快取存入: {key[:16]}..., 大小: {entry.size_bytes} bytes"
                )
                return True

            except Exception as e:
                logger.error(f"磁碟快取存入失敗: {e}")
                return False

    async def delete(self, key: str) -> bool:
        """刪除快取條目"""
        with self._lock:
            if key not in self._index:
                return False

            try:
                cache_file = self._get_cache_file_path(key)

                # 刪除檔案
                if cache_file.exists():
                    cache_file.unlink()

                # 更新統計
                entry_info = self._index[key]
                self.stats["total_size_bytes"] -= entry_info.get("size_bytes", 0)
                self.stats["entry_count"] -= 1

                # 更新索引
                del self._index[key]
                await self._save_index()

                logger.debug(f"磁碟快取刪除: {key[:16]}...")
                return True

            except Exception as e:
                logger.error(f"磁碟快取刪除失敗: {e}")
                return False

    async def clear(self) -> None:
        """清空快取"""
        with self._lock:
            try:
                # 刪除所有快取檔案
                for cache_file in self.cache_dir.rglob("*.pkl"):
                    cache_file.unlink()

                # 清空索引
                self._index.clear()
                await self._save_index()

                # 重置統計
                self.stats["total_size_bytes"] = 0
                self.stats["entry_count"] = 0

                logger.info("磁碟快取已清空")

            except Exception as e:
                logger.error(f"清空磁碟快取失敗: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """取得快取統計資訊"""
        with self._lock:
            cache_stats = self.stats.copy()
            cache_stats.update(
                {
                    "hit_rate": self.hit_rate,
                    "memory_usage_ratio": self.memory_usage_ratio,
                    "max_size_mb": self.max_size_bytes // 1024 // 1024,
                    "current_size_mb": self.stats["total_size_bytes"] // 1024 // 1024,
                    "cache_dir": str(self.cache_dir),
                    "cache_type": "disk",
                }
            )
            return cache_stats

    async def _evict_if_needed(self, new_entry_size: int) -> None:
        """根據需要淘汰條目"""
        projected_size = self.stats["total_size_bytes"] + new_entry_size

        if projected_size <= self.max_size_bytes:
            return

        # 按淘汰策略排序
        eviction_candidates = []

        for key, info in self._index.items():
            # 建立臨時 CacheEntry 用於策略判斷
            temp_entry = CacheEntry(
                key=key,
                embeddings=np.array([]),  # 空陣列，只用於策略判斷
                texts=[],
                model_name=info["model_name"],
                timestamp=info["timestamp"],
                access_count=info["access_count"],
                last_access=info["last_access"],
            )

            cache_stats = await self.get_stats()
            if self.strategy.should_evict(temp_entry, cache_stats):
                priority = self.strategy.get_eviction_priority(temp_entry)
                eviction_candidates.append((priority, key, info["size_bytes"]))

        # 按優先級排序並淘汰
        eviction_candidates.sort(key=lambda x: x[0], reverse=True)

        bytes_to_free = projected_size - self.max_size_bytes
        freed_bytes = 0

        for priority, key, size_bytes in eviction_candidates:
            if freed_bytes >= bytes_to_free:
                break

            await self.delete(key)
            freed_bytes += size_bytes
            self.stats["evictions"] += 1


class MultiLevelCache:
    """多層次快取系統

    結合記憶體快取和磁碟快取，提供更好的效能和持久性
    """

    def __init__(
        self,
        memory_cache_mb: int = 512,
        disk_cache_mb: int = 2048,
        cache_dir: Union[str, Path] = "./cache/embeddings",
        strategy: CacheStrategy = None,
    ):
        """初始化多層次快取

        Args:
            memory_cache_mb: 記憶體快取大小（MB）
            disk_cache_mb: 磁碟快取大小（MB）
            cache_dir: 磁碟快取目錄
            strategy: 淘汰策略
        """
        self.memory_cache = MemoryCache(
            max_size_mb=memory_cache_mb, strategy=strategy or LRUStrategy()
        )

        self.disk_cache = DiskCache(
            max_size_mb=disk_cache_mb,
            cache_dir=cache_dir,
            strategy=strategy or LRUStrategy(),
        )

        logger.info(
            f"初始化多層次快取 - 記憶體: {memory_cache_mb}MB, 磁碟: {disk_cache_mb}MB"
        )

    async def get(self, key: str) -> Optional[CacheEntry]:
        """取得快取條目（先記憶體後磁碟）"""
        # 先檢查記憶體快取
        entry = await self.memory_cache.get(key)
        if entry:
            return entry

        # 檢查磁碟快取
        entry = await self.disk_cache.get(key)
        if entry:
            # 將熱點資料提升到記憶體快取
            await self.memory_cache.put(key, entry)
            return entry

        return None

    async def put(self, key: str, entry: CacheEntry) -> bool:
        """存入快取條目（同時存入記憶體和磁碟）"""
        # 存入記憶體快取
        memory_success = await self.memory_cache.put(key, entry)

        # 存入磁碟快取
        disk_success = await self.disk_cache.put(key, entry)

        return memory_success or disk_success

    async def delete(self, key: str) -> bool:
        """刪除快取條目（從兩層快取中刪除）"""
        memory_deleted = await self.memory_cache.delete(key)
        disk_deleted = await self.disk_cache.delete(key)

        return memory_deleted or disk_deleted

    async def clear(self) -> None:
        """清空所有快取"""
        await self.memory_cache.clear()
        await self.disk_cache.clear()
        logger.info("多層次快取已清空")

    async def get_stats(self) -> Dict[str, Any]:
        """取得綜合快取統計資訊"""
        memory_stats = await self.memory_cache.get_stats()
        disk_stats = await self.disk_cache.get_stats()

        total_hits = memory_stats["hits"] + disk_stats["hits"]
        total_misses = memory_stats["misses"] + disk_stats["misses"]
        total_requests = total_hits + total_misses

        return {
            "cache_type": "multi_level",
            "total_hit_rate": (
                total_hits / total_requests if total_requests > 0 else 0.0
            ),
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "memory_entries": memory_stats["entry_count"],
            "disk_entries": disk_stats["entry_count"],
            "memory_cache": memory_stats,
            "disk_cache": disk_stats,
        }

    async def preload_cache(
        self, embedding_service, texts: List[str], batch_size: int = 32
    ) -> Dict[str, Any]:
        """快取預熱

        Args:
            embedding_service: embedding 服務
            texts: 要預載入的文本列表
            batch_size: 批次大小

        Returns:
            Dict[str, Any]: 預載入結果統計
        """
        logger.info(f"開始快取預熱，文本數量: {len(texts)}")

        preload_stats = {
            "total_texts": len(texts),
            "processed_batches": 0,
            "cached_entries": 0,
            "errors": 0,
            "start_time": time.time(),
        }

        # 分批處理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # 生成快取鍵值
                cache_key = self.memory_cache._generate_cache_key(
                    batch_texts, embedding_service.model_name
                )

                # 檢查是否已快取
                if await self.get(cache_key):
                    continue

                # 計算 embeddings
                result = await embedding_service.embed_texts(batch_texts)

                # 建立快取條目
                entry = CacheEntry(
                    key=cache_key,
                    embeddings=result.embeddings,
                    texts=batch_texts,
                    model_name=result.model_name,
                    timestamp=time.time(),
                )

                # 存入快取
                if await self.put(cache_key, entry):
                    preload_stats["cached_entries"] += 1

                preload_stats["processed_batches"] += 1

                if preload_stats["processed_batches"] % 10 == 0:
                    logger.info(
                        f"預熱進度: {preload_stats['processed_batches']} 批次完成"
                    )

            except Exception as e:
                logger.error(f"預熱批次失敗: {e}")
                preload_stats["errors"] += 1

        preload_stats["duration_seconds"] = time.time() - preload_stats["start_time"]

        logger.info(
            f"快取預熱完成: {preload_stats['cached_entries']} 個條目，"
            f"耗時 {preload_stats['duration_seconds']:.2f} 秒"
        )

        return preload_stats


def create_embedding_cache(
    cache_type: str = "multi_level", **kwargs
) -> Union[MemoryCache, DiskCache, MultiLevelCache]:
    """建立 embedding 快取的便利函數

    Args:
        cache_type: 快取類型 ('memory', 'disk', 'multi_level')
        **kwargs: 快取配置參數

    Returns:
        快取實例
    """
    if cache_type == "memory":
        return MemoryCache(**kwargs)
    elif cache_type == "disk":
        return DiskCache(**kwargs)
    elif cache_type == "multi_level":
        return MultiLevelCache(**kwargs)
    else:
        raise ValueError(f"不支援的快取類型: {cache_type}")
