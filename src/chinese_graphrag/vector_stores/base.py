"""
向量儲存基礎類別和介面定義
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger


class VectorStoreType(Enum):
    """向量儲存類型枚舉"""

    LANCEDB = "lancedb"
    CHROMA = "chroma"
    FAISS = "faiss"
    PGVECTOR = "pgvector"


class SearchType(Enum):
    """搜尋類型枚舉"""

    DENSE = "dense"  # 密集向量搜尋
    SPARSE = "sparse"  # 稀疏向量搜尋
    HYBRID = "hybrid"  # 混合搜尋
    SEMANTIC = "semantic"  # 語義搜尋
    KEYWORD = "keyword"  # 關鍵字搜尋


class RerankingMethod(Enum):
    """重新排序方法枚舉"""

    RECIPROCAL_RANK_FUSION = "rrf"  # 倒數排名融合
    WEIGHTED_SCORE = "weighted"  # 加權分數
    CROSS_ENCODER = "cross_encoder"  # 交叉編碼器
    NONE = "none"  # 不重新排序


@dataclass
class VectorSearchResult:
    """向量搜尋結果資料類別"""

    ids: List[str]  # 文件 ID 列表
    distances: List[float]  # 距離分數列表
    similarities: List[float]  # 相似度分數列表
    metadata: List[Dict[str, Any]]  # 元資料列表
    embeddings: Optional[List[np.ndarray]] = None  # 向量列表（可選）
    search_type: Optional[SearchType] = None  # 搜尋類型
    reranking_scores: Optional[List[float]] = None  # 重新排序分數

    def __post_init__(self):
        """後處理初始化"""
        # 確保所有列表長度一致
        lengths = [
            len(self.ids),
            len(self.distances),
            len(self.similarities),
            len(self.metadata),
        ]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("搜尋結果各列表長度必須一致")

        # 如果提供了 embeddings，也要檢查長度
        if self.embeddings is not None and len(self.embeddings) != lengths[0]:
            raise ValueError("embeddings 列表長度必須與其他列表一致")

        # 如果提供了重新排序分數，也要檢查長度
        if (
            self.reranking_scores is not None
            and len(self.reranking_scores) != lengths[0]
        ):
            raise ValueError("reranking_scores 列表長度必須與其他列表一致")


@dataclass
class HybridSearchConfig:
    """混合搜尋配置"""

    dense_weight: float = 0.7  # 密集向量權重
    sparse_weight: float = 0.3  # 稀疏向量權重
    reranking_method: RerankingMethod = RerankingMethod.RECIPROCAL_RANK_FUSION
    reranking_k: int = 100  # 重新排序的候選數量
    alpha: float = 60.0  # RRF 參數

    def __post_init__(self):
        """驗證配置參數"""
        if not (0 <= self.dense_weight <= 1):
            raise ValueError("dense_weight 必須在 0 到 1 之間")
        if not (0 <= self.sparse_weight <= 1):
            raise ValueError("sparse_weight 必須在 0 到 1 之間")
        if abs(self.dense_weight + self.sparse_weight - 1.0) > 1e-6:
            raise ValueError("dense_weight 和 sparse_weight 的和必須等於 1")


@dataclass
class SearchFilter:
    """搜尋過濾器"""

    conditions: Dict[str, Any] = field(default_factory=dict)  # 過濾條件
    date_range: Optional[Tuple[str, str]] = None  # 日期範圍
    score_threshold: Optional[float] = None  # 分數閾值
    custom_filter: Optional[Callable[[Dict[str, Any]], bool]] = None  # 自訂過濾函數


@dataclass
class VectorCollection:
    """向量集合資訊"""

    name: str  # 集合名稱
    dimension: int  # 向量維度
    count: int  # 向量數量
    metadata_schema: Dict[str, str]  # 元資料結構描述
    created_at: str  # 建立時間
    updated_at: str  # 更新時間


class VectorStore(ABC):
    """向量儲存抽象基類

    定義所有向量儲存實作必須遵循的介面
    """

    def __init__(
        self,
        store_type: VectorStoreType,
        connection_string: Optional[str] = None,
        **kwargs,
    ):
        """初始化向量儲存

        Args:
            store_type: 儲存類型
            connection_string: 連線字串
            **kwargs: 其他配置參數
        """
        self.store_type = store_type
        self.connection_string = connection_string
        self.config = kwargs
        self.is_connected = False

        # 安全地獲取類型字符串
        if hasattr(store_type, "value"):
            type_str = store_type.value
        else:
            type_str = str(store_type)
        logger.info(f"初始化向量儲存: {type_str}")

    @abstractmethod
    async def connect(self) -> None:
        """建立連線"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """關閉連線"""
        pass

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        metadata_schema: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> bool:
        """建立向量集合

        Args:
            name: 集合名稱
            dimension: 向量維度
            metadata_schema: 元資料結構描述
            **kwargs: 其他參數

        Returns:
            bool: 是否建立成功
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """刪除向量集合

        Args:
            name: 集合名稱

        Returns:
            bool: 是否刪除成功
        """
        pass

    @abstractmethod
    async def list_collections(self) -> List[VectorCollection]:
        """列出所有集合"""
        pass

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """檢查集合是否存在"""
        pass

    @abstractmethod
    async def get_collection_info(self, name: str) -> Optional[VectorCollection]:
        """取得集合資訊"""
        pass

    @abstractmethod
    async def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: Union[List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """插入向量資料

        Args:
            collection_name: 集合名稱
            ids: 向量 ID 列表
            vectors: 向量列表或向量矩陣
            metadata: 元資料列表

        Returns:
            bool: 是否插入成功
        """
        pass

    @abstractmethod
    async def update_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: Union[List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """更新向量資料

        Args:
            collection_name: 集合名稱
            ids: 向量 ID 列表
            vectors: 向量列表或向量矩陣
            metadata: 元資料列表

        Returns:
            bool: 是否更新成功
        """
        pass

    @abstractmethod
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """刪除向量資料

        Args:
            collection_name: 集合名稱
            ids: 要刪除的向量 ID 列表

        Returns:
            bool: 是否刪除成功
        """
        pass

    @abstractmethod
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> VectorSearchResult:
        """向量相似性搜尋

        Args:
            collection_name: 集合名稱
            query_vector: 查詢向量
            k: 返回結果數量
            filter_conditions: 過濾條件
            include_embeddings: 是否包含向量資料

        Returns:
            VectorSearchResult: 搜尋結果
        """
        pass

    async def hybrid_search(
        self,
        collection_name: str,
        dense_vector: Optional[np.ndarray] = None,
        sparse_vector: Optional[Dict[str, float]] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        config: Optional[HybridSearchConfig] = None,
        search_filter: Optional[SearchFilter] = None,
        include_embeddings: bool = False,
    ) -> VectorSearchResult:
        """混合搜尋（密集+稀疏）

        Args:
            collection_name: 集合名稱
            dense_vector: 密集查詢向量
            sparse_vector: 稀疏查詢向量（詞彙-權重字典）
            query_text: 查詢文本（用於關鍵字搜尋）
            k: 返回結果數量
            config: 混合搜尋配置
            search_filter: 搜尋過濾器
            include_embeddings: 是否包含向量資料

        Returns:
            VectorSearchResult: 搜尋結果
        """
        if config is None:
            config = HybridSearchConfig()

        results = []

        # 密集向量搜尋
        if dense_vector is not None:
            dense_result = await self.search_vectors(
                collection_name=collection_name,
                query_vector=dense_vector,
                k=config.reranking_k,
                filter_conditions=search_filter.conditions if search_filter else None,
                include_embeddings=include_embeddings,
            )
            dense_result.search_type = SearchType.DENSE
            results.append((dense_result, config.dense_weight))

        # 稀疏向量搜尋（如果支援）
        if sparse_vector is not None or query_text is not None:
            sparse_result = await self._sparse_search(
                collection_name=collection_name,
                sparse_vector=sparse_vector,
                query_text=query_text,
                k=config.reranking_k,
                search_filter=search_filter,
                include_embeddings=include_embeddings,
            )
            if sparse_result:
                sparse_result.search_type = SearchType.SPARSE
                results.append((sparse_result, config.sparse_weight))

        if not results:
            return VectorSearchResult(
                ids=[],
                distances=[],
                similarities=[],
                metadata=[],
                embeddings=[] if include_embeddings else None,
                search_type=SearchType.HYBRID,
            )

        # 融合結果
        return self._fuse_search_results(
            results, k, config.reranking_method, config.alpha
        )

    async def _sparse_search(
        self,
        collection_name: str,
        sparse_vector: Optional[Dict[str, float]] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        search_filter: Optional[SearchFilter] = None,
        include_embeddings: bool = False,
    ) -> Optional[VectorSearchResult]:
        """稀疏向量搜尋（預設實作為關鍵字搜尋）

        子類別可以覆寫此方法以提供更好的稀疏搜尋實作
        """
        # 預設實作：簡單的關鍵字過濾
        if query_text is None and sparse_vector is None:
            return None

        # 這裡提供一個基本的關鍵字搜尋實作
        # 實際的稀疏搜尋應該由具體的向量儲存實作
        logger.warning("使用基本的關鍵字搜尋實作，建議子類別覆寫 _sparse_search 方法")

        # 返回空結果，讓子類別實作具體邏輯
        return VectorSearchResult(
            ids=[],
            distances=[],
            similarities=[],
            metadata=[],
            embeddings=[] if include_embeddings else None,
            search_type=SearchType.SPARSE,
        )

    def _fuse_search_results(
        self,
        results: List[Tuple[VectorSearchResult, float]],
        k: int,
        reranking_method: RerankingMethod,
        alpha: float = 60.0,
    ) -> VectorSearchResult:
        """融合多個搜尋結果"""
        if not results:
            return VectorSearchResult(
                ids=[],
                distances=[],
                similarities=[],
                metadata=[],
                search_type=SearchType.HYBRID,
            )

        if len(results) == 1:
            result, _ = results[0]
            result.search_type = SearchType.HYBRID
            return result

        # 收集所有候選項目
        all_candidates = {}

        for result, weight in results:
            for i, doc_id in enumerate(result.ids):
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = {
                        "id": doc_id,
                        "metadata": result.metadata[i],
                        "embedding": (
                            result.embeddings[i] if result.embeddings else None
                        ),
                        "scores": [],
                        "distances": [],
                        "ranks": [],
                    }

                all_candidates[doc_id]["scores"].append(result.similarities[i] * weight)
                all_candidates[doc_id]["distances"].append(result.distances[i])
                all_candidates[doc_id]["ranks"].append(i + 1)

        # 根據重新排序方法計算最終分數
        if reranking_method == RerankingMethod.RECIPROCAL_RANK_FUSION:
            final_scores = self._reciprocal_rank_fusion(all_candidates, alpha)
        elif reranking_method == RerankingMethod.WEIGHTED_SCORE:
            final_scores = self._weighted_score_fusion(all_candidates)
        else:
            final_scores = self._weighted_score_fusion(all_candidates)

        # 排序並取前 k 個結果
        sorted_candidates = sorted(
            final_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        # 構建最終結果
        final_ids = []
        final_similarities = []
        final_distances = []
        final_metadata = []
        final_embeddings = (
            []
            if any(c["embedding"] is not None for c in all_candidates.values())
            else None
        )
        final_reranking_scores = []

        for doc_id, score in sorted_candidates:
            candidate = all_candidates[doc_id]
            final_ids.append(doc_id)
            final_similarities.append(
                max(candidate["scores"]) if candidate["scores"] else 0.0
            )
            final_distances.append(
                min(candidate["distances"]) if candidate["distances"] else float("inf")
            )
            final_metadata.append(candidate["metadata"])
            final_reranking_scores.append(score)

            if final_embeddings is not None:
                final_embeddings.append(candidate["embedding"])

        return VectorSearchResult(
            ids=final_ids,
            distances=final_distances,
            similarities=final_similarities,
            metadata=final_metadata,
            embeddings=final_embeddings,
            search_type=SearchType.HYBRID,
            reranking_scores=final_reranking_scores,
        )

    def _reciprocal_rank_fusion(
        self, candidates: Dict[str, Dict], alpha: float = 60.0
    ) -> Dict[str, float]:
        """倒數排名融合"""
        scores = {}

        for doc_id, candidate in candidates.items():
            rrf_score = 0.0
            for rank in candidate["ranks"]:
                rrf_score += 1.0 / (alpha + rank)
            scores[doc_id] = rrf_score

        return scores

    def _weighted_score_fusion(self, candidates: Dict[str, Dict]) -> Dict[str, float]:
        """加權分數融合"""
        scores = {}

        for doc_id, candidate in candidates.items():
            weighted_score = sum(candidate["scores"])
            scores[doc_id] = weighted_score

        return scores

    @abstractmethod
    async def batch_search_vectors(
        self,
        collection_name: str,
        query_vectors: Union[List[np.ndarray], np.ndarray],
        k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> List[VectorSearchResult]:
        """批次向量相似性搜尋

        Args:
            collection_name: 集合名稱
            query_vectors: 查詢向量列表
            k: 返回結果數量
            filter_conditions: 過濾條件
            include_embeddings: 是否包含向量資料

        Returns:
            List[VectorSearchResult]: 搜尋結果列表
        """
        pass

    @abstractmethod
    async def get_vector_by_id(
        self, collection_name: str, vector_id: str, include_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        """根據 ID 取得向量資料

        Args:
            collection_name: 集合名稱
            vector_id: 向量 ID
            include_embedding: 是否包含向量資料

        Returns:
            Optional[Dict[str, Any]]: 向量資料
        """
        pass

    async def optimize_index(
        self, collection_name: str, optimization_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """優化向量索引

        Args:
            collection_name: 集合名稱
            optimization_params: 優化參數

        Returns:
            bool: 是否優化成功
        """
        # 預設實作：記錄警告並返回 True
        # 安全地獲取類型字符串
        if hasattr(self.store_type, "value"):
            type_str = self.store_type.value
        else:
            type_str = str(self.store_type)
        logger.warning(f"向量儲存 {type_str} 未實作索引優化功能")
        return True

    async def get_index_stats(self, collection_name: str) -> Dict[str, Any]:
        """取得索引統計資訊

        Args:
            collection_name: 集合名稱

        Returns:
            Dict[str, Any]: 索引統計資訊
        """
        try:
            collection_info = await self.get_collection_info(collection_name)
            if not collection_info:
                return {"error": f"集合 {collection_name} 不存在"}

            return {
                "collection_name": collection_name,
                "vector_count": collection_info.count,
                "dimension": collection_info.dimension,
                "index_type": "default",
                "optimization_status": "unknown",
            }

        except Exception as e:
            return {"error": str(e)}

    async def semantic_search(
        self,
        collection_name: str,
        query_text: str,
        k: int = 10,
        embedding_function: Optional[callable] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> VectorSearchResult:
        """語義搜尋

        Args:
            collection_name: 集合名稱
            query_text: 查詢文本
            k: 返回結果數量
            embedding_function: 文本向量化函數
            filter_conditions: 過濾條件
            include_embeddings: 是否包含向量資料

        Returns:
            VectorSearchResult: 搜尋結果
        """
        if embedding_function is None:
            raise ValueError("語義搜尋需要提供 embedding_function")

        # 將查詢文本轉換為向量
        query_vector = embedding_function(query_text)

        # 執行向量搜尋
        result = await self.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            k=k,
            filter_conditions=filter_conditions,
            include_embeddings=include_embeddings,
        )

        result.search_type = SearchType.SEMANTIC
        return result

    async def multi_vector_search(
        self,
        collection_name: str,
        query_vectors: List[np.ndarray],
        weights: Optional[List[float]] = None,
        k: int = 10,
        fusion_method: str = "weighted_sum",
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> VectorSearchResult:
        """多向量搜尋

        Args:
            collection_name: 集合名稱
            query_vectors: 查詢向量列表
            weights: 向量權重列表
            k: 返回結果數量
            fusion_method: 融合方法 ("weighted_sum", "max", "mean")
            filter_conditions: 過濾條件
            include_embeddings: 是否包含向量資料

        Returns:
            VectorSearchResult: 搜尋結果
        """
        if not query_vectors:
            return VectorSearchResult(
                ids=[],
                distances=[],
                similarities=[],
                metadata=[],
                embeddings=[] if include_embeddings else None,
                search_type=SearchType.DENSE,
            )

        if weights is None:
            weights = [1.0] * len(query_vectors)
        elif len(weights) != len(query_vectors):
            raise ValueError("權重數量必須與查詢向量數量相同")

        # 執行多個向量搜尋
        all_results = []
        for i, query_vector in enumerate(query_vectors):
            result = await self.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                k=k * 2,  # 取更多結果用於融合
                filter_conditions=filter_conditions,
                include_embeddings=include_embeddings,
            )
            all_results.append((result, weights[i]))

        # 融合結果
        return self._fuse_multi_vector_results(all_results, k, fusion_method)

    def _fuse_multi_vector_results(
        self,
        results: List[Tuple[VectorSearchResult, float]],
        k: int,
        fusion_method: str = "weighted_sum",
    ) -> VectorSearchResult:
        """融合多向量搜尋結果"""
        if not results:
            return VectorSearchResult(
                ids=[],
                distances=[],
                similarities=[],
                metadata=[],
                search_type=SearchType.DENSE,
            )

        if len(results) == 1:
            result, _ = results[0]
            return result

        # 收集所有候選項目
        all_candidates = {}

        for result, weight in results:
            for i, doc_id in enumerate(result.ids):
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = {
                        "id": doc_id,
                        "metadata": result.metadata[i],
                        "embedding": (
                            result.embeddings[i] if result.embeddings else None
                        ),
                        "weighted_similarities": [],
                        "distances": [],
                    }

                weighted_sim = result.similarities[i] * weight
                all_candidates[doc_id]["weighted_similarities"].append(weighted_sim)
                all_candidates[doc_id]["distances"].append(result.distances[i])

        # 根據融合方法計算最終分數
        final_scores = {}
        for doc_id, candidate in all_candidates.items():
            if fusion_method == "weighted_sum":
                final_scores[doc_id] = sum(candidate["weighted_similarities"])
            elif fusion_method == "max":
                final_scores[doc_id] = max(candidate["weighted_similarities"])
            elif fusion_method == "mean":
                final_scores[doc_id] = sum(candidate["weighted_similarities"]) / len(
                    candidate["weighted_similarities"]
                )
            else:
                final_scores[doc_id] = sum(candidate["weighted_similarities"])

        # 排序並取前 k 個結果
        sorted_candidates = sorted(
            final_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        # 構建最終結果
        final_ids = []
        final_similarities = []
        final_distances = []
        final_metadata = []
        final_embeddings = (
            []
            if any(c["embedding"] is not None for c in all_candidates.values())
            else None
        )

        for doc_id, score in sorted_candidates:
            candidate = all_candidates[doc_id]
            final_ids.append(doc_id)
            final_similarities.append(score)
            final_distances.append(
                min(candidate["distances"]) if candidate["distances"] else float("inf")
            )
            final_metadata.append(candidate["metadata"])

            if final_embeddings is not None:
                final_embeddings.append(candidate["embedding"])

        return VectorSearchResult(
            ids=final_ids,
            distances=final_distances,
            similarities=final_similarities,
            metadata=final_metadata,
            embeddings=final_embeddings,
            search_type=SearchType.DENSE,
        )

    async def health_check(self) -> Dict[str, Any]:
        """健康狀態檢查"""
        try:
            if not self.is_connected:
                return {
                    "status": "disconnected",
                    "store_type": (
                        self.store_type.value
                        if hasattr(self.store_type, "value")
                        else str(self.store_type)
                    ),
                    "error": "未建立連線",
                }

            # 測試基本操作
            collections = await self.list_collections()

            return {
                "status": "healthy",
                "store_type": (
                    self.store_type.value
                    if hasattr(self.store_type, "value")
                    else str(self.store_type)
                ),
                "collections_count": len(collections),
                "is_connected": self.is_connected,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "store_type": (
                    self.store_type.value
                    if hasattr(self.store_type, "value")
                    else str(self.store_type)
                ),
                "error": str(e),
            }

    def _validate_vectors(
        self,
        vectors: Union[List[np.ndarray], np.ndarray],
        expected_dimension: Optional[int] = None,
    ) -> np.ndarray:
        """驗證向量資料格式"""
        if isinstance(vectors, list):
            if not vectors:
                raise ValueError("向量列表不能為空")

            # 檢查所有向量維度是否一致
            dimensions = [len(v) for v in vectors]
            if not all(dim == dimensions[0] for dim in dimensions):
                raise ValueError("所有向量維度必須一致")

            # 轉換為矩陣
            vectors = np.array(vectors)

        if not isinstance(vectors, np.ndarray):
            raise ValueError("向量必須是 numpy array 或 numpy array 列表")

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim != 2:
            raise ValueError("向量必須是一維或二維陣列")

        # 檢查維度
        if expected_dimension is not None and vectors.shape[1] != expected_dimension:
            raise ValueError(
                f"向量維度不匹配: 預期 {expected_dimension}, 實際 {vectors.shape[1]}"
            )

        return vectors

    def _validate_ids_and_vectors(
        self, ids: List[str], vectors: Union[List[np.ndarray], np.ndarray]
    ) -> Tuple[List[str], np.ndarray]:
        """驗證 ID 和向量的對應關係"""
        if not ids:
            raise ValueError("ID 列表不能為空")

        # 驗證向量
        vectors = self._validate_vectors(vectors)

        # 檢查數量是否匹配
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ID 數量與向量數量不匹配: {len(ids)} vs {vectors.shape[0]}"
            )

        # 檢查 ID 是否有重複
        if len(set(ids)) != len(ids):
            raise ValueError("ID 列表中不能有重複項目")

        # 檢查 ID 是否為空
        empty_ids = [
            i for i, id_val in enumerate(ids) if not id_val or not id_val.strip()
        ]
        if empty_ids:
            raise ValueError(f"ID 不能為空，位置: {empty_ids}")

        return ids, vectors

    async def __aenter__(self):
        """異步上下文管理器入口"""
        if not self.is_connected:
            await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器出口"""
        if self.is_connected:
            await self.disconnect()


class VectorStoreError(Exception):
    """向量儲存相關異常"""

    pass


class ConnectionError(VectorStoreError):
    """連線異常"""

    pass


class CollectionError(VectorStoreError):
    """集合操作異常"""

    pass


class VectorOperationError(VectorStoreError):
    """向量操作異常"""

    pass
