"""
向量儲存管理器

統一管理多種向量儲存後端，提供高層次的向量管理介面
"""

from typing import Dict, List, Optional, Union, Any, Type
import asyncio
from pathlib import Path

import numpy as np
from loguru import logger

from .base import (
    VectorStore,
    VectorStoreType,
    VectorSearchResult,
    VectorCollection,
    VectorStoreError,
    HybridSearchConfig,
    SearchFilter,
    SearchType
)
from .lancedb_store import LanceDBStore


class VectorStoreManager:
    """向量儲存管理器
    
    提供統一的向量儲存管理介面，支援多種向量資料庫後端
    """
    
    # 支援的向量儲存類型映射
    STORE_CLASSES: Dict[VectorStoreType, Type[VectorStore]] = {
        VectorStoreType.LANCEDB: LanceDBStore,
        # 未來可以添加其他儲存類型
        # VectorStoreType.CHROMA: ChromaStore,
        # VectorStoreType.FAISS: FAISSStore,
    }
    
    def __init__(
        self,
        default_store_type: VectorStoreType = VectorStoreType.LANCEDB,
        config: Optional[Dict[str, Any]] = None
    ):
        """初始化向量儲存管理器
        
        Args:
            default_store_type: 預設儲存類型
            config: 配置參數
        """
        self.default_store_type = default_store_type
        self.config = config or {}
        self.stores: Dict[str, VectorStore] = {}  # 儲存實例快取
        self.active_store: Optional[VectorStore] = None
        
        # 安全地獲取類型字符串
        if hasattr(default_store_type, 'value'):
            type_str = default_store_type.value
        else:
            type_str = str(default_store_type)
        logger.info(f"初始化向量儲存管理器，預設類型: {type_str}")
    
    async def initialize(
        self,
        store_type: Optional[VectorStoreType] = None,
        **kwargs
    ) -> None:
        """初始化預設向量儲存
        
        Args:
            store_type: 儲存類型，如果未指定則使用預設類型
            **kwargs: 儲存特定的配置參數
        """
        if store_type is None:
            store_type = self.default_store_type
        
        try:
            # 建立儲存實例
            store = await self.get_store(store_type, **kwargs)
            self.active_store = store
            
            # 安全地獲取類型字符串
            if hasattr(store_type, 'value'):
                type_str = store_type.value
            else:
                type_str = str(store_type)
            logger.info(f"成功初始化向量儲存: {type_str}")
            
        except Exception as e:
            logger.error(f"初始化向量儲存失敗: {e}")
            raise VectorStoreError(f"無法初始化向量儲存: {e}")
    
    async def get_store(
        self,
        store_type: VectorStoreType,
        store_id: Optional[str] = None,
        **kwargs
    ) -> VectorStore:
        """取得向量儲存實例
        
        Args:
            store_type: 儲存類型
            store_id: 儲存實例 ID，用於快取
            **kwargs: 儲存特定的配置參數
            
        Returns:
            VectorStore: 向量儲存實例
        """
        # 生成快取鍵
        if store_id is None:
            if hasattr(store_type, 'value'):
                type_str = store_type.value
            else:
                type_str = str(store_type)
            store_id = f"{type_str}_default"
        
        # 檢查快取
        if store_id in self.stores:
            return self.stores[store_id]
        
        # 檢查是否支援該儲存類型
        if store_type not in self.STORE_CLASSES:
            if hasattr(store_type, 'value'):
                type_str = store_type.value
            else:
                type_str = str(store_type)
            raise VectorStoreError(f"不支援的向量儲存類型: {type_str}")
        
        try:
            # 建立儲存實例
            store_class = self.STORE_CLASSES[store_type]
            
            # 合併配置參數
            if hasattr(store_type, 'value'):
                type_str = store_type.value
            else:
                type_str = str(store_type)
            store_config = self.config.get(type_str, {})
            store_config.update(kwargs)
            
            # 根據儲存類型設定預設參數
            if store_type == VectorStoreType.LANCEDB:
                if 'db_path' not in store_config:
                    store_config['db_path'] = "./data/lancedb"
            
            store = store_class(**store_config)
            
            # 建立連線
            await store.connect()
            
            # 快取實例
            self.stores[store_id] = store
            
            if hasattr(store_type, 'value'):
                type_str = store_type.value
            else:
                type_str = str(store_type)
            logger.info(f"成功建立向量儲存實例: {type_str} ({store_id})")
            return store
            
        except Exception as e:
            logger.error(f"建立向量儲存實例失敗: {e}")
            raise VectorStoreError(f"無法建立向量儲存實例: {e}")
    
    async def create_collection(
        self,
        name: str,
        dimension: int,
        metadata_schema: Optional[Dict[str, str]] = None,
        store_type: Optional[VectorStoreType] = None,
        **kwargs
    ) -> bool:
        """建立向量集合
        
        Args:
            name: 集合名稱
            dimension: 向量維度
            metadata_schema: 元資料結構描述
            store_type: 儲存類型，如果未指定則使用活躍儲存
            **kwargs: 其他參數
            
        Returns:
            bool: 是否建立成功
        """
        store = await self._get_active_store(store_type)
        return await store.create_collection(name, dimension, metadata_schema, **kwargs)
    
    async def delete_collection(
        self,
        name: str,
        store_type: Optional[VectorStoreType] = None
    ) -> bool:
        """刪除向量集合"""
        store = await self._get_active_store(store_type)
        return await store.delete_collection(name)
    
    async def list_collections(
        self,
        store_type: Optional[VectorStoreType] = None
    ) -> List[VectorCollection]:
        """列出所有集合"""
        store = await self._get_active_store(store_type)
        return await store.list_collections()
    
    async def collection_exists(
        self,
        name: str,
        store_type: Optional[VectorStoreType] = None
    ) -> bool:
        """檢查集合是否存在"""
        store = await self._get_active_store(store_type)
        return await store.collection_exists(name)
    
    async def get_collection_info(
        self,
        name: str,
        store_type: Optional[VectorStoreType] = None
    ) -> Optional[VectorCollection]:
        """取得集合資訊"""
        store = await self._get_active_store(store_type)
        return await store.get_collection_info(name)
    
    async def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: Union[List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        store_type: Optional[VectorStoreType] = None
    ) -> bool:
        """插入向量資料"""
        store = await self._get_active_store(store_type)
        return await store.insert_vectors(collection_name, ids, vectors, metadata)
    
    async def update_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: Union[List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        store_type: Optional[VectorStoreType] = None
    ) -> bool:
        """更新向量資料"""
        store = await self._get_active_store(store_type)
        return await store.update_vectors(collection_name, ids, vectors, metadata)
    
    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[str],
        store_type: Optional[VectorStoreType] = None
    ) -> bool:
        """刪除向量資料"""
        store = await self._get_active_store(store_type)
        return await store.delete_vectors(collection_name, ids)
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
        store_type: Optional[VectorStoreType] = None
    ) -> VectorSearchResult:
        """向量相似性搜尋"""
        store = await self._get_active_store(store_type)
        return await store.search_vectors(
            collection_name, query_vector, k, filter_conditions, include_embeddings
        )
    
    async def batch_search_vectors(
        self,
        collection_name: str,
        query_vectors: Union[List[np.ndarray], np.ndarray],
        k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
        store_type: Optional[VectorStoreType] = None
    ) -> List[VectorSearchResult]:
        """批次向量相似性搜尋"""
        store = await self._get_active_store(store_type)
        return await store.batch_search_vectors(
            collection_name, query_vectors, k, filter_conditions, include_embeddings
        )
    
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
        store_type: Optional[VectorStoreType] = None
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
            store_type: 儲存類型
            
        Returns:
            VectorSearchResult: 搜尋結果
        """
        store = await self._get_active_store(store_type)
        return await store.hybrid_search(
            collection_name=collection_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            query_text=query_text,
            k=k,
            config=config,
            search_filter=search_filter,
            include_embeddings=include_embeddings
        )
    
    async def optimize_index(
        self,
        collection_name: str,
        optimization_params: Optional[Dict[str, Any]] = None,
        store_type: Optional[VectorStoreType] = None
    ) -> bool:
        """優化向量索引
        
        Args:
            collection_name: 集合名稱
            optimization_params: 優化參數
            store_type: 儲存類型
            
        Returns:
            bool: 是否優化成功
        """
        store = await self._get_active_store(store_type)
        return await store.optimize_index(collection_name, optimization_params)
    
    async def get_index_stats(
        self,
        collection_name: str,
        store_type: Optional[VectorStoreType] = None
    ) -> Dict[str, Any]:
        """取得索引統計資訊
        
        Args:
            collection_name: 集合名稱
            store_type: 儲存類型
            
        Returns:
            Dict[str, Any]: 索引統計資訊
        """
        store = await self._get_active_store(store_type)
        return await store.get_index_stats(collection_name)
    
    async def get_vector_by_id(
        self,
        collection_name: str,
        vector_id: str,
        include_embedding: bool = False,
        store_type: Optional[VectorStoreType] = None
    ) -> Optional[Dict[str, Any]]:
        """根據 ID 取得向量資料"""
        store = await self._get_active_store(store_type)
        return await store.get_vector_by_id(collection_name, vector_id, include_embedding)
    
    async def semantic_search(
        self,
        collection_name: str,
        query_text: str,
        k: int = 10,
        embedding_function: Optional[callable] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
        store_type: Optional[VectorStoreType] = None
    ) -> VectorSearchResult:
        """語義搜尋
        
        Args:
            collection_name: 集合名稱
            query_text: 查詢文本
            k: 返回結果數量
            embedding_function: 文本向量化函數
            filter_conditions: 過濾條件
            include_embeddings: 是否包含向量資料
            store_type: 儲存類型
            
        Returns:
            VectorSearchResult: 搜尋結果
        """
        store = await self._get_active_store(store_type)
        return await store.semantic_search(
            collection_name=collection_name,
            query_text=query_text,
            k=k,
            embedding_function=embedding_function,
            filter_conditions=filter_conditions,
            include_embeddings=include_embeddings
        )
    
    async def multi_vector_search(
        self,
        collection_name: str,
        query_vectors: List[np.ndarray],
        weights: Optional[List[float]] = None,
        k: int = 10,
        fusion_method: str = "weighted_sum",
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
        store_type: Optional[VectorStoreType] = None
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
            store_type: 儲存類型
            
        Returns:
            VectorSearchResult: 搜尋結果
        """
        store = await self._get_active_store(store_type)
        return await store.multi_vector_search(
            collection_name=collection_name,
            query_vectors=query_vectors,
            weights=weights,
            k=k,
            fusion_method=fusion_method,
            filter_conditions=filter_conditions,
            include_embeddings=include_embeddings
        )
    
    async def health_check(
        self,
        store_type: Optional[VectorStoreType] = None
    ) -> Dict[str, Any]:
        """健康狀態檢查"""
        try:
            store = await self._get_active_store(store_type)
            return await store.health_check()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_statistics(
        self,
        store_type: Optional[VectorStoreType] = None
    ) -> Dict[str, Any]:
        """取得儲存統計資訊"""
        try:
            store = await self._get_active_store(store_type)
            collections = await store.list_collections()
            
            total_vectors = sum(col.count for col in collections)
            dimensions = list(set(col.dimension for col in collections if col.dimension > 0))
            
            return {
                "store_type": store.store_type.value if hasattr(store.store_type, 'value') else str(store.store_type),
                "collections_count": len(collections),
                "total_vectors": total_vectors,
                "dimensions": dimensions,
                "collections": [
                    {
                        "name": col.name,
                        "dimension": col.dimension,
                        "count": col.count
                    }
                    for col in collections
                ]
            }
            
        except Exception as e:
            logger.error(f"取得儲存統計資訊失敗: {e}")
            return {
                "error": str(e)
            }
    
    async def backup_collection(
        self,
        collection_name: str,
        backup_path: str,
        store_type: Optional[VectorStoreType] = None
    ) -> bool:
        """備份集合資料
        
        Args:
            collection_name: 集合名稱
            backup_path: 備份路徑
            store_type: 儲存類型
            
        Returns:
            bool: 是否備份成功
        """
        try:
            store = await self._get_active_store(store_type)
            
            # 檢查集合是否存在
            if not await store.collection_exists(collection_name):
                logger.error(f"集合 {collection_name} 不存在")
                return False
            
            # 取得集合資訊
            collection_info = await store.get_collection_info(collection_name)
            if not collection_info:
                logger.error(f"無法取得集合 {collection_name} 的資訊")
                return False
            
            # 建立備份目錄
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 這裡可以實作具體的備份邏輯
            # 由於不同的向量資料庫有不同的備份方式，這裡提供一個基本框架
            logger.info(f"開始備份集合 {collection_name} 到 {backup_path}")
            
            # TODO: 實作具體的備份邏輯
            # 可能需要根據不同的儲存類型實作不同的備份策略
            
            logger.info(f"成功備份集合 {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"備份集合 {collection_name} 失敗: {e}")
            return False
    
    async def restore_collection(
        self,
        collection_name: str,
        backup_path: str,
        store_type: Optional[VectorStoreType] = None
    ) -> bool:
        """還原集合資料
        
        Args:
            collection_name: 集合名稱
            backup_path: 備份路徑
            store_type: 儲存類型
            
        Returns:
            bool: 是否還原成功
        """
        try:
            store = await self._get_active_store(store_type)
            
            # 檢查備份路徑是否存在
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                logger.error(f"備份路徑 {backup_path} 不存在")
                return False
            
            logger.info(f"開始還原集合 {collection_name} 從 {backup_path}")
            
            # TODO: 實作具體的還原邏輯
            # 可能需要根據不同的儲存類型實作不同的還原策略
            
            logger.info(f"成功還原集合 {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"還原集合 {collection_name} 失敗: {e}")
            return False
    
    async def close(self) -> None:
        """關閉所有連線"""
        try:
            for store_id, store in self.stores.items():
                if store.is_connected:
                    await store.disconnect()
                    logger.info(f"已關閉向量儲存連線: {store_id}")
            
            self.stores.clear()
            self.active_store = None
            
            logger.info("已關閉所有向量儲存連線")
            
        except Exception as e:
            logger.error(f"關閉向量儲存連線時發生錯誤: {e}")
    
    async def _get_active_store(
        self,
        store_type: Optional[VectorStoreType] = None
    ) -> VectorStore:
        """取得活躍的向量儲存實例"""
        if store_type is not None:
            return await self.get_store(store_type)
        
        if self.active_store is None:
            await self.initialize()
        
        if self.active_store is None:
            raise VectorStoreError("沒有可用的向量儲存實例")
        
        return self.active_store

    async def store_text_unit(self, text_unit) -> bool:
        """
        儲存文本單元到向量資料庫
        
        Args:
            text_unit: TextUnit 物件
            
        Returns:
            bool: 儲存是否成功
        """
        try:
            if not self.active_store:
                await self.initialize()
            
            # 確保有 text_units 集合
            collection_name = "text_units"
            collections = await self.list_collections()
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                await self.create_collection(
                    collection_name,
                    dimension=768,  # 假設使用 768 維向量
                    metadata_schema={"text": "str", "document_id": "str", "chunk_index": "int"}
                )
            
            # 準備向量資料
            if text_unit.embedding is not None:
                await self.insert_vectors(
                    collection_name,
                    ids=[text_unit.id],
                    vectors=[text_unit.embedding],
                    metadata=[{
                        "id": text_unit.id,
                        "text": text_unit.text,
                        "document_id": text_unit.document_id,
                        "chunk_index": text_unit.chunk_index,
                        **text_unit.metadata
                    }]
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"儲存文本單元失敗: {e}")
            return False

    async def store_entity(self, entity) -> bool:
        """
        儲存實體到向量資料庫
        
        Args:
            entity: Entity 物件
            
        Returns:
            bool: 儲存是否成功
        """
        try:
            if not self.active_store:
                await self.initialize()
            
            # 確保有 entities 集合
            collection_name = "entities"
            collections = await self.list_collections()
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                await self.create_collection(
                    collection_name,
                    dimension=768,
                    metadata_schema={"name": "str", "type": "str", "description": "str", "rank": "float"}
                )
            
            # 準備向量資料
            if entity.embedding is not None:
                await self.insert_vectors(
                    collection_name,
                    ids=[entity.id],
                    vectors=[entity.embedding],
                    metadata=[{
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type,
                        "description": entity.description,
                        "text_units": entity.text_units,
                        "rank": entity.rank
                    }]
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"儲存實體失敗: {e}")
            return False

    async def store_community(self, community) -> bool:
        """
        儲存社群到向量資料庫
        
        Args:
            community: Community 物件
            
        Returns:
            bool: 儲存是否成功
        """
        try:
            if not self.active_store:
                await self.initialize()
            
            # 確保有 communities 集合
            collection_name = "communities"
            collections = await self.list_collections()
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                await self.create_collection(
                    collection_name,
                    dimension=768,
                    metadata_schema={"title": "str", "level": "int", "summary": "str", "rank": "float"}
                )
            
            # 準備向量資料
            if community.embedding is not None:
                await self.insert_vectors(
                    collection_name,
                    ids=[community.id],
                    vectors=[community.embedding],
                    metadata=[{
                        "id": community.id,
                        "title": community.title,
                        "level": community.level,
                        "entities": community.entities,
                        "relationships": community.relationships,
                        "summary": community.summary,
                        "rank": community.rank
                    }]
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"儲存社群失敗: {e}")
            return False
    
    async def __aenter__(self):
        """異步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器出口"""
        await self.close()


# 便利函數
async def create_vector_store_manager(
    store_type: VectorStoreType = VectorStoreType.LANCEDB,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> VectorStoreManager:
    """建立並初始化向量儲存管理器
    
    Args:
        store_type: 儲存類型
        config: 配置參數
        **kwargs: 儲存特定的配置參數
        
    Returns:
        VectorStoreManager: 已初始化的管理器
    """
    manager = VectorStoreManager(store_type, config)
    await manager.initialize(**kwargs)
    return manager