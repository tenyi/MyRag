"""
LanceDB 向量儲存實作
"""

import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
import asyncio

import numpy as np
import lancedb
from lancedb.table import Table
from loguru import logger

from .base import (
    VectorStore,
    VectorStoreType,
    VectorSearchResult,
    VectorCollection,
    VectorStoreError,
    ConnectionError,
    CollectionError,
    VectorOperationError,
    SearchType,
    HybridSearchConfig,
    SearchFilter,
    RerankingMethod
)


class LanceDBStore(VectorStore):
    """LanceDB 向量儲存實作"""
    
    def __init__(
        self,
        db_path: str = "./data/lancedb",
        **kwargs
    ):
        """初始化 LanceDB 儲存
        
        Args:
            db_path: 資料庫路徑
            **kwargs: 其他配置參數
        """
        super().__init__(VectorStoreType.LANCEDB, db_path, **kwargs)
        self.db_path = db_path
        self.db = None
        self._tables_cache = {}  # 快取表格物件
        
        logger.info(f"初始化 LanceDB 儲存，路徑: {db_path}")
    
    async def connect(self) -> None:
        """建立連線"""
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # 連接到 LanceDB
            self.db = lancedb.connect(self.db_path)
            self.is_connected = True
            
            logger.info(f"成功連接到 LanceDB: {self.db_path}")
            
        except Exception as e:
            logger.error(f"連接 LanceDB 失敗: {e}")
            raise ConnectionError(f"無法連接到 LanceDB: {e}")
    
    async def disconnect(self) -> None:
        """關閉連線"""
        try:
            if self.db is not None:
                # LanceDB 不需要顯式關閉連線
                self.db = None
                self._tables_cache.clear()
                self.is_connected = False
                
                logger.info("已斷開 LanceDB 連線")
                
        except Exception as e:
            logger.error(f"斷開 LanceDB 連線時發生錯誤: {e}")
    
    async def create_collection(
        self,
        name: str,
        dimension: int,
        metadata_schema: Optional[Dict[str, str]] = None,
        **kwargs
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
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            # 檢查集合是否已存在
            if await self.collection_exists(name):
                logger.warning(f"集合 {name} 已存在")
                return False
            
            # 建立初始資料結構
            # LanceDB 需要至少一筆資料來建立表格
            import pandas as pd
            
            initial_data = [{
                "id": "__init__",
                "vector": np.zeros(dimension, dtype=np.float32),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }]
            
            # 添加元資料欄位
            if metadata_schema:
                for field_name, field_type in metadata_schema.items():
                    if field_type == "str":
                        initial_data[0][field_name] = ""
                    elif field_type == "int":
                        initial_data[0][field_name] = 0
                    elif field_type == "float":
                        initial_data[0][field_name] = 0.0
                    elif field_type == "bool":
                        initial_data[0][field_name] = False
                    else:
                        initial_data[0][field_name] = None
            
            # 建立表格
            table = self.db.create_table(name, initial_data)
            
            # 刪除初始資料
            table.delete("id = '__init__'")
            
            # 快取表格物件
            self._tables_cache[name] = table
            
            logger.info(f"成功建立集合: {name}, 維度: {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"建立集合 {name} 失敗: {e}")
            raise CollectionError(f"無法建立集合 {name}: {e}")
    
    async def delete_collection(self, name: str) -> bool:
        """刪除向量集合"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            if not await self.collection_exists(name):
                logger.warning(f"集合 {name} 不存在")
                return False
            
            # 刪除表格
            self.db.drop_table(name)
            
            # 從快取中移除
            if name in self._tables_cache:
                del self._tables_cache[name]
            
            logger.info(f"成功刪除集合: {name}")
            return True
            
        except Exception as e:
            logger.error(f"刪除集合 {name} 失敗: {e}")
            raise CollectionError(f"無法刪除集合 {name}: {e}")
    
    async def list_collections(self) -> List[VectorCollection]:
        """列出所有集合"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            table_names = self.db.table_names()
            collections = []
            
            for name in table_names:
                collection_info = await self.get_collection_info(name)
                if collection_info:
                    collections.append(collection_info)
            
            return collections
            
        except Exception as e:
            logger.error(f"列出集合失敗: {e}")
            raise CollectionError(f"無法列出集合: {e}")
    
    async def collection_exists(self, name: str) -> bool:
        """檢查集合是否存在"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            return name in self.db.table_names()
        except Exception as e:
            logger.error(f"檢查集合 {name} 是否存在時失敗: {e}")
            return False
    
    async def get_collection_info(self, name: str) -> Optional[VectorCollection]:
        """取得集合資訊"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            if not await self.collection_exists(name):
                return None
            
            table = self._get_table(name)
            
            # 取得表格統計資訊
            count = table.count_rows()
            
            # 取得向量維度（從第一筆資料推斷）
            dimension = 0
            if count > 0:
                sample = table.head(1).to_pandas()
                if 'vector' in sample.columns and len(sample) > 0:
                    vector_data = sample['vector'].iloc[0]
                    if isinstance(vector_data, np.ndarray):
                        dimension = len(vector_data)
                    elif hasattr(vector_data, '__len__'):
                        dimension = len(vector_data)
            
            # 取得元資料結構
            schema = table.schema
            metadata_schema = {}
            for field in schema:
                if field.name not in ['id', 'vector', 'created_at', 'updated_at']:
                    metadata_schema[field.name] = str(field.type)
            
            return VectorCollection(
                name=name,
                dimension=dimension,
                count=count,
                metadata_schema=metadata_schema,
                created_at=datetime.now().isoformat(),  # LanceDB 沒有建立時間，使用當前時間
                updated_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"取得集合 {name} 資訊失敗: {e}")
            return None
    
    async def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: Union[List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """插入向量資料"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            # 驗證輸入
            ids, vectors = self._validate_ids_and_vectors(ids, vectors)
            
            if not await self.collection_exists(collection_name):
                raise CollectionError(f"集合 {collection_name} 不存在")
            
            table = self._get_table(collection_name)
            
            # 準備資料 - LanceDB 需要記錄列表格式
            current_time = datetime.now().isoformat()
            records = []
            
            for i, (vector_id, vector) in enumerate(zip(ids, vectors)):
                record = {
                    "id": vector_id,
                    "vector": vector.astype(np.float32),
                    "created_at": current_time,
                    "updated_at": current_time
                }
                
                # 添加元資料
                if metadata and i < len(metadata):
                    record.update(metadata[i])
                
                records.append(record)
            
            # 插入資料
            table.add(records)
            
            logger.info(f"成功插入 {len(ids)} 個向量到集合 {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"插入向量到集合 {collection_name} 失敗: {e}")
            raise VectorOperationError(f"無法插入向量: {e}")
    
    async def update_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: Union[List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """更新向量資料"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            # 驗證輸入
            ids, vectors = self._validate_ids_and_vectors(ids, vectors)
            
            if not await self.collection_exists(collection_name):
                raise CollectionError(f"集合 {collection_name} 不存在")
            
            table = self._get_table(collection_name)
            
            # LanceDB 的更新策略：先刪除再插入
            # 刪除現有資料
            for vector_id in ids:
                table.delete(f"id = '{vector_id}'")
            
            # 插入新資料
            return await self.insert_vectors(collection_name, ids, vectors, metadata)
            
        except Exception as e:
            logger.error(f"更新集合 {collection_name} 中的向量失敗: {e}")
            raise VectorOperationError(f"無法更新向量: {e}")
    
    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """刪除向量資料"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            if not await self.collection_exists(collection_name):
                raise CollectionError(f"集合 {collection_name} 不存在")
            
            table = self._get_table(collection_name)
            
            # 批次刪除
            for vector_id in ids:
                table.delete(f"id = '{vector_id}'")
            
            logger.info(f"成功從集合 {collection_name} 刪除 {len(ids)} 個向量")
            return True
            
        except Exception as e:
            logger.error(f"從集合 {collection_name} 刪除向量失敗: {e}")
            raise VectorOperationError(f"無法刪除向量: {e}")
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> VectorSearchResult:
        """向量相似性搜尋"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            if not await self.collection_exists(collection_name):
                raise CollectionError(f"集合 {collection_name} 不存在")
            
            table = self._get_table(collection_name)
            
            # 準備查詢向量
            if query_vector.ndim != 1:
                raise ValueError("查詢向量必須是一維陣列")
            
            query_vector = query_vector.astype(np.float32)
            
            # 建立搜尋查詢
            search_query = table.search(query_vector)
            
            # 應用過濾條件
            if filter_conditions:
                for field, value in filter_conditions.items():
                    if isinstance(value, str):
                        search_query = search_query.where(f"{field} = '{value}'")
                    elif isinstance(value, (int, float)):
                        search_query = search_query.where(f"{field} = {value}")
                    elif isinstance(value, list):
                        # 支援 IN 查詢
                        if all(isinstance(v, str) for v in value):
                            value_str = "', '".join(value)
                            search_query = search_query.where(f"{field} IN ('{value_str}')")
                        else:
                            value_str = ", ".join(str(v) for v in value)
                            search_query = search_query.where(f"{field} IN ({value_str})")
            
            # 執行搜尋
            results = search_query.limit(k).to_pandas()
            
            if len(results) == 0:
                return VectorSearchResult(
                    ids=[],
                    distances=[],
                    similarities=[],
                    metadata=[],
                    embeddings=[] if include_embeddings else None,
                    search_type=SearchType.DENSE
                )
            
            # 提取結果
            ids = results['id'].tolist()
            distances = results['_distance'].tolist()
            
            # 將距離轉換為相似度分數（距離越小相似度越高）
            similarities = [1.0 / (1.0 + dist) for dist in distances]
            
            # 提取元資料
            metadata = []
            for _, row in results.iterrows():
                meta = {}
                for col in results.columns:
                    if col not in ['id', 'vector', '_distance', 'created_at', 'updated_at']:
                        meta[col] = row[col]
                metadata.append(meta)
            
            # 提取向量（如果需要）
            embeddings = None
            if include_embeddings:
                embeddings = [np.array(vec) for vec in results['vector'].tolist()]
            
            return VectorSearchResult(
                ids=ids,
                distances=distances,
                similarities=similarities,
                metadata=metadata,
                embeddings=embeddings,
                search_type=SearchType.DENSE
            )
            
        except Exception as e:
            logger.error(f"在集合 {collection_name} 中搜尋向量失敗: {e}")
            raise VectorOperationError(f"無法搜尋向量: {e}")
    
    async def _sparse_search(
        self,
        collection_name: str,
        sparse_vector: Optional[Dict[str, float]] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        search_filter: Optional[SearchFilter] = None,
        include_embeddings: bool = False
    ) -> Optional[VectorSearchResult]:
        """LanceDB 的稀疏搜尋實作（基於關鍵字匹配）"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            if not await self.collection_exists(collection_name):
                raise CollectionError(f"集合 {collection_name} 不存在")
            
            # 如果沒有查詢文本，嘗試從稀疏向量中提取關鍵字
            if query_text is None and sparse_vector:
                # 取權重最高的詞作為關鍵字
                sorted_terms = sorted(sparse_vector.items(), key=lambda x: x[1], reverse=True)
                query_text = " ".join([term for term, _ in sorted_terms[:10]])  # 取前10個詞
            
            if not query_text:
                return None
            
            table = self._get_table(collection_name)
            
            # 使用關鍵字進行全文搜尋
            # 這裡實作一個簡單的關鍵字匹配邏輯
            keywords = self._extract_keywords(query_text)
            
            if not keywords:
                return VectorSearchResult(
                    ids=[], distances=[], similarities=[], metadata=[],
                    embeddings=[] if include_embeddings else None,
                    search_type=SearchType.SPARSE
                )
            
            # 構建搜尋條件
            search_conditions = []
            for keyword in keywords:
                # 在元資料中搜尋關鍵字（只使用確實存在的欄位）
                for field in ['content', 'title']:
                    search_conditions.append(f"{field} LIKE '%{keyword}%'")
            
            if search_conditions:
                where_clause = " OR ".join(search_conditions)
                
                # 應用搜尋過濾器
                if search_filter and search_filter.conditions:
                    filter_conditions = []
                    for field, value in search_filter.conditions.items():
                        if isinstance(value, str):
                            filter_conditions.append(f"{field} = '{value}'")
                        else:
                            filter_conditions.append(f"{field} = {value}")
                    
                    if filter_conditions:
                        where_clause = f"({where_clause}) AND ({' AND '.join(filter_conditions)})"
                
                try:
                    # 執行搜尋
                    results = table.search().where(where_clause).limit(k).to_pandas()
                    
                    if len(results) == 0:
                        return VectorSearchResult(
                            ids=[], distances=[], similarities=[], metadata=[],
                            embeddings=[] if include_embeddings else None,
                            search_type=SearchType.SPARSE
                        )
                    
                    # 計算關鍵字匹配分數
                    similarities = []
                    for _, row in results.iterrows():
                        score = self._calculate_keyword_score(keywords, row, sparse_vector)
                        similarities.append(score)
                    
                    # 根據相似度排序
                    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
                    
                    # 提取結果
                    ids = [results.iloc[i]['id'] for i in sorted_indices]
                    distances = [1.0 - similarities[i] for i in sorted_indices]  # 轉換為距離
                    sorted_similarities = [similarities[i] for i in sorted_indices]
                    
                    # 提取元資料
                    metadata = []
                    for i in sorted_indices:
                        row = results.iloc[i]
                        meta = {}
                        for col in results.columns:
                            if col not in ['id', 'vector', 'created_at', 'updated_at']:
                                meta[col] = row[col]
                        metadata.append(meta)
                    
                    # 提取向量（如果需要）
                    embeddings = None
                    if include_embeddings:
                        embeddings = [np.array(results.iloc[i]['vector']) for i in sorted_indices]
                    
                    return VectorSearchResult(
                        ids=ids,
                        distances=distances,
                        similarities=sorted_similarities,
                        metadata=metadata,
                        embeddings=embeddings,
                        search_type=SearchType.SPARSE
                    )
                    
                except Exception as e:
                    logger.warning(f"關鍵字搜尋失敗，返回空結果: {e}")
                    return VectorSearchResult(
                        ids=[], distances=[], similarities=[], metadata=[],
                        embeddings=[] if include_embeddings else None,
                        search_type=SearchType.SPARSE
                    )
            
            return VectorSearchResult(
                ids=[], distances=[], similarities=[], metadata=[],
                embeddings=[] if include_embeddings else None,
                search_type=SearchType.SPARSE
            )
            
        except Exception as e:
            logger.error(f"稀疏搜尋失敗: {e}")
            return VectorSearchResult(
                ids=[], distances=[], similarities=[], metadata=[],
                embeddings=[] if include_embeddings else None,
                search_type=SearchType.SPARSE
            )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """從文本中提取關鍵字"""
        # 簡單的關鍵字提取邏輯
        # 移除標點符號並分割單詞
        import string
        
        # 移除標點符號
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 分割並過濾短詞
        words = [word.strip().lower() for word in text.split() if len(word.strip()) > 2]
        
        # 移除常見停用詞（簡化版）
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = [word for word in words if word not in stop_words]
        
        return keywords[:10]  # 限制關鍵字數量
    
    def _calculate_keyword_score(
        self,
        keywords: List[str],
        row: Any,
        sparse_vector: Optional[Dict[str, float]] = None
    ) -> float:
        """計算關鍵字匹配分數"""
        score = 0.0
        total_weight = 0.0
        
        # 檢查各個文本欄位
        text_fields = ['content', 'title']
        
        for keyword in keywords:
            keyword_weight = sparse_vector.get(keyword, 1.0) if sparse_vector else 1.0
            total_weight += keyword_weight
            
            for field in text_fields:
                if field in row and row[field] is not None:
                    field_text = str(row[field]).lower()
                    if keyword in field_text:
                        # 計算詞頻
                        count = field_text.count(keyword)
                        field_score = count * keyword_weight
                        
                        # 根據欄位重要性調整分數
                        if field == 'title':
                            field_score *= 2.0
                        elif field == 'description':
                            field_score *= 1.5
                        
                        score += field_score
                        break  # 找到匹配就跳出欄位循環
        
        # 正規化分數
        if total_weight > 0:
            score = score / total_weight
        
        # 限制分數範圍在 0-1 之間
        return min(1.0, score)
    
    async def batch_search_vectors(
        self,
        collection_name: str,
        query_vectors: Union[List[np.ndarray], np.ndarray],
        k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[VectorSearchResult]:
        """批次向量相似性搜尋"""
        if isinstance(query_vectors, np.ndarray):
            if query_vectors.ndim == 1:
                query_vectors = [query_vectors]
            else:
                query_vectors = [query_vectors[i] for i in range(query_vectors.shape[0])]
        
        results = []
        for query_vector in query_vectors:
            result = await self.search_vectors(
                collection_name,
                query_vector,
                k,
                filter_conditions,
                include_embeddings
            )
            results.append(result)
        
        return results
    
    async def get_vector_by_id(
        self,
        collection_name: str,
        vector_id: str,
        include_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        """根據 ID 取得向量資料"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            if not await self.collection_exists(collection_name):
                raise CollectionError(f"集合 {collection_name} 不存在")
            
            table = self._get_table(collection_name)
            
            # 查詢資料 - 使用 filter 而不是 search().where()
            try:
                results = table.to_pandas().query(f"id == '{vector_id}'")
            except Exception:
                # 如果 query 失敗，嘗試使用 search 方法
                dummy_vector = np.zeros(1, dtype=np.float32)  # 創建一個假的查詢向量
                results = table.search(dummy_vector).where(f"id = '{vector_id}'").limit(1).to_pandas()
            
            if len(results) == 0:
                return None
            
            row = results.iloc[0]
            data = {
                "id": row['id'],
                "created_at": row.get('created_at'),
                "updated_at": row.get('updated_at')
            }
            
            # 添加元資料
            for col in results.columns:
                if col not in ['id', 'vector', 'created_at', 'updated_at']:
                    data[col] = row[col]
            
            # 添加向量（如果需要）
            if include_embedding:
                data['vector'] = np.array(row['vector'])
            
            return data
            
        except Exception as e:
            logger.error(f"從集合 {collection_name} 取得向量 {vector_id} 失敗: {e}")
            return None
    
    async def optimize_index(
        self,
        collection_name: str,
        optimization_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """優化 LanceDB 向量索引"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            if not await self.collection_exists(collection_name):
                raise CollectionError(f"集合 {collection_name} 不存在")
            
            table = self._get_table(collection_name)
            
            # 設定預設優化參數
            default_params = {
                "index_type": "IVF_PQ",  # 預設使用 IVF_PQ 索引
                "num_partitions": 256,   # 分區數量
                "num_sub_vectors": 16,   # 子向量數量
                "accelerator": "cuda" if self._has_gpu() else "cpu"
            }
            
            if optimization_params:
                default_params.update(optimization_params)
            
            logger.info(f"開始優化集合 {collection_name} 的索引，參數: {default_params}")
            
            # 建立向量索引
            try:
                # LanceDB 的索引建立方法（根據版本調整 API）
                try:
                    # 嘗試新版本 API
                    table.create_index(
                        "vector",
                        index_type=default_params["index_type"],
                        num_partitions=default_params["num_partitions"],
                        num_sub_vectors=default_params["num_sub_vectors"]
                    )
                except TypeError:
                    # 嘗試舊版本 API
                    table.create_index(
                        "vector",
                        config=lancedb.index.IvfPq(
                            num_partitions=default_params["num_partitions"],
                            num_sub_vectors=default_params["num_sub_vectors"]
                        )
                    )
                
                logger.info(f"成功優化集合 {collection_name} 的索引")
                return True
                
            except Exception as index_error:
                # 如果索引建立失敗，記錄警告但不視為錯誤
                logger.warning(f"索引優化失敗，但不影響基本功能: {index_error}")
                return True  # 返回 True 因為基本功能仍然可用
            
        except Exception as e:
            logger.error(f"優化集合 {collection_name} 索引失敗: {e}")
            return False
    
    async def get_index_stats(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """取得 LanceDB 索引統計資訊"""
        if not self.is_connected:
            raise ConnectionError("未建立連線")
        
        try:
            if not await self.collection_exists(collection_name):
                return {"error": f"集合 {collection_name} 不存在"}
            
            table = self._get_table(collection_name)
            collection_info = await self.get_collection_info(collection_name)
            
            # 取得表格統計資訊
            stats = {
                "collection_name": collection_name,
                "vector_count": collection_info.count if collection_info else 0,
                "dimension": collection_info.dimension if collection_info else 0,
                "index_type": "unknown",
                "optimization_status": "unknown",
                "storage_size_mb": 0,
                "index_build_time": None
            }
            
            try:
                # 嘗試取得更詳細的統計資訊
                # 注意：這些方法可能因 LanceDB 版本而異
                table_stats = table.stats()
                if table_stats:
                    stats.update({
                        "storage_size_mb": table_stats.get("size_bytes", 0) / (1024 * 1024),
                        "index_type": table_stats.get("index_type", "unknown"),
                        "optimization_status": "optimized" if table_stats.get("has_index", False) else "not_optimized"
                    })
                
            except Exception as stats_error:
                logger.debug(f"無法取得詳細統計資訊: {stats_error}")
            
            return stats
            
        except Exception as e:
            logger.error(f"取得集合 {collection_name} 索引統計失敗: {e}")
            return {"error": str(e)}
    
    def _has_gpu(self) -> bool:
        """檢查是否有可用的 GPU"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import cupy
                return True
            except ImportError:
                return False
    
    def _get_table(self, name: str) -> Table:
        """取得表格物件（使用快取）"""
        if name not in self._tables_cache:
            self._tables_cache[name] = self.db.open_table(name)
        return self._tables_cache[name]