"""
統一查詢引擎

整合全域和本地搜尋功能的統一查詢介面，提供完整的中文 GraphRAG 查詢功能，
支援 LLM 選擇和切換邏輯，結構化回答生成機制。
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from loguru import logger

from chinese_graphrag.models import (
    Community, Entity, Relationship, TextUnit, QueryResult
)
from chinese_graphrag.config import GraphRAGConfig
from chinese_graphrag.vector_stores import VectorStoreManager
from chinese_graphrag.indexing.engine import GraphRAGIndexer

from .manager import LLMManager, LLMConfig, TaskType
from .processor import ChineseQueryProcessor, QueryAnalysis, QueryType, QueryIntent
from .global_search import GlobalSearchEngine, GlobalSearchResult
from .local_search import LocalSearchEngine, LocalSearchResult


@dataclass
class QueryEngineConfig:
    """查詢引擎配置"""
    # LLM 配置
    llm_configs: List[LLMConfig]
    
    # 搜尋配置
    default_global_strategy: str = "community_based"
    default_local_strategy: str = "entity_focused"
    
    # 結果配置
    max_global_communities: int = 5
    max_local_entities: int = 10
    max_text_units: int = 20
    
    # 自動選擇閾值
    auto_selection_threshold: float = 0.7
    hybrid_search_threshold: float = 0.5
    
    # 效能配置
    query_timeout: float = 60.0
    enable_caching: bool = True
    cache_ttl: int = 3600  # 快取存活時間（秒）


@dataclass
class UnifiedQueryResult:
    """統一查詢結果"""
    query: str
    answer: str
    confidence: float
    search_type: str  # "global", "local", "hybrid"
    global_result: Optional[GlobalSearchResult] = None
    local_result: Optional[LocalSearchResult] = None
    analysis: Optional[QueryAnalysis] = None
    sources: List[str] = None
    reasoning_path: List[str] = None
    search_time: float = 0.0
    llm_model_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        result = {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "search_type": self.search_type,
            "sources": self.sources or [],
            "reasoning_path": self.reasoning_path or [],
            "search_time": self.search_time,
            "llm_model_used": self.llm_model_used
        }
        
        if self.analysis:
            result["analysis"] = self.analysis.to_dict()
        
        if self.global_result:
            result["global_details"] = self.global_result.to_dict()
        
        if self.local_result:
            result["local_details"] = self.local_result.to_dict()
        
        return result
    
    def to_query_result(self) -> QueryResult:
        """轉換為系統 QueryResult 模型"""
        return QueryResult(
            query=self.query,
            answer=self.answer,
            metadata={
                "confidence": self.confidence,
                "search_type": self.search_type,
                "sources": self.sources or [],
                "reasoning_path": self.reasoning_path or [],
                "search_time": self.search_time,
                "llm_model_used": self.llm_model_used
            }
        )


class QueryCache:
    """查詢快取管理器"""
    
    def __init__(self, enable_cache: bool = True, ttl: int = 3600):
        """
        初始化查詢快取
        
        Args:
            enable_cache: 是否啟用快取
            ttl: 快取存活時間（秒）
        """
        self.enable_cache = enable_cache
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"查詢快取初始化，啟用: {enable_cache}, TTL: {ttl}s")
    
    def _generate_cache_key(self, query: str, analysis: QueryAnalysis) -> str:
        """生成快取鍵"""
        # 基於查詢內容和分析結果生成唯一鍵
        key_components = [
            query.lower().strip(),
            analysis.query_type.value,
            analysis.intent.value,
            "_".join(sorted(analysis.entities)),
            "_".join(sorted(analysis.keywords))
        ]
        return "|".join(key_components)
    
    def get(self, query: str, analysis: QueryAnalysis) -> Optional[UnifiedQueryResult]:
        """從快取獲取結果"""
        if not self.enable_cache:
            return None
        
        cache_key = self._generate_cache_key(query, analysis)
        
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            
            # 檢查是否過期
            import time
            if time.time() - cache_entry["timestamp"] < self.ttl:
                logger.debug(f"快取命中: {cache_key}")
                return cache_entry["result"]
            else:
                # 移除過期條目
                del self.cache[cache_key]
                logger.debug(f"快取過期，已移除: {cache_key}")
        
        return None
    
    def set(self, query: str, analysis: QueryAnalysis, result: UnifiedQueryResult):
        """將結果存入快取"""
        if not self.enable_cache:
            return
        
        cache_key = self._generate_cache_key(query, analysis)
        
        import time
        self.cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        logger.debug(f"結果已快取: {cache_key}")
    
    def clear(self):
        """清空快取"""
        self.cache.clear()
        logger.info("查詢快取已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取快取統計資訊"""
        import time
        current_time = time.time()
        
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values()
            if current_time - entry["timestamp"] >= self.ttl
        )
        
        return {
            "enabled": self.enable_cache,
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "ttl": self.ttl
        }


class SearchTypeSelector:
    """搜尋類型選擇器"""
    
    def __init__(self, config: QueryEngineConfig):
        self.config = config
    
    def select_search_type(self, analysis: QueryAnalysis) -> str:
        """
        選擇搜尋類型
        
        Args:
            analysis: 查詢分析結果
            
        Returns:
            搜尋類型: "global", "local", "hybrid"
        """
        # 1. 基於查詢類型的基本判斷
        if analysis.query_type in [
            QueryType.GLOBAL_SEARCH, 
            QueryType.SUMMARY, 
            QueryType.COMPARISON,
            QueryType.CAUSAL,
            QueryType.TEMPORAL
        ]:
            base_type = "global"
        elif analysis.query_type in [
            QueryType.LOCAL_SEARCH,
            QueryType.ENTITY_SEARCH,
            QueryType.RELATION_SEARCH
        ]:
            base_type = "local"
        else:
            base_type = "hybrid"
        
        # 2. 基於信心度調整
        if analysis.confidence < self.config.hybrid_search_threshold:
            return "hybrid"  # 信心度低時使用混合搜尋
        
        # 3. 基於實體數量調整
        entity_count = len(analysis.entities)
        if entity_count == 0:
            return "global"  # 沒有明確實體時使用全域搜尋
        elif entity_count == 1:
            return "local"   # 單一實體時使用本地搜尋
        elif entity_count >= 4:
            return "global"  # 多實體時使用全域搜尋
        
        # 4. 基於查詢意圖調整
        if analysis.intent in [QueryIntent.EXPLANATION, QueryIntent.COMPARISON]:
            return "global"
        elif analysis.intent in [QueryIntent.FACT_CHECKING]:
            return "local"
        
        return base_type
    
    def should_use_hybrid(self, analysis: QueryAnalysis) -> bool:
        """判斷是否應該使用混合搜尋"""
        # 信心度較低時
        if analysis.confidence < self.config.hybrid_search_threshold:
            return True
        
        # 複雜查詢（多實體 + 特定意圖）
        if (len(analysis.entities) >= 2 and 
            analysis.intent in [QueryIntent.COMPARISON, QueryIntent.PROBLEM_SOLVING]):
            return True
        
        return False


class ResultSynthesizer:
    """結果綜合器"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
    
    async def synthesize_hybrid_result(
        self,
        query: str,
        analysis: QueryAnalysis,
        global_result: Optional[GlobalSearchResult],
        local_result: Optional[LocalSearchResult]
    ) -> UnifiedQueryResult:
        """
        綜合混合搜尋結果
        
        Args:
            query: 原始查詢
            analysis: 查詢分析結果
            global_result: 全域搜尋結果
            local_result: 本地搜尋結果
            
        Returns:
            統一查詢結果
        """
        logger.info("開始綜合混合搜尋結果")
        
        # 如果只有一個結果，直接使用
        if global_result and not local_result:
            return self._convert_global_result(query, analysis, global_result)
        elif local_result and not global_result:
            return self._convert_local_result(query, analysis, local_result)
        elif not global_result and not local_result:
            return self._create_fallback_result(query, analysis)
        
        # 綜合兩個結果
        try:
            synthesized_answer = await self._synthesize_answers(
                query, analysis, global_result, local_result
            )
            
            # 合併來源和推理路徑
            combined_sources = []
            combined_reasoning = []
            
            if global_result:
                combined_sources.extend(global_result.sources)
                combined_reasoning.extend([f"全域: {step}" for step in global_result.reasoning_path])
            
            if local_result:
                combined_sources.extend(local_result.sources)
                combined_reasoning.extend([f"本地: {step}" for step in local_result.reasoning_path])
            
            combined_reasoning.append("綜合全域和本地搜尋結果")
            
            # 計算綜合信心度
            combined_confidence = self._calculate_combined_confidence(
                global_result, local_result
            )
            
            # 計算總搜尋時間
            total_search_time = 0.0
            if global_result:
                total_search_time += global_result.search_time
            if local_result:
                total_search_time += local_result.search_time
            
            return UnifiedQueryResult(
                query=query,
                answer=synthesized_answer,
                confidence=combined_confidence,
                search_type="hybrid",
                global_result=global_result,
                local_result=local_result,
                analysis=analysis,
                sources=combined_sources[:20],  # 限制來源數量
                reasoning_path=combined_reasoning,
                search_time=total_search_time,
                llm_model_used="hybrid_synthesis"
            )
            
        except Exception as e:
            logger.error(f"綜合結果時發生錯誤: {e}")
            
            # 降級策略：選擇信心度較高的結果
            if global_result and local_result:
                if global_result.confidence >= local_result.confidence:
                    return self._convert_global_result(query, analysis, global_result)
                else:
                    return self._convert_local_result(query, analysis, local_result)
            
            return self._create_fallback_result(query, analysis)
    
    async def _synthesize_answers(
        self,
        query: str,
        analysis: QueryAnalysis,
        global_result: GlobalSearchResult,
        local_result: LocalSearchResult
    ) -> str:
        """綜合答案"""
        
        synthesis_prompt = f"""# 指令：綜合多重搜尋結果

## 任務
基於全域搜尋和本地搜尋的結果，為使用者查詢提供綜合、完整的中文回答。

## 使用者查詢
{query}

## 全域搜尋結果
{global_result.answer}

## 本地搜尋結果  
{local_result.answer}

## 要求
1. **整合性**：將兩個搜尋結果的優點結合
2. **一致性**：確保回答內容邏輯一致，避免矛盾
3. **完整性**：提供全面的回答，涵蓋兩個結果的重要資訊
4. **簡潔性**：避免重複內容，保持回答簡潔明瞭
5. **結構化**：使用清晰的段落組織回答

## 綜合策略
- 以全域搜尋提供整體框架和背景
- 以本地搜尋提供具體細節和證據
- 突出兩個結果的互補性
- 如有衝突，說明不同視角

請提供綜合回答："""

        try:
            synthesized_answer = await self.llm_manager.generate(
                synthesis_prompt,
                TaskType.GENERAL_QA,
                max_tokens=2000,
                temperature=0.6
            )
            
            return synthesized_answer.strip()
            
        except Exception as e:
            logger.error(f"綜合回答生成失敗: {e}")
            
            # 降級策略：簡單連接兩個結果
            return f"{global_result.answer}\n\n{local_result.answer}"
    
    def _convert_global_result(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        global_result: GlobalSearchResult
    ) -> UnifiedQueryResult:
        """轉換全域搜尋結果"""
        return UnifiedQueryResult(
            query=query,
            answer=global_result.answer,
            confidence=global_result.confidence,
            search_type="global",
            global_result=global_result,
            analysis=analysis,
            sources=global_result.sources,
            reasoning_path=global_result.reasoning_path,
            search_time=global_result.search_time,
            llm_model_used=global_result.llm_model_used
        )
    
    def _convert_local_result(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        local_result: LocalSearchResult
    ) -> UnifiedQueryResult:
        """轉換本地搜尋結果"""
        return UnifiedQueryResult(
            query=query,
            answer=local_result.answer,
            confidence=local_result.confidence,
            search_type="local",
            local_result=local_result,
            analysis=analysis,
            sources=local_result.sources,
            reasoning_path=local_result.reasoning_path,
            search_time=local_result.search_time,
            llm_model_used=local_result.llm_model_used
        )
    
    def _create_fallback_result(
        self, 
        query: str, 
        analysis: QueryAnalysis
    ) -> UnifiedQueryResult:
        """建立降級結果"""
        return UnifiedQueryResult(
            query=query,
            answer="抱歉，目前無法處理您的查詢。請嘗試重新表述問題或聯繫系統管理員。",
            confidence=0.1,
            search_type="fallback",
            analysis=analysis,
            sources=[],
            reasoning_path=["搜尋失敗，使用降級策略"],
            search_time=0.0,
            llm_model_used="fallback"
        )
    
    def _calculate_combined_confidence(
        self,
        global_result: Optional[GlobalSearchResult],
        local_result: Optional[LocalSearchResult]
    ) -> float:
        """計算綜合信心度"""
        if global_result and local_result:
            # 加權平均，全域搜尋權重稍高
            return (global_result.confidence * 0.6 + local_result.confidence * 0.4)
        elif global_result:
            return global_result.confidence
        elif local_result:
            return local_result.confidence
        else:
            return 0.1


class QueryEngine:
    """統一查詢引擎
    
    整合全域和本地搜尋功能的統一查詢介面，提供完整的 GraphRAG 查詢功能。
    """
    
    def __init__(
        self,
        config: QueryEngineConfig,
        graphrag_config: GraphRAGConfig,
        indexer: GraphRAGIndexer,
        vector_store: VectorStoreManager
    ):
        """
        初始化查詢引擎
        
        Args:
            config: 查詢引擎配置
            graphrag_config: GraphRAG 系統配置
            indexer: GraphRAG 索引器
            vector_store: 向量存儲管理器
        """
        self.config = config
        self.graphrag_config = graphrag_config
        self.indexer = indexer
        self.vector_store = vector_store
        
        # 初始化核心元件
        self.llm_manager = LLMManager(config.llm_configs)
        self.query_processor = ChineseQueryProcessor()
        self.global_search_engine = GlobalSearchEngine(
            self.llm_manager, 
            vector_store,
            config.default_global_strategy
        )
        self.local_search_engine = LocalSearchEngine(
            self.llm_manager,
            vector_store, 
            config.default_local_strategy
        )
        
        # 初始化輔助元件
        self.search_selector = SearchTypeSelector(config)
        self.result_synthesizer = ResultSynthesizer(self.llm_manager)
        self.query_cache = QueryCache(config.enable_caching, config.cache_ttl)
        
        logger.info("統一查詢引擎初始化完成")
    
    async def query(
        self,
        query: str,
        search_type: Optional[str] = None,
        enable_cache: bool = True,
        **kwargs
    ) -> UnifiedQueryResult:
        """
        執行查詢
        
        Args:
            query: 查詢字串
            search_type: 強制指定搜尋類型 ("global", "local", "hybrid")
            enable_cache: 是否啟用快取
            **kwargs: 其他參數
            
        Returns:
            統一查詢結果
        """
        import time
        start_time = time.time()
        
        logger.info(f"開始處理查詢: {query}")
        
        try:
            # 1. 查詢預處理和分析
            analysis = self.query_processor.process_query(query)
            logger.debug(f"查詢分析完成: {analysis.query_type.value}")
            
            # 2. 檢查快取
            if enable_cache:
                cached_result = self.query_cache.get(query, analysis)
                if cached_result:
                    logger.info("返回快取結果")
                    return cached_result
            
            # 3. 選擇搜尋類型
            selected_search_type = search_type or self.search_selector.select_search_type(analysis)
            logger.info(f"選擇搜尋類型: {selected_search_type}")
            
            # 4. 執行搜尋
            result = await self._execute_search(query, analysis, selected_search_type, **kwargs)
            
            # 5. 更新搜尋時間
            total_time = time.time() - start_time
            result.search_time = total_time
            
            # 6. 快取結果
            if enable_cache and result.confidence > 0.3:
                self.query_cache.set(query, analysis, result)
            
            logger.info(f"查詢完成，耗時 {total_time:.2f} 秒，信心度 {result.confidence:.2f}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"查詢超時: {query}")
            return UnifiedQueryResult(
                query=query,
                answer="查詢處理超時，請稍後重試或簡化您的問題。",
                confidence=0.0,
                search_type="timeout",
                search_time=time.time() - start_time,
                llm_model_used="none"
            )
        except Exception as e:
            logger.error(f"查詢處理失敗: {e}")
            return UnifiedQueryResult(
                query=query,
                answer=f"查詢處理時發生錯誤：{str(e)}。請檢查您的問題或聯繫系統管理員。",
                confidence=0.0,
                search_type="error",
                search_time=time.time() - start_time,
                llm_model_used="none"
            )
    
    async def _execute_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        search_type: str,
        **kwargs
    ) -> UnifiedQueryResult:
        """執行具體的搜尋"""
        
        if search_type == "global":
            return await self._execute_global_search(query, analysis, **kwargs)
        elif search_type == "local":
            return await self._execute_local_search(query, analysis, **kwargs)
        elif search_type == "hybrid":
            return await self._execute_hybrid_search(query, analysis, **kwargs)
        else:
            raise ValueError(f"不支援的搜尋類型: {search_type}")
    
    async def _execute_global_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        **kwargs
    ) -> UnifiedQueryResult:
        """執行全域搜尋"""
        
        # 獲取所需資料
        communities = list(self.indexer.communities.values())
        entities = list(self.indexer.entities.values())
        relationships = list(self.indexer.relationships.values())
        
        # 執行全域搜尋
        global_result = await self.global_search_engine.search(
            query=query,
            analysis=analysis,
            communities=communities,
            entities=entities,
            relationships=relationships,
            max_communities=kwargs.get("max_communities", self.config.max_global_communities),
            **kwargs
        )
        
        return self.result_synthesizer._convert_global_result(query, analysis, global_result)
    
    async def _execute_local_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        **kwargs
    ) -> UnifiedQueryResult:
        """執行本地搜尋"""
        
        # 獲取所需資料
        entities = list(self.indexer.entities.values())
        relationships = list(self.indexer.relationships.values())
        text_units = list(self.indexer.text_units.values())
        
        # 執行本地搜尋
        local_result = await self.local_search_engine.search(
            query=query,
            analysis=analysis,
            entities=entities,
            relationships=relationships,
            text_units=text_units,
            max_entities=kwargs.get("max_entities", self.config.max_local_entities),
            max_text_units=kwargs.get("max_text_units", self.config.max_text_units),
            **kwargs
        )
        
        return self.result_synthesizer._convert_local_result(query, analysis, local_result)
    
    async def _execute_hybrid_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        **kwargs
    ) -> UnifiedQueryResult:
        """執行混合搜尋"""
        
        # 並行執行全域和本地搜尋
        global_task = self._execute_global_search(query, analysis, **kwargs)
        local_task = self._execute_local_search(query, analysis, **kwargs)
        
        try:
            # 等待兩個搜尋完成
            results = await asyncio.gather(global_task, local_task, return_exceptions=True)
            
            global_unified = results[0] if not isinstance(results[0], Exception) else None
            local_unified = results[1] if not isinstance(results[1], Exception) else None
            
            # 提取原始搜尋結果
            global_result = global_unified.global_result if global_unified else None
            local_result = local_unified.local_result if local_unified else None
            
            # 綜合結果
            return await self.result_synthesizer.synthesize_hybrid_result(
                query, analysis, global_result, local_result
            )
            
        except Exception as e:
            logger.error(f"混合搜尋失敗: {e}")
            return self.result_synthesizer._create_fallback_result(query, analysis)
    
    async def batch_query(
        self,
        queries: List[str],
        max_concurrent: int = 3,
        **kwargs
    ) -> List[UnifiedQueryResult]:
        """
        批次查詢
        
        Args:
            queries: 查詢列表
            max_concurrent: 最大並發數
            **kwargs: 其他參數
            
        Returns:
            查詢結果列表
        """
        logger.info(f"開始批次查詢，共 {len(queries)} 個查詢")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def query_with_semaphore(query: str) -> UnifiedQueryResult:
            async with semaphore:
                return await self.query(query, **kwargs)
        
        # 並行執行查詢
        tasks = [query_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 處理異常結果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"查詢 {i} 失敗: {result}")
                processed_results.append(UnifiedQueryResult(
                    query=queries[i],
                    answer=f"查詢失敗: {str(result)}",
                    confidence=0.0,
                    search_type="error"
                ))
            else:
                processed_results.append(result)
        
        logger.info(f"批次查詢完成，成功 {sum(1 for r in processed_results if r.confidence > 0)} 個")
        return processed_results
    
    def get_engine_status(self) -> Dict[str, Any]:
        """獲取引擎狀態資訊"""
        return {
            "config": {
                "default_global_strategy": self.config.default_global_strategy,
                "default_local_strategy": self.config.default_local_strategy,
                "max_global_communities": self.config.max_global_communities,
                "max_local_entities": self.config.max_local_entities,
                "query_timeout": self.config.query_timeout
            },
            "cache": self.query_cache.get_cache_stats(),
            "data_status": {
                "entities_count": len(self.indexer.entities),
                "relationships_count": len(self.indexer.relationships),
                "communities_count": len(self.indexer.communities),
                "text_units_count": len(self.indexer.text_units)
            },
            "llm_status": self.llm_manager.get_adapter_info()
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """執行健康檢查"""
        health_status = {
            "query_processor": True,
            "llm_manager": False,
            "global_search": False,
            "local_search": False,
            "vector_store": False,
            "indexer": False
        }
        
        try:
            # 檢查 LLM 管理器
            llm_health = await self.llm_manager.health_check_all()
            health_status["llm_manager"] = any(llm_health.values())
            
            # 檢查搜尋引擎
            global_health = await self.global_search_engine.health_check()
            health_status["global_search"] = all(global_health.values())
            
            local_health = await self.local_search_engine.health_check()
            health_status["local_search"] = all(local_health.values())
            
            # 檢查索引器（檢查是否有資料）
            health_status["indexer"] = (
                len(self.indexer.entities) > 0 or 
                len(self.indexer.communities) > 0
            )
            
            # 檢查向量存儲
            health_status["vector_store"] = True  # 暫時假設健康
            
        except Exception as e:
            logger.error(f"健康檢查失敗: {e}")
        
        return health_status
    
    def clear_cache(self):
        """清空查詢快取"""
        self.query_cache.clear()
        logger.info("查詢快取已清空")