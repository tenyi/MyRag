"""
全域搜尋引擎

基於整體知識圖譜的高層次問答引擎，整合 GraphRAG 全域搜尋功能，
提供基於社群的高層次問答和不同 LLM 的全域搜尋策略。
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from chinese_graphrag.models import Community, Entity, Relationship, QueryResult
from chinese_graphrag.vector_stores import VectorStoreManager
from .manager import LLMManager, TaskType
from .processor import QueryAnalysis, QueryType


@dataclass
class GlobalSearchContext:
    """全域搜尋上下文"""
    query: str
    analysis: QueryAnalysis
    communities: List[Community]
    relevant_entities: List[Entity]
    relevant_relationships: List[Relationship]
    search_level: int = 1  # 搜尋深度層級
    max_communities: int = 5  # 最大社群數量
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "query": self.query,
            "analysis": self.analysis.to_dict(),
            "communities_count": len(self.communities),
            "entities_count": len(self.relevant_entities),
            "relationships_count": len(self.relevant_relationships),
            "search_level": self.search_level,
            "max_communities": self.max_communities
        }


@dataclass
class GlobalSearchResult:
    """全域搜尋結果"""
    query: str
    answer: str
    confidence: float
    sources: List[str]
    communities_used: List[str]
    entities_used: List[str]
    reasoning_path: List[str]
    search_time: float
    llm_model_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "communities_used": self.communities_used,
            "entities_used": self.entities_used,
            "reasoning_path": self.reasoning_path,
            "search_time": self.search_time,
            "llm_model_used": self.llm_model_used
        }


class CommunityRanker:
    """社群排序器"""
    
    def rank_communities_for_query(
        self, 
        query: str,
        analysis: QueryAnalysis,
        communities: List[Community],
        vector_store: VectorStoreManager
    ) -> List[Tuple[Community, float]]:
        """
        為查詢排序相關社群
        
        Args:
            query: 查詢字串
            analysis: 查詢分析結果
            communities: 候選社群列表
            vector_store: 向量存儲管理器
            
        Returns:
            排序後的 (社群, 相關性分數) 列表
        """
        ranked_communities = []
        
        for community in communities:
            score = self._calculate_community_relevance(
                query, analysis, community, vector_store
            )
            ranked_communities.append((community, score))
        
        # 按分數降序排序
        ranked_communities.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_communities
    
    def _calculate_community_relevance(
        self,
        query: str,
        analysis: QueryAnalysis,
        community: Community,
        vector_store: VectorStoreManager
    ) -> float:
        """計算社群相關性分數"""
        score = 0.0
        
        # 1. 基於社群摘要的文本相似性
        if community.summary:
            # 這裡應該使用向量相似性計算，暫時使用關鍵詞匹配
            summary_lower = community.summary.lower()
            for keyword in analysis.keywords:
                if keyword.lower() in summary_lower:
                    score += 0.3
        
        # 2. 基於實體重疊度
        query_entities = set(analysis.entities)
        if query_entities and community.entities:
            # 需要通過實體ID獲取實體名稱，這裡簡化處理
            entity_overlap = len(query_entities.intersection(set(community.entities)))
            if entity_overlap > 0:
                score += entity_overlap * 0.4
        
        # 3. 基於社群層級（較高層級的社群通常包含更廣泛的資訊）
        if analysis.query_type in [QueryType.GLOBAL_SEARCH, QueryType.SUMMARY]:
            score += community.level * 0.1
        
        # 4. 基於社群大小（較大的社群可能包含更多相關資訊）
        if community.entities:
            entity_count_bonus = min(len(community.entities) / 10.0, 0.2)
            score += entity_count_bonus
        
        return min(score, 1.0)


class GlobalSearchStrategy(ABC):
    """全域搜尋策略抽象基類"""
    
    @abstractmethod
    async def search(
        self,
        context: GlobalSearchContext,
        llm_manager: LLMManager,
        vector_store: VectorStoreManager
    ) -> GlobalSearchResult:
        """執行全域搜尋"""
        pass


class CommunityBasedStrategy(GlobalSearchStrategy):
    """基於社群的搜尋策略"""
    
    async def search(
        self,
        context: GlobalSearchContext,
        llm_manager: LLMManager,
        vector_store: VectorStoreManager
    ) -> GlobalSearchResult:
        """基於社群執行全域搜尋"""
        import time
        start_time = time.time()
        
        logger.info(f"開始基於社群的全域搜尋: {context.query}")
        
        # 1. 選擇最相關的社群
        ranker = CommunityRanker()
        ranked_communities = ranker.rank_communities_for_query(
            context.query, context.analysis, context.communities, vector_store
        )
        
        # 選擇前 N 個社群
        top_communities = ranked_communities[:context.max_communities]
        selected_communities = [comm for comm, score in top_communities]
        
        logger.debug(f"選擇了 {len(selected_communities)} 個相關社群")
        
        # 2. 構建搜尋上下文
        search_context = self._build_search_context(
            context.query, context.analysis, selected_communities
        )
        
        # 3. 生成回答
        answer = await self._generate_answer(
            search_context, context.analysis, llm_manager
        )
        
        # 4. 提取推理路徑和來源
        reasoning_path, sources = self._extract_reasoning_and_sources(
            search_context, selected_communities
        )
        
        search_time = time.time() - start_time
        
        result = GlobalSearchResult(
            query=context.query,
            answer=answer,
            confidence=self._calculate_confidence(context.analysis, selected_communities),
            sources=sources,
            communities_used=[comm.title for comm in selected_communities],
            entities_used=context.analysis.entities,
            reasoning_path=reasoning_path,
            search_time=search_time,
            llm_model_used="global_search_model"  # 應該從 LLM manager 獲取
        )
        
        logger.info(f"全域搜尋完成，耗時 {search_time:.2f} 秒")
        return result
    
    def _build_search_context(
        self,
        query: str,
        analysis: QueryAnalysis,
        communities: List[Community]
    ) -> str:
        """構建搜尋上下文"""
        context_parts = []
        
        # 添加查詢資訊
        context_parts.append(f"使用者查詢：{query}")
        context_parts.append(f"查詢類型：{analysis.query_type.value}")
        context_parts.append(f"查詢意圖：{analysis.intent.value}")
        
        if analysis.entities:
            context_parts.append(f"相關實體：{', '.join(analysis.entities)}")
        
        if analysis.keywords:
            context_parts.append(f"關鍵詞：{', '.join(analysis.keywords)}")
        
        context_parts.append("\n=== 相關社群資訊 ===")
        
        # 添加社群資訊
        for i, community in enumerate(communities, 1):
            context_parts.append(f"\n## 社群 {i}：{community.title}")
            context_parts.append(f"層級：{community.level}")
            context_parts.append(f"實體數量：{len(community.entities)}")
            
            if community.summary:
                context_parts.append(f"摘要：{community.summary}")
            
            # 添加部分實體資訊（如果有的話）
            if community.entities:
                entity_preview = community.entities[:5]  # 只顯示前5個
                context_parts.append(f"主要實體：{', '.join(entity_preview)}")
                if len(community.entities) > 5:
                    context_parts.append(f"（還有 {len(community.entities) - 5} 個實體）")
        
        return "\n".join(context_parts)
    
    async def _generate_answer(
        self,
        search_context: str,
        analysis: QueryAnalysis,
        llm_manager: LLMManager
    ) -> str:
        """生成回答"""
        
        # 構建 prompt
        prompt = self._build_answer_prompt(search_context, analysis)
        
        try:
            # 使用 LLM 生成回答
            answer = await llm_manager.generate(
                prompt,
                TaskType.GLOBAL_SEARCH,
                max_tokens=2000,
                temperature=0.7
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"生成回答時發生錯誤: {e}")
            return f"抱歉，在處理您的查詢時遇到了問題：{str(e)}"
    
    def _build_answer_prompt(self, search_context: str, analysis: QueryAnalysis) -> str:
        """構建回答 prompt"""
        
        prompt_template = """# 指令：基於知識圖譜的全域問答

## 角色
您是一個專業的中文知識問答助手，擅長基於知識圖譜資訊進行綜合分析和推理。

## 任務
基於以下提供的知識圖譜社群資訊，回答使用者的查詢。請提供準確、全面且結構化的中文回答。

## 要求
1. **準確性**：確保回答基於提供的資訊，不要添加未提及的內容
2. **完整性**：盡可能全面地回答問題，涵蓋相關的各個方面
3. **結構化**：使用清晰的段落和要點組織回答
4. **中文表達**：使用自然流暢的繁體中文
5. **引用來源**：在適當的地方提及相關的社群或實體

## 上下文資訊
{search_context}

## 回答指導
根據查詢類型 "{query_type}" 和意圖 "{intent}"，請：

- 如果是摘要類查詢：提供概括性的總結
- 如果是比較類查詢：分析不同實體或概念的異同
- 如果是解釋類查詢：提供深入的說明和原理
- 如果是因果類查詢：分析原因和結果的關係
- 如果是時間類查詢：按時間順序組織資訊

## 回答格式
請按以下格式組織回答：

1. **直接回答**：先給出問題的核心答案
2. **詳細說明**：展開詳細的解釋和分析
3. **相關資訊**：補充相關的背景資訊
4. **結論**：總結要點

請開始回答："""

        return prompt_template.format(
            search_context=search_context,
            query_type=analysis.query_type.value,
            intent=analysis.intent.value
        )
    
    def _extract_reasoning_and_sources(
        self,
        search_context: str,
        communities: List[Community]
    ) -> Tuple[List[str], List[str]]:
        """提取推理路徑和來源"""
        
        reasoning_path = [
            "分析使用者查詢的類型和意圖",
            f"識別 {len(communities)} 個相關社群",
            "基於社群資訊構建回答上下文",
            "使用 LLM 生成綜合回答"
        ]
        
        sources = []
        for community in communities:
            sources.append(f"社群: {community.title}")
            if community.summary:
                sources.append(f"摘要: {community.summary[:100]}...")
        
        return reasoning_path, sources
    
    def _calculate_confidence(
        self,
        analysis: QueryAnalysis,
        communities: List[Community]
    ) -> float:
        """計算回答的信心度"""
        confidence = analysis.confidence * 0.4  # 基礎信心度
        
        # 基於社群品質調整信心度
        if communities:
            avg_community_quality = sum(
                1.0 if comm.summary else 0.5 for comm in communities
            ) / len(communities)
            confidence += avg_community_quality * 0.3
        
        # 基於實體匹配度調整
        if analysis.entities:
            confidence += min(len(analysis.entities) / 5.0, 0.3)
        
        return min(confidence, 1.0)


class HierarchicalStrategy(GlobalSearchStrategy):
    """階層式搜尋策略"""
    
    async def search(
        self,
        context: GlobalSearchContext,
        llm_manager: LLMManager,
        vector_store: VectorStoreManager
    ) -> GlobalSearchResult:
        """執行階層式全域搜尋"""
        import time
        start_time = time.time()
        
        logger.info(f"開始階層式全域搜尋: {context.query}")
        
        # 1. 從高層級社群開始搜尋
        level_results = []
        max_level = max(comm.level for comm in context.communities) if context.communities else 1
        
        for level in range(max_level, 0, -1):
            level_communities = [
                comm for comm in context.communities 
                if comm.level == level
            ]
            
            if level_communities:
                level_result = await self._search_at_level(
                    context, level_communities, level, llm_manager, vector_store
                )
                level_results.append(level_result)
        
        # 2. 綜合各層級結果
        final_answer = await self._synthesize_results(
            context, level_results, llm_manager
        )
        
        search_time = time.time() - start_time
        
        # 3. 構建最終結果
        all_communities = []
        all_sources = []
        reasoning_path = ["執行階層式搜尋"]
        
        for level_result in level_results:
            all_communities.extend(level_result["communities"])
            all_sources.extend(level_result["sources"])
            reasoning_path.append(f"分析第 {level_result['level']} 層級社群")
        
        reasoning_path.append("綜合各層級結果生成最終回答")
        
        result = GlobalSearchResult(
            query=context.query,
            answer=final_answer,
            confidence=self._calculate_hierarchical_confidence(level_results),
            sources=all_sources,
            communities_used=[comm.title for comm in all_communities],
            entities_used=context.analysis.entities,
            reasoning_path=reasoning_path,
            search_time=search_time,
            llm_model_used="hierarchical_search_model"
        )
        
        logger.info(f"階層式搜尋完成，耗時 {search_time:.2f} 秒")
        return result
    
    async def _search_at_level(
        self,
        context: GlobalSearchContext,
        level_communities: List[Community],
        level: int,
        llm_manager: LLMManager,
        vector_store: VectorStoreManager
    ) -> Dict[str, Any]:
        """在特定層級執行搜尋"""
        
        # 使用社群排序器選擇最相關的社群
        ranker = CommunityRanker()
        ranked_communities = ranker.rank_communities_for_query(
            context.query, context.analysis, level_communities, vector_store
        )
        
        # 選擇該層級的前幾個相關社群
        max_communities_per_level = min(3, len(ranked_communities))
        selected = [comm for comm, score in ranked_communities[:max_communities_per_level]]
        
        # 構建該層級的搜尋上下文
        level_context = f"層級 {level} 社群分析：\n"
        for comm in selected:
            level_context += f"- {comm.title}: {comm.summary or '無摘要'}\n"
        
        # 生成該層級的部分回答
        prompt = f"""基於以下第 {level} 層級的社群資訊，針對查詢「{context.query}」提供分析：

{level_context}

請提供這個層級的相關資訊和分析："""
        
        try:
            partial_answer = await llm_manager.generate(
                prompt,
                TaskType.GLOBAL_SEARCH,
                max_tokens=800,
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"層級 {level} 搜尋失敗: {e}")
            partial_answer = f"層級 {level} 分析暫時無法完成"
        
        return {
            "level": level,
            "communities": selected,
            "answer": partial_answer,
            "sources": [f"層級 {level}: {comm.title}" for comm in selected]
        }
    
    async def _synthesize_results(
        self,
        context: GlobalSearchContext,
        level_results: List[Dict[str, Any]],
        llm_manager: LLMManager
    ) -> str:
        """綜合各層級結果"""
        
        if not level_results:
            return "抱歉，沒有找到相關資訊。"
        
        # 構建綜合 prompt
        synthesis_context = f"查詢：{context.query}\n\n各層級分析結果：\n"
        
        for result in level_results:
            synthesis_context += f"\n=== 層級 {result['level']} ===\n"
            synthesis_context += result['answer']
        
        prompt = f"""請基於以下各層級的分析結果，為查詢「{context.query}」提供一個綜合、完整的回答：

{synthesis_context}

要求：
1. 整合各層級的資訊，避免重複
2. 按邏輯順序組織回答
3. 突出重點和核心內容
4. 使用清晰的中文表達

綜合回答："""
        
        try:
            final_answer = await llm_manager.generate(
                prompt,
                TaskType.GLOBAL_SEARCH,
                max_tokens=2000,
                temperature=0.6
            )
            return final_answer.strip()
        except Exception as e:
            logger.error(f"綜合結果時發生錯誤: {e}")
            # 降級策略：直接連接各層級結果
            return "\n\n".join([result['answer'] for result in level_results])
    
    def _calculate_hierarchical_confidence(self, level_results: List[Dict[str, Any]]) -> float:
        """計算階層式搜尋的信心度"""
        if not level_results:
            return 0.0
        
        # 基於層級數量和每層級的社群數量計算信心度
        base_confidence = 0.6
        level_bonus = len(level_results) * 0.1
        community_bonus = sum(len(result['communities']) for result in level_results) * 0.05
        
        return min(base_confidence + level_bonus + community_bonus, 1.0)


class GlobalSearchEngine:
    """全域搜尋引擎
    
    基於整體知識圖譜的高層次問答引擎，提供多種搜尋策略。
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        vector_store: VectorStoreManager,
        default_strategy: str = "community_based"
    ):
        """
        初始化全域搜尋引擎
        
        Args:
            llm_manager: LLM 管理器
            vector_store: 向量存儲管理器  
            default_strategy: 預設搜尋策略
        """
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.default_strategy = default_strategy
        
        # 初始化搜尋策略
        self.strategies = {
            "community_based": CommunityBasedStrategy(),
            "hierarchical": HierarchicalStrategy()
        }
        
        logger.info(f"全域搜尋引擎初始化完成，預設策略: {default_strategy}")
    
    async def search(
        self,
        query: str,
        analysis: QueryAnalysis,
        communities: List[Community],
        entities: Optional[List[Entity]] = None,
        relationships: Optional[List[Relationship]] = None,
        strategy: Optional[str] = None,
        max_communities: int = 5,
        search_level: int = 1
    ) -> GlobalSearchResult:
        """
        執行全域搜尋
        
        Args:
            query: 查詢字串
            analysis: 查詢分析結果
            communities: 可用的社群列表
            entities: 相關實體列表
            relationships: 相關關係列表
            strategy: 搜尋策略名稱
            max_communities: 最大使用社群數量
            search_level: 搜尋深度層級
            
        Returns:
            全域搜尋結果
        """
        # 選擇搜尋策略
        strategy_name = strategy or self._select_strategy(analysis)
        search_strategy = self.strategies.get(strategy_name)
        
        if not search_strategy:
            raise ValueError(f"不支援的搜尋策略: {strategy_name}")
        
        logger.info(f"使用 {strategy_name} 策略執行全域搜尋")
        
        # 構建搜尋上下文
        context = GlobalSearchContext(
            query=query,
            analysis=analysis,
            communities=communities,
            relevant_entities=entities or [],
            relevant_relationships=relationships or [],
            search_level=search_level,
            max_communities=max_communities
        )
        
        # 執行搜尋
        try:
            result = await search_strategy.search(context, self.llm_manager, self.vector_store)
            
            logger.info(f"全域搜尋成功完成: {result.llm_model_used}")
            return result
            
        except Exception as e:
            logger.error(f"全域搜尋失敗: {e}")
            
            # 降級策略：返回基本回答
            fallback_result = GlobalSearchResult(
                query=query,
                answer=f"抱歉，在執行全域搜尋時遇到問題：{str(e)}。請嘗試重新表述您的問題。",
                confidence=0.1,
                sources=[],
                communities_used=[],
                entities_used=analysis.entities,
                reasoning_path=["搜尋失敗，使用降級策略"],
                search_time=0.0,
                llm_model_used="fallback"
            )
            
            return fallback_result
    
    def _select_strategy(self, analysis: QueryAnalysis) -> str:
        """根據查詢分析選擇搜尋策略"""
        
        # 根據查詢類型選擇策略
        if analysis.query_type in [QueryType.SUMMARY, QueryType.COMPARISON]:
            return "hierarchical"
        elif analysis.query_type in [QueryType.GLOBAL_SEARCH]:
            if len(analysis.entities) > 3:
                return "hierarchical"
            else:
                return "community_based"
        else:
            return self.default_strategy
    
    def get_available_strategies(self) -> List[str]:
        """獲取可用的搜尋策略列表"""
        return list(self.strategies.keys())
    
    def add_strategy(self, name: str, strategy: GlobalSearchStrategy):
        """添加新的搜尋策略"""
        self.strategies[name] = strategy
        logger.info(f"添加搜尋策略: {name}")
    
    async def health_check(self) -> Dict[str, bool]:
        """檢查搜尋引擎健康狀態"""
        health_status = {
            "llm_manager": False,
            "vector_store": False,
            "strategies": False
        }
        
        try:
            # 檢查 LLM 管理器
            llm_health = await self.llm_manager.health_check_all()
            health_status["llm_manager"] = any(llm_health.values())
            
            # 檢查向量存儲（這裡需要實作 vector_store 的健康檢查）
            # health_status["vector_store"] = await self.vector_store.health_check()
            health_status["vector_store"] = True  # 暫時假設健康
            
            # 檢查策略
            health_status["strategies"] = len(self.strategies) > 0
            
        except Exception as e:
            logger.error(f"健康檢查失敗: {e}")
        
        return health_status