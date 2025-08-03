"""
中文查詢處理器

專門處理中文查詢的意圖理解、查詢類型判斷和 LLM 路由策略。
針對中文語言特性進行優化，提供智慧的查詢處理功能。
"""

import re
import jieba
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from loguru import logger

from .manager import TaskType


class QueryType(str, Enum):
    """查詢類型"""
    GLOBAL_SEARCH = "global_search"    # 全域搜尋：需要整體知識圖譜的高層次問答
    LOCAL_SEARCH = "local_search"      # 本地搜尋：針對特定實體或關係的詳細問答
    ENTITY_SEARCH = "entity_search"    # 實體搜尋：查找特定實體資訊
    RELATION_SEARCH = "relation_search" # 關係搜尋：查找實體間關係
    SUMMARY = "summary"                # 摘要：生成概括性回答
    COMPARISON = "comparison"          # 比較：比較多個實體或概念
    CAUSAL = "causal"                  # 因果：分析因果關係
    TEMPORAL = "temporal"              # 時間：涉及時間序列的查詢


class QueryIntent(str, Enum):
    """查詢意圖"""
    INFORMATION_SEEKING = "information_seeking"  # 資訊尋求
    FACT_CHECKING = "fact_checking"             # 事實查證
    EXPLANATION = "explanation"                 # 解釋說明
    COMPARISON = "comparison"                   # 比較分析
    RECOMMENDATION = "recommendation"           # 建議推薦
    PROBLEM_SOLVING = "problem_solving"         # 問題解決
    PLANNING = "planning"                       # 規劃安排


@dataclass
class QueryAnalysis:
    """查詢分析結果"""
    original_query: str
    normalized_query: str
    query_type: QueryType
    intent: QueryIntent
    entities: List[str]
    keywords: List[str]
    confidence: float
    suggested_llm_task: TaskType
    preprocessing_notes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "query_type": self.query_type.value,
            "intent": self.intent.value,
            "entities": self.entities,
            "keywords": self.keywords,
            "confidence": self.confidence,
            "suggested_llm_task": self.suggested_llm_task.value,
            "preprocessing_notes": self.preprocessing_notes
        }


class QueryClassifier(ABC):
    """查詢分類器抽象基類"""
    
    @abstractmethod
    def classify_query_type(self, query: str, entities: List[str]) -> Tuple[QueryType, float]:
        """分類查詢類型"""
        pass
    
    @abstractmethod
    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """分類查詢意圖"""
        pass


class RuleBasedClassifier(QueryClassifier):
    """基於規則的查詢分類器"""
    
    def __init__(self):
        # 查詢類型關鍵詞模式
        self.query_type_patterns = {
            QueryType.GLOBAL_SEARCH: [
                r'整體|全域|總體|概況|全貌|綜合|宏觀',
                r'所有.*?是什麼|全部.*?如何',
                r'總共有.*?個|一共.*?種',
                r'系統.*?功能|平台.*?特點'
            ],
            QueryType.LOCAL_SEARCH: [
                r'具體|詳細|特定|明確|準確',
                r'.*?是什麼|.*?如何|.*?為什麼',
                r'告訴我.*?的|介紹.*?的',
                r'.*?的定義|.*?的含義'
            ],
            QueryType.ENTITY_SEARCH: [
                r'.*?公司|.*?組織|.*?機構',
                r'.*?人物|.*?專家|.*?學者',
                r'.*?產品|.*?服務|.*?技術',
                r'尋找|查找|搜尋.*?相關'
            ],
            QueryType.RELATION_SEARCH: [
                r'.*?和.*?的關係|.*?與.*?之間',
                r'.*?如何影響.*?|.*?導致.*?',
                r'.*?屬於.*?|.*?包含.*?',
                r'相關性|關聯性|依賴關係'
            ],
            QueryType.SUMMARY: [
                r'總結|摘要|概括|歸納',
                r'簡述|簡要說明|大致介紹',
                r'主要內容|核心要點|重點'
            ],
            QueryType.COMPARISON: [
                r'比較|對比|區別|差異',
                r'.*?vs.*?|.*?和.*?哪個',
                r'優缺點|利弊|好壞',
                r'相同點|不同點|異同'
            ],
            QueryType.CAUSAL: [
                r'為什麼|原因|起因|導致',
                r'因為|由於|造成|引起',
                r'結果|後果|影響|效果',
                r'如何產生|怎麼形成'
            ],
            QueryType.TEMPORAL: [
                r'時間|日期|年份|月份',
                r'何時|什麼時候|多久',
                r'之前|之後|期間|過程',
                r'發展歷程|演變過程|時間線'
            ]
        }
        
        # 查詢意圖關鍵詞模式
        self.intent_patterns = {
            QueryIntent.INFORMATION_SEEKING: [
                r'是什麼|什麼是|如何|怎樣',
                r'告訴我|介紹|說明|解釋',
                r'資訊|資料|內容|詳情'
            ],
            QueryIntent.FACT_CHECKING: [
                r'是否|是不是|對不對|正確嗎',
                r'真的嗎|確定嗎|可信嗎',
                r'驗證|查證|確認|核實'
            ],
            QueryIntent.EXPLANATION: [
                r'為什麼|怎麼回事|原理|機制',
                r'解釋|說明|闡述|論述',
                r'工作原理|運作方式|實現方法'
            ],
            QueryIntent.COMPARISON: [
                r'比較|對比|區別|差異',
                r'哪個更好|哪種更適合',
                r'優勢|劣勢|特點|特色'
            ],
            QueryIntent.RECOMMENDATION: [
                r'推薦|建議|選擇|挑選',
                r'應該|最好|適合|合適',
                r'方案|策略|方法|途徑'
            ],
            QueryIntent.PROBLEM_SOLVING: [
                r'解決|處理|應對|克服',
                r'問題|困難|挑戰|障礙',
                r'方法|辦法|措施|對策'
            ],
            QueryIntent.PLANNING: [
                r'計劃|規劃|安排|組織',
                r'如何實施|怎麼執行|步驟',
                r'流程|程序|順序|階段'
            ]
        }
    
    def classify_query_type(self, query: str, entities: List[str]) -> Tuple[QueryType, float]:
        """分類查詢類型"""
        scores = {}
        
        for query_type, patterns in self.query_type_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 1.0
            
            # 根據實體數量調整分數
            if query_type == QueryType.ENTITY_SEARCH and len(entities) > 0:
                score += 0.5
            elif query_type == QueryType.RELATION_SEARCH and len(entities) >= 2:
                score += 0.5
            elif query_type == QueryType.GLOBAL_SEARCH and len(entities) > 3:
                score += 0.3
            
            scores[query_type] = score / len(patterns) if patterns else 0.0
        
        # 如果沒有明確匹配，根據實體數量判斷
        if all(score < 0.1 for score in scores.values()):
            if len(entities) == 0:
                return QueryType.GLOBAL_SEARCH, 0.6
            elif len(entities) == 1:
                return QueryType.ENTITY_SEARCH, 0.6
            elif len(entities) >= 2:
                return QueryType.RELATION_SEARCH, 0.6
        
        best_type = max(scores.items(), key=lambda x: x[1])
        return best_type[0], min(best_type[1], 1.0)
    
    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """分類查詢意圖"""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 1.0
            scores[intent] = score / len(patterns) if patterns else 0.0
        
        # 預設為資訊尋求
        if all(score < 0.1 for score in scores.values()):
            return QueryIntent.INFORMATION_SEEKING, 0.5
        
        best_intent = max(scores.items(), key=lambda x: x[1])
        return best_intent[0], min(best_intent[1], 1.0)


class ChineseTextNormalizer:
    """中文文本正規化器"""
    
    def __init__(self):
        # 載入停用詞
        self.stopwords = self._load_stopwords()
        
        # 標點符號模式
        self.punctuation_pattern = r'[！？。，、；：""''（）《》【】『』「」〈〉〔〕…—～·]'
        
        # 數字模式
        self.number_pattern = r'\d+\.?\d*'
        
        # 英文字母模式
        self.english_pattern = r'[a-zA-Z]+'
    
    def _load_stopwords(self) -> Set[str]:
        """載入中文停用詞"""
        # 基本中文停用詞
        basic_stopwords = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一個',
            '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有', '看', '好',
            '自己', '這', '而且', '那', '什麼', '為', '但是', '如果', '他', '她', '它',
            '這個', '那個', '可以', '還', '對', '因為', '所以', '然後', '還是', '或者',
            '以及', '等等', '之類', '什麼的', '之前', '之後', '當中', '其中', '包括'
        }
        return basic_stopwords
    
    def normalize(self, text: str) -> str:
        """正規化文本"""
        # 轉換為簡體中文（如果需要）
        # text = self._traditional_to_simplified(text)
        
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 統一標點符號
        text = re.sub(r'[？?]', '？', text)
        text = re.sub(r'[！!]', '！', text)
        text = re.sub(r'[，,]', '，', text)
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """提取關鍵詞"""
        # 分詞
        words = jieba.cut(text)
        
        # 過濾停用詞、標點符號和短詞
        keywords = []
        for word in words:
            word = word.strip()
            if (word and 
                len(word) > 1 and 
                word not in self.stopwords and
                not re.match(self.punctuation_pattern, word) and
                not re.match(r'^\d+$', word)):
                keywords.append(word)
        
        # 去重並保持順序
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:max_keywords]


class EntityExtractor:
    """實體提取器"""
    
    def __init__(self):
        # 實體模式
        self.entity_patterns = {
            'person': r'[\u4e00-\u9fff]{2,4}(?:先生|女士|教授|博士|主任|經理|總監|董事|局長|部長)',
            'organization': r'[\u4e00-\u9fff]{2,15}(?:公司|企業|機構|組織|部門|學校|大學|研究所|協會|基金會)',
            'location': r'[\u4e00-\u9fff]{2,10}(?:市|縣|區|省|國|地區|城市|鄉|鎮|村|街道)',
            'product': r'[\u4e00-\u9fff]{2,10}(?:系統|平台|軟體|應用|工具|服務|產品|技術)',
            'concept': r'[\u4e00-\u9fff]{2,8}(?:概念|理論|方法|模型|框架|標準|規範|原則)'
        }
    
    def extract_entities(self, text: str) -> List[str]:
        """從文本中提取實體"""
        entities = []
        found_entities = set()
        
        # 1. 使用正則表達式提取實體
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in found_entities:
                    entities.append(match)
                    found_entities.add(match)
        
        # 2. 改進的分詞方法
        # 2a. 使用精確模式分詞
        words_exact = list(jieba.cut(text, cut_all=False))
        
        # 2b. 使用搜索引擎模式分詞（對實體識別更友好）
        words_search = list(jieba.cut_for_search(text))
        
        # 2c. 單字符分割作為備選
        chars = list(text)
        
        # 合併所有分詞結果
        all_tokens = set(words_exact + words_search + chars)
        
        # 3. 從所有分詞結果中提取可能的實體
        for word in all_tokens:
            word = word.strip()
            if (len(word) >= 2 and 
                word not in found_entities and
                self._is_likely_entity(word)):
                entities.append(word)
                found_entities.add(word)
        
        # 4. 特殊處理中文人名
        person_entities = self._extract_chinese_names(text)
        for person in person_entities:
            if person not in found_entities:
                entities.append(person)
                found_entities.add(person)
        
        return entities[:15]  # 增加實體數量限制以獲得更多候選  # 限制實體數量
    
    def _is_likely_entity(self, word: str) -> bool:
        """判斷詞語是否可能是實體"""
        # 排除常見動詞、形容詞等
        exclude_words = {
            '可以', '應該', '需要', '必須', '能夠', '想要', '希望', '建議',
            '重要', '必要', '有效', '成功', '失敗', '困難', '簡單', '複雜',
            '進行', '實施', '執行', '完成', '開始', '結束', '繼續', '停止'
        }
        
        return (word not in exclude_words and
                not re.match(r'^[\d\.\,]+$', word) and
                len(word) >= 2)

    
    def _extract_chinese_names(self, text: str) -> List[str]:
        """提取中文人名的專門方法"""
        names = []
        
        # 常見中文姓氏
        common_surnames = {
            '王', '李', '張', '劉', '陳', '楊', '趙', '黃', '周', '吳',
            '徐', '孫', '胡', '朱', '高', '林', '何', '郭', '馬', '羅',
            '梁', '宋', '鄭', '謝', '韓', '唐', '馮', '于', '董', '蕭',
            '程', '柴', '袁', '鄧', '許', '傅', '沈', '曾', '彭', '呂'
        }
        
        # 1. 基於姓氏的人名模式匹配
        for surname in common_surnames:
            # 姓 + 1-2個字的名字
            pattern = rf'{surname}[一-龥]{{1,2}}'
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        # 2. 常見人名模式 - 修正 lookbehind 問題
        name_patterns = [
            r'[一-龥]{2,3}(?=姓|的|是|在|有|和|與)',  # 名字 + 動作詞
            r'[一-龥]{2}(?=先生|女士|老師|經理|主任)',  # 職稱前的名字
        ]
        
        # 使用簡單的前後文匹配替代 lookbehind
        # 尋找 "叫XXX"、"名叫XXX"、"稱為XXX" 的模式
        call_patterns = [
            r'叫([一-龥]{2,3})',
            r'名叫([一-龥]{2,3})',
            r'稱為([一-龥]{2,3})',
        ]
        
        for pattern in call_patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        # 3. 移除重複並過濾
        unique_names = []
        seen = set()
        for name in names:
            if name not in seen and len(name) >= 2 and len(name) <= 4:
                unique_names.append(name)
                seen.add(name)
        
        return unique_names


class ChineseQueryProcessor:
    """中文查詢處理器
    
    專門處理中文查詢的意圖理解、查詢類型判斷和 LLM 路由策略。
    提供智慧的中文查詢分析和處理功能。
    """
    
    def __init__(self):
        """初始化中文查詢處理器"""
        self.classifier = RuleBasedClassifier()
        self.normalizer = ChineseTextNormalizer()
        self.entity_extractor = EntityExtractor()
        
        logger.info("中文查詢處理器初始化完成")
    
    def process_query(self, query: str) -> QueryAnalysis:
        """
        處理中文查詢
        
        Args:
            query: 原始查詢字串
            
        Returns:
            查詢分析結果
        """
        logger.debug(f"處理查詢: {query}")
        
        preprocessing_notes = []
        
        # 1. 文本正規化
        normalized_query = self.normalizer.normalize(query)
        if normalized_query != query:
            preprocessing_notes.append("文本已正規化")
        
        # 2. 提取實體
        entities = self.entity_extractor.extract_entities(normalized_query)
        if entities:
            preprocessing_notes.append(f"提取到 {len(entities)} 個實體")
        
        # 3. 提取關鍵詞
        keywords = self.normalizer.extract_keywords(normalized_query)
        if keywords:
            preprocessing_notes.append(f"提取到 {len(keywords)} 個關鍵詞")
        
        # 4. 分類查詢類型
        query_type, type_confidence = self.classifier.classify_query_type(
            normalized_query, entities
        )
        
        # 5. 分類查詢意圖
        intent, intent_confidence = self.classifier.classify_intent(normalized_query)
        
        # 6. 計算整體信心度
        overall_confidence = (type_confidence + intent_confidence) / 2
        
        # 7. 建議 LLM 任務類型
        suggested_llm_task = self._suggest_llm_task(query_type, intent, entities)
        
        # 8. 建立分析結果
        analysis = QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            query_type=query_type,
            intent=intent,
            entities=entities,
            keywords=keywords,
            confidence=overall_confidence,
            suggested_llm_task=suggested_llm_task,
            preprocessing_notes=preprocessing_notes
        )
        
        logger.info(f"查詢分析完成: {query_type.value} ({overall_confidence:.2f})")
        return analysis
    
    def _suggest_llm_task(
        self, 
        query_type: QueryType, 
        intent: QueryIntent, 
        entities: List[str]
    ) -> TaskType:
        """建議適合的 LLM 任務類型"""
        
        # 根據查詢類型映射到 LLM 任務
        type_to_task = {
            QueryType.GLOBAL_SEARCH: TaskType.GLOBAL_SEARCH,
            QueryType.LOCAL_SEARCH: TaskType.LOCAL_SEARCH,
            QueryType.ENTITY_SEARCH: TaskType.LOCAL_SEARCH,
            QueryType.RELATION_SEARCH: TaskType.LOCAL_SEARCH,
            QueryType.SUMMARY: TaskType.GLOBAL_SEARCH,
            QueryType.COMPARISON: TaskType.GLOBAL_SEARCH,
            QueryType.CAUSAL: TaskType.GLOBAL_SEARCH,
            QueryType.TEMPORAL: TaskType.GLOBAL_SEARCH
        }
        
        suggested_task = type_to_task.get(query_type, TaskType.GENERAL_QA)
        
        # 根據意圖和實體數量進行微調
        if intent == QueryIntent.FACT_CHECKING and len(entities) <= 2:
            suggested_task = TaskType.LOCAL_SEARCH
        elif intent == QueryIntent.COMPARISON and len(entities) >= 2:
            suggested_task = TaskType.GLOBAL_SEARCH
        elif intent == QueryIntent.EXPLANATION:
            suggested_task = TaskType.GLOBAL_SEARCH
        
        return suggested_task
    
    def suggest_query_enhancement(self, analysis: QueryAnalysis) -> List[str]:
        """建議查詢增強策略"""
        suggestions = []
        
        # 信心度較低時的建議
        if analysis.confidence < 0.6:
            suggestions.append("查詢可能需要更明確的表達")
            
            if not analysis.entities:
                suggestions.append("建議在查詢中包含具體的實體名稱")
            
            if len(analysis.keywords) < 3:
                suggestions.append("建議添加更多關鍵詞來明確查詢意圖")
        
        # 根據查詢類型的建議
        if analysis.query_type == QueryType.GLOBAL_SEARCH:
            suggestions.append("這是一個全域查詢，系統將基於整體知識圖譜回答")
        elif analysis.query_type == QueryType.LOCAL_SEARCH:
            suggestions.append("這是一個本地查詢，系統將基於相關實體詳細回答")
        
        # 根據實體數量的建議
        if len(analysis.entities) > 5:
            suggestions.append("查詢包含較多實體，可能需要分解為多個子查詢")
        
        return suggestions
    
    def get_query_context(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """獲取查詢上下文資訊，用於後續處理"""
        return {
            "query_type": analysis.query_type.value,
            "intent": analysis.intent.value,
            "entities": analysis.entities,
            "keywords": analysis.keywords,
            "confidence": analysis.confidence,
            "suggested_strategy": self._get_search_strategy(analysis),
            "enhancement_suggestions": self.suggest_query_enhancement(analysis)
        }
    
    def _get_search_strategy(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """獲取搜尋策略建議"""
        strategy = {
            "primary_method": "global_search" if analysis.query_type in [
                QueryType.GLOBAL_SEARCH, QueryType.SUMMARY, QueryType.COMPARISON
            ] else "local_search",
            "entity_focus": len(analysis.entities) > 0,
            "requires_reasoning": analysis.intent in [
                QueryIntent.EXPLANATION, QueryIntent.PROBLEM_SOLVING, QueryIntent.PLANNING
            ],
            "complexity": "high" if analysis.confidence > 0.8 else "medium"
        }
        
        return strategy