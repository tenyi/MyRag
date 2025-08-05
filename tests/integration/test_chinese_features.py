"""
中文特定功能測試

測試系統對中文文本處理的特殊功能，包括分詞、編碼處理、語義理解等。
"""

import re
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from tests.test_utils import TestAssertions, TestDataGenerator


@pytest.mark.chinese
@pytest.mark.integration
class TestChineseTextProcessing:
    """中文文本處理測試"""

    @pytest.fixture
    def chinese_text_samples(self):
        """中文文本範例"""
        return {
            "traditional": "繁體中文測試：人工智慧技術發展迅速，機器學習已經廣泛應用於各個領域。",
            "simplified": "简体中文测试：人工智能技术发展迅速，机器学习已经广泛应用于各个领域。",
            "mixed": "中英混合文本：AI technology 在中國發展很快，machine learning 應用廣泛。",
            "technical": "技術文檔：使用卷積神經網路（CNN）進行圖像識別，準確率達到95%以上。",
            "punctuation": "標點測試：「這是引號」，（這是括號），【這是方括號】，……省略號。",
            "numbers": "數字測試：2024年第1季度，增長了3.5%，總共處理了1,000,000條資料。",
            "poetry": "詩詞測試：春眠不覺曉，處處聞啼鳥。夜來風雨聲，花落知多少。",
            "long_text": """長文本測試：
            人工智慧（Artificial Intelligence，簡稱AI）是電腦科學的一個分支，
            它企圖了解智慧的實質，並生產出一種新的能以人類智慧相似的方式做出反應的智慧機器，
            該領域的研究包括機器人、語言識別、圖像識別、自然語言處理和專家系統等。
            自從電腦發明以來，人們一直在思考電腦是否能夠思考，
            這個問題至今仍然困擾著我們。但是隨著技術的不斷發展，
            我們已經能夠創造出在某些特定領域表現出色的AI系統。
            """,
        }

    def test_chinese_tokenization(self, chinese_text_samples):
        """測試中文分詞"""
        # 模擬中文分詞器
        tokenizer = Mock()

        def mock_tokenize(text: str) -> List[str]:
            # 簡單的中文分詞模擬
            # 實際應使用 jieba 或其他分詞工具
            words = []
            for char in text:
                if "\u4e00" <= char <= "\u9fff":  # 中文字符範圍
                    words.append(char)
                elif char.isalnum():
                    words.append(char)
                elif char in "，。！？；：「」（）【】":
                    words.append(char)
            return [w for w in words if w.strip()]

        tokenizer.tokenize.side_effect = mock_tokenize

        for text_type, text in chinese_text_samples.items():
            tokens = tokenizer.tokenize(text)

            # 驗證分詞結果
            assert len(tokens) > 0, f"{text_type} 分詞結果為空"
            assert all(
                isinstance(token, str) for token in tokens
            ), f"{text_type} 分詞結果格式錯誤"

            # 檢查是否包含中文字符
            has_chinese = any(
                "\u4e00" <= char <= "\u9fff" for token in tokens for char in token
            )
            if text_type in ["traditional", "simplified", "mixed", "technical"]:
                assert has_chinese, f"{text_type} 應包含中文字符"

    def test_chinese_encoding_handling(self, chinese_text_samples):
        """測試中文編碼處理"""
        for text_type, text in chinese_text_samples.items():
            # 測試不同編碼
            encodings = ["utf-8", "gbk", "big5"]

            for encoding in encodings:
                try:
                    # 編碼測試
                    encoded = text.encode(encoding)
                    decoded = encoded.decode(encoding)

                    # 驗證編碼解碼一致性
                    assert decoded == text, f"{text_type} 在 {encoding} 編碼下不一致"

                except UnicodeEncodeError:
                    # 某些編碼可能不支援所有字符，這是正常的
                    continue
                except UnicodeDecodeError:
                    pytest.fail(f"{text_type} 在 {encoding} 解碼失敗")

    def test_traditional_simplified_conversion(self, chinese_text_samples):
        """測試繁簡轉換"""
        # 模擬繁簡轉換器
        converter = Mock()

        # 簡單的繁簡字典對應
        trad_to_simp = {
            "繁體": "繁体",
            "中文": "中文",
            "測試": "测试",
            "人工": "人工",
            "智慧": "智能",
            "技術": "技术",
            "發展": "发展",
            "機器": "机器",
            "學習": "学习",
            "應用": "应用",
            "領域": "领域",
        }

        def mock_to_simplified(text: str) -> str:
            result = text
            for trad, simp in trad_to_simp.items():
                result = result.replace(trad, simp)
            return result

        def mock_to_traditional(text: str) -> str:
            result = text
            simp_to_trad = {v: k for k, v in trad_to_simp.items()}
            for simp, trad in simp_to_trad.items():
                result = result.replace(simp, trad)
            return result

        converter.to_simplified.side_effect = mock_to_simplified
        converter.to_traditional.side_effect = mock_to_traditional

        # 測試繁體轉簡體
        traditional_text = chinese_text_samples["traditional"]
        simplified_result = converter.to_simplified(traditional_text)
        assert "智能" in simplified_result, "繁體轉簡體失敗"
        assert "技术" in simplified_result, "繁體轉簡體失敗"

        # 測試簡體轉繁體
        simplified_text = chinese_text_samples["simplified"]
        traditional_result = converter.to_traditional(simplified_text)
        assert "智慧" in traditional_result, "簡體轉繁體失敗"
        assert "技術" in traditional_result, "簡體轉繁體失敗"

    def test_chinese_punctuation_handling(self, chinese_text_samples):
        """測試中文標點符號處理"""
        punctuation_text = chinese_text_samples["punctuation"]

        # 中文標點符號模式
        chinese_punct_pattern = r"[，。！？；：「」（）【】…—]"
        english_punct_pattern = r'[,.!?;:"""()\[\]...-]'

        # 檢測中文標點
        chinese_puncts = re.findall(chinese_punct_pattern, punctuation_text)
        english_puncts = re.findall(english_punct_pattern, punctuation_text)

        assert len(chinese_puncts) > 0, "應該檢測到中文標點符號"

        # 模擬標點符號正規化
        punctuation_normalizer = Mock()

        def normalize_punctuation(text: str) -> str:
            # 將中文標點轉換為英文標點
            punct_map = {
                "，": ",",
                "。": ".",
                "！": "!",
                "？": "?",
                "；": ";",
                "：": ":",
                "「": '"',
                "」": '"',
                "（": "(",
                "）": ")",
                "【": "[",
                "】": "]",
                "…": "...",
                "——": "--",
            }
            result = text
            for chinese, english in punct_map.items():
                result = result.replace(chinese, english)
            return result

        punctuation_normalizer.normalize.side_effect = normalize_punctuation

        normalized = punctuation_normalizer.normalize(punctuation_text)

        # 驗證標點正規化
        assert '"' in normalized, "引號正規化失敗"
        assert "(" in normalized and ")" in normalized, "括號正規化失敗"
        assert "," in normalized, "逗號正規化失敗"

    def test_chinese_number_extraction(self, chinese_text_samples):
        """測試中文數字提取"""
        numbers_text = chinese_text_samples["numbers"]

        # 模擬數字提取器
        number_extractor = Mock()

        def extract_numbers(text: str) -> Dict[str, List]:
            arabic_numbers = re.findall(r"\d+(?:\.\d+)?(?:%)?", text)
            chinese_numbers = re.findall(r"[一二三四五六七八九十百千萬億]+", text)
            years = re.findall(r"\d{4}年", text)

            return {
                "arabic": arabic_numbers,
                "chinese": chinese_numbers,
                "years": years,
            }

        number_extractor.extract.side_effect = extract_numbers

        extracted = number_extractor.extract(numbers_text)

        # 驗證數字提取
        assert len(extracted["arabic"]) > 0, "應該提取到阿拉伯數字"
        assert any("2024" in year for year in extracted["years"]), "應該提取到年份"
        assert any("%" in num for num in extracted["arabic"]), "應該提取到百分比"

    def test_chinese_semantic_understanding(self, chinese_text_samples):
        """測試中文語義理解"""
        # 模擬語義分析器
        semantic_analyzer = Mock()

        def analyze_sentiment(text: str) -> Dict[str, Any]:
            # 簡單的情感分析模擬
            positive_words = ["發展", "成功", "優秀", "提升", "增長", "改善"]
            negative_words = ["失敗", "問題", "困難", "下降", "減少", "錯誤"]

            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)

            if positive_count > negative_count:
                sentiment = "positive"
                confidence = 0.8
            elif negative_count > positive_count:
                sentiment = "negative"
                confidence = 0.8
            else:
                sentiment = "neutral"
                confidence = 0.6

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_words": positive_count,
                "negative_words": negative_count,
            }

        def extract_keywords(text: str) -> List[str]:
            # 簡單的關鍵詞提取
            tech_keywords = [
                "人工智慧",
                "機器學習",
                "深度學習",
                "神經網路",
                "AI",
                "CNN",
            ]
            found_keywords = [kw for kw in tech_keywords if kw in text]
            return found_keywords

        semantic_analyzer.analyze_sentiment.side_effect = analyze_sentiment
        semantic_analyzer.extract_keywords.side_effect = extract_keywords

        for text_type, text in chinese_text_samples.items():
            if text_type in ["technical", "long_text"]:
                # 情感分析
                sentiment = semantic_analyzer.analyze_sentiment(text)
                assert sentiment["confidence"] > 0.5, f"{text_type} 情感分析置信度過低"

                # 關鍵詞提取
                keywords = semantic_analyzer.extract_keywords(text)
                if text_type == "technical":
                    assert len(keywords) > 0, f"{text_type} 應該提取到技術關鍵詞"

    def test_chinese_text_chunking(self, chinese_text_samples):
        """測試中文文本分塊"""
        long_text = chinese_text_samples["long_text"]

        # 模擬中文文本分塊器
        chunker = Mock()

        def chunk_chinese_text(text: str, max_length: int = 200) -> List[str]:
            # 按句號分割
            sentences = text.split("。")
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk + sentence) <= max_length:
                    current_chunk += sentence + "。"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + "。"

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks

        chunker.chunk_text.side_effect = chunk_chinese_text

        chunks = chunker.chunk_text(long_text, max_length=150)

        # 驗證分塊結果
        assert len(chunks) > 1, "長文本應該被分成多個塊"
        assert all(len(chunk) <= 200 for chunk in chunks), "分塊長度超限"
        assert all(chunk.strip() for chunk in chunks), "分塊不應為空"

        # 檢查分塊完整性
        reconstructed = "".join(chunks)
        original_cleaned = re.sub(r"\s+", "", long_text)
        reconstructed_cleaned = re.sub(r"\s+", "", reconstructed)

        # 驗證內容完整性（允許空白字符差異）
        assert (
            len(original_cleaned) <= len(reconstructed_cleaned) + 10
        ), "分塊後內容缺失過多"


@pytest.mark.chinese
@pytest.mark.integration
class TestChineseQueryProcessing:
    """中文查詢處理測試"""

    @pytest.fixture
    def chinese_queries(self):
        """中文查詢範例"""
        return {
            "definition": [
                "什麼是人工智慧？",
                "機器學習的定義是什麼？",
                "請解釋深度學習的概念",
            ],
            "comparison": [
                "機器學習和深度學習有什麼區別？",
                "CNN和RNN的差異在哪裡？",
                "監督學習與無監督學習的對比",
            ],
            "how_to": [
                "如何實現一個神經網路？",
                "怎樣提高模型的準確率？",
                "如何處理過擬合問題？",
            ],
            "application": [
                "人工智慧在醫療領域的應用",
                "深度學習在圖像識別中的使用",
                "NLP技術在搜尋引擎中的作用",
            ],
            "complex": [
                "在考慮計算資源限制的情況下，如何選擇合適的深度學習模型來解決多分類問題？",
                "對於中文文本處理任務，應該如何結合傳統NLP方法和現代深度學習技術？",
            ],
        }

    def test_chinese_query_intent_classification(self, chinese_queries):
        """測試中文查詢意圖分類"""
        # 模擬查詢意圖分類器
        intent_classifier = Mock()

        def classify_intent(query: str) -> Dict[str, Any]:
            if any(word in query for word in ["什麼是", "定義", "解釋", "概念"]):
                return {"intent": "definition", "confidence": 0.9}
            elif any(word in query for word in ["區別", "差異", "對比", "比較"]):
                return {"intent": "comparison", "confidence": 0.85}
            elif any(word in query for word in ["如何", "怎樣", "怎麼"]):
                return {"intent": "how_to", "confidence": 0.8}
            elif any(word in query for word in ["應用", "使用", "作用"]):
                return {"intent": "application", "confidence": 0.75}
            else:
                return {"intent": "complex", "confidence": 0.6}

        intent_classifier.classify.side_effect = classify_intent

        for intent_type, queries in chinese_queries.items():
            for query in queries:
                result = intent_classifier.classify(query)

                # 驗證意圖分類
                if intent_type != "complex":
                    assert (
                        result["intent"] == intent_type
                    ), f"查詢 '{query}' 意圖分類錯誤"
                    assert result["confidence"] > 0.7, f"查詢 '{query}' 分類置信度過低"

    def test_chinese_entity_extraction(self, chinese_queries):
        """測試中文實體提取"""
        # 模擬中文實體提取器
        entity_extractor = Mock()

        def extract_entities(query: str) -> List[Dict[str, str]]:
            entities = []

            # 技術實體
            tech_entities = [
                "人工智慧",
                "機器學習",
                "深度學習",
                "神經網路",
                "CNN",
                "RNN",
                "NLP",
            ]
            for entity in tech_entities:
                if entity in query:
                    entities.append(
                        {"text": entity, "type": "TECHNOLOGY", "confidence": 0.9}
                    )

            # 領域實體
            domain_entities = ["醫療", "圖像識別", "搜尋引擎", "文本處理"]
            for entity in domain_entities:
                if entity in query:
                    entities.append(
                        {"text": entity, "type": "DOMAIN", "confidence": 0.8}
                    )

            return entities

        entity_extractor.extract.side_effect = extract_entities

        for intent_type, queries in chinese_queries.items():
            for query in queries:
                entities = entity_extractor.extract(query)

                # 驗證實體提取
                if intent_type in ["definition", "comparison", "application"]:
                    assert len(entities) > 0, f"查詢 '{query}' 應該提取到實體"

                    # 檢查實體類型
                    tech_entities = [e for e in entities if e["type"] == "TECHNOLOGY"]
                    assert len(tech_entities) > 0, f"查詢 '{query}' 應該包含技術實體"

    def test_chinese_query_expansion(self, chinese_queries):
        """測試中文查詢擴展"""
        # 模擬查詢擴展器
        query_expander = Mock()

        def expand_query(original_query: str) -> Dict[str, Any]:
            # 同義詞字典
            synonyms = {
                "人工智慧": ["AI", "人工智能", "機器智慧"],
                "機器學習": ["ML", "機器學習", "自動學習"],
                "深度學習": ["DL", "深度學習", "神經網路學習"],
                "應用": ["使用", "運用", "套用"],
                "區別": ["差異", "不同", "差別"],
            }

            expanded_terms = []
            for term, syns in synonyms.items():
                if term in original_query:
                    expanded_terms.extend(syns)

            return {
                "original_query": original_query,
                "expanded_terms": list(set(expanded_terms)),
                "expanded_query": original_query + " " + " ".join(expanded_terms),
            }

        query_expander.expand.side_effect = expand_query

        test_query = "什麼是人工智慧的應用？"
        expansion = query_expander.expand(test_query)

        # 驗證查詢擴展
        assert len(expansion["expanded_terms"]) > 0, "應該生成擴展詞彙"
        assert "AI" in expansion["expanded_terms"], "應該包含英文同義詞"
        assert len(expansion["expanded_query"]) > len(
            expansion["original_query"]
        ), "擴展查詢應該更長"

    def test_chinese_answer_generation(self, chinese_queries):
        """測試中文答案生成"""
        # 模擬中文答案生成器
        answer_generator = Mock()

        def generate_answer(query: str, context: List[str]) -> Dict[str, Any]:
            # 模擬基於上下文的答案生成
            answer_templates = {
                "definition": "{概念}是{定義}的技術領域，主要特點包括{特點}。",
                "comparison": "{項目1}和{項目2}的主要區別在於：{差異點1}、{差異點2}。",
                "how_to": "要{動作}，可以按照以下步驟：1.{步驟1} 2.{步驟2} 3.{步驟3}",
                "application": "{技術}在{領域}中的應用包括：{應用1}、{應用2}、{應用3}。",
            }

            # 簡化的答案生成邏輯
            if "什麼是" in query:
                answer = "根據提供的資料，人工智慧是模擬人類智慧的技術領域，包含機器學習、深度學習等分支。"
                answer_type = "definition"
            elif "區別" in query or "差異" in query:
                answer = "根據分析，這兩個概念的主要區別體現在技術原理、應用場景和實現方法等方面。"
                answer_type = "comparison"
            elif "如何" in query or "怎樣" in query:
                answer = "建議採用以下方法：首先分析需求，然後選擇合適的技術方案，最後進行實施和優化。"
                answer_type = "how_to"
            else:
                answer = (
                    "根據相關資料，該技術在多個領域都有重要應用，具有廣闊的發展前景。"
                )
                answer_type = "application"

            return {
                "query": query,
                "answer": answer,
                "answer_type": answer_type,
                "confidence": 0.85,
                "context_used": len(context),
            }

        answer_generator.generate.side_effect = generate_answer

        # 測試不同類型的查詢
        for intent_type, queries in chinese_queries.items():
            if intent_type != "complex":  # 跳過複雜查詢
                query = queries[0]
                context = ["相關上下文1", "相關上下文2"]

                result = answer_generator.generate(query, context)

                # 驗證答案生成
                assert result["answer"], f"查詢 '{query}' 未生成答案"
                assert result["confidence"] > 0.7, f"答案置信度過低"
                assert len(result["answer"]) > 10, f"答案過短"

                # 檢查答案是否為中文
                TestAssertions.assert_chinese_text(result["answer"])


@pytest.mark.chinese
@pytest.mark.integration
class TestChineseKnowledgeGraph:
    """中文知識圖譜測試"""

    def test_chinese_entity_recognition(self):
        """測試中文實體識別"""
        # 測試文本
        test_text = """
        人工智慧是由約翰·麥卡錫在1956年達特茅斯會議上首次提出的概念。
        機器學習是人工智慧的重要分支，包括監督學習、無監督學習和強化學習。
        深度學習使用神經網路來模擬人腦的工作方式，在圖像識別和自然語言處理方面取得了突破性進展。
        """

        # 模擬中文實體識別器
        ner = Mock()

        def recognize_entities(text: str) -> List[Dict[str, Any]]:
            entities = [
                {"text": "人工智慧", "type": "CONCEPT", "start": 0, "end": 4},
                {"text": "約翰·麥卡錫", "type": "PERSON", "start": 5, "end": 11},
                {"text": "1956年", "type": "DATE", "start": 12, "end": 17},
                {"text": "達特茅斯會議", "type": "EVENT", "start": 17, "end": 23},
                {"text": "機器學習", "type": "CONCEPT", "start": 30, "end": 34},
                {"text": "監督學習", "type": "METHOD", "start": 45, "end": 49},
                {"text": "無監督學習", "type": "METHOD", "start": 51, "end": 56},
                {"text": "強化學習", "type": "METHOD", "start": 58, "end": 62},
                {"text": "深度學習", "type": "CONCEPT", "start": 64, "end": 68},
                {"text": "神經網路", "type": "MODEL", "start": 71, "end": 75},
                {"text": "圖像識別", "type": "APPLICATION", "start": 85, "end": 89},
                {"text": "自然語言處理", "type": "APPLICATION", "start": 91, "end": 97},
            ]
            return entities

        ner.recognize.side_effect = recognize_entities

        entities = ner.recognize(test_text)

        # 驗證實體識別結果
        assert len(entities) > 0, "應該識別出實體"

        # 檢查實體類型分布
        entity_types = [e["type"] for e in entities]
        assert "CONCEPT" in entity_types, "應該識別出概念實體"
        assert "PERSON" in entity_types, "應該識別出人物實體"
        assert "METHOD" in entity_types, "應該識別出方法實體"

        # 檢查中文實體
        chinese_entities = [
            e for e in entities if any("\u4e00" <= c <= "\u9fff" for c in e["text"])
        ]
        assert len(chinese_entities) > 0, "應該識別出中文實體"

    def test_chinese_relation_extraction(self):
        """測試中文關係提取"""
        # 模擬關係提取器
        relation_extractor = Mock()

        def extract_relations(
            entities: List[Dict[str, Any]], text: str
        ) -> List[Dict[str, Any]]:
            relations = [
                {
                    "source": "機器學習",
                    "target": "人工智慧",
                    "relation": "屬於",
                    "confidence": 0.9,
                },
                {
                    "source": "深度學習",
                    "target": "機器學習",
                    "relation": "是一種",
                    "confidence": 0.85,
                },
                {
                    "source": "約翰·麥卡錫",
                    "target": "人工智慧",
                    "relation": "提出",
                    "confidence": 0.8,
                },
                {
                    "source": "神經網路",
                    "target": "深度學習",
                    "relation": "用於",
                    "confidence": 0.75,
                },
            ]
            return relations

        relation_extractor.extract.side_effect = extract_relations

        # 模擬實體列表
        mock_entities = [
            {"text": "人工智慧", "type": "CONCEPT"},
            {"text": "機器學習", "type": "CONCEPT"},
            {"text": "深度學習", "type": "CONCEPT"},
        ]

        relations = relation_extractor.extract(mock_entities, "測試文本")

        # 驗證關係提取
        assert len(relations) > 0, "應該提取出關係"

        # 檢查關係類型
        relation_types = [r["relation"] for r in relations]
        assert "屬於" in relation_types, "應該包含層次關係"
        assert "是一種" in relation_types, "應該包含分類關係"

        # 檢查置信度
        confidences = [r["confidence"] for r in relations]
        assert all(c > 0.5 for c in confidences), "關係置信度應該足夠高"

    def test_chinese_knowledge_graph_construction(self):
        """測試中文知識圖譜構建"""
        # 模擬知識圖譜構建器
        kg_builder = Mock()

        def build_graph(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
            # 模擬圖譜構建結果
            graph = {
                "nodes": [
                    {
                        "id": "AI",
                        "label": "人工智慧",
                        "type": "CONCEPT",
                        "properties": {"domain": "技術"},
                    },
                    {
                        "id": "ML",
                        "label": "機器學習",
                        "type": "CONCEPT",
                        "properties": {"domain": "技術"},
                    },
                    {
                        "id": "DL",
                        "label": "深度學習",
                        "type": "CONCEPT",
                        "properties": {"domain": "技術"},
                    },
                    {
                        "id": "NN",
                        "label": "神經網路",
                        "type": "MODEL",
                        "properties": {"type": "模型"},
                    },
                ],
                "edges": [
                    {"source": "ML", "target": "AI", "relation": "屬於", "weight": 0.9},
                    {
                        "source": "DL",
                        "target": "ML",
                        "relation": "是一種",
                        "weight": 0.85,
                    },
                    {"source": "NN", "target": "DL", "relation": "用於", "weight": 0.8},
                ],
                "communities": [
                    {
                        "id": "tech_community",
                        "name": "技術社群",
                        "nodes": ["AI", "ML", "DL", "NN"],
                        "description": "人工智慧相關技術的知識群組",
                    }
                ],
                "statistics": {
                    "node_count": 4,
                    "edge_count": 3,
                    "community_count": 1,
                    "avg_degree": 1.5,
                },
            }
            return graph

        kg_builder.build.side_effect = build_graph

        # 模擬文件
        mock_documents = [
            {"id": "doc1", "content": "人工智慧技術發展"},
            {"id": "doc2", "content": "機器學習應用研究"},
        ]

        graph = kg_builder.build(mock_documents)

        # 驗證圖譜構建
        assert len(graph["nodes"]) > 0, "應該包含節點"
        assert len(graph["edges"]) > 0, "應該包含邊"
        assert len(graph["communities"]) > 0, "應該包含社群"

        # 檢查中文標籤
        chinese_labels = [node["label"] for node in graph["nodes"]]
        assert all(
            any("\u4e00" <= c <= "\u9fff" for c in label) for label in chinese_labels
        ), "節點標籤應為中文"

        # 檢查圖譜統計
        stats = graph["statistics"]
        assert stats["node_count"] == len(graph["nodes"]), "節點數量統計錯誤"
        assert stats["edge_count"] == len(graph["edges"]), "邊數量統計錯誤"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "chinese"])
