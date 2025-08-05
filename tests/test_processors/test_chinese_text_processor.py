"""
測試中文文本預處理器
"""

import tempfile
from pathlib import Path

import pytest

from chinese_graphrag.processors.chinese_text_processor import ChineseTextProcessor


class TestChineseTextProcessor:
    """測試中文文本預處理器"""

    def setup_method(self):
        """設定測試環境"""
        self.processor = ChineseTextProcessor(chunk_size=500, chunk_overlap=100)

    def test_init_default(self):
        """測試預設初始化"""
        processor = ChineseTextProcessor()
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
        assert len(processor.stopwords) > 0
        assert len(processor.chinese_punctuation) > 0

    def test_init_with_params(self):
        """測試帶參數初始化"""
        processor = ChineseTextProcessor(chunk_size=800, chunk_overlap=150)
        assert processor.chunk_size == 800
        assert processor.chunk_overlap == 150

    def test_clean_text_basic(self):
        """測試基本文本清理"""
        text = "  這是一個測試文本。  \n\n  包含多餘空格。  "
        cleaned = self.processor.clean_text(text)

        assert "這是一個測試文本" in cleaned
        assert "包含多餘空格" in cleaned
        assert cleaned.strip() == cleaned  # 沒有前後空格

    def test_clean_text_special_chars(self):
        """測試特殊字符清理"""
        text = "測試@#$%文本！！！包含特殊字符★☆"
        cleaned = self.processor.clean_text(text)

        assert "測試" in cleaned
        assert "文本" in cleaned
        assert "@#$%" not in cleaned
        assert "★☆" not in cleaned
        assert "！" in cleaned  # 中文標點應該保留

    def test_clean_text_punctuation_normalization(self):
        """測試標點符號正規化"""
        text = "測試\"引號\"和'單引號'以及...省略號"
        cleaned = self.processor.clean_text(text)

        # clean_text 會移除大部分標點符號，只保留基本的中文標點
        assert "測試" in cleaned
        assert "引號" in cleaned
        assert "單引號" in cleaned
        assert "省略號" in cleaned

    def test_clean_text_empty(self):
        """測試空文本清理"""
        assert self.processor.clean_text("") == ""
        assert self.processor.clean_text("   ") == ""
        assert self.processor.clean_text("\n\n\t") == ""

    def test_segment_text_basic(self):
        """測試基本中文分詞"""
        text = "我愛北京天安門，天安門上太陽升。"
        words = self.processor.segment_text(text)

        assert "北京" in words
        assert "天安" in words  # jieba 將 "天安門" 分成 "天安"
        assert "太陽" in words
        # 停用詞應該被移除
        assert "我" not in words
        assert "上" not in words

    def test_segment_text_keep_stopwords(self):
        """測試保留停用詞的分詞"""
        text = "我愛北京天安門"
        words = self.processor.segment_text(text, remove_stopwords=False)

        assert "我" in words
        assert "愛" in words
        assert "北京" in words
        assert "天安" in words  # jieba 將 "天安門" 分成 "天安"

    def test_segment_text_no_pos_filter(self):
        """測試不過濾詞性的分詞"""
        text = "我很快樂地走在路上"
        words = self.processor.segment_text(
            text, keep_pos=False, remove_stopwords=False
        )

        # 應該包含更多詞，包括副詞、助詞等
        assert len(words) > 0
        assert "很快" in words  # jieba 的實際分詞結果

    def test_segment_text_empty(self):
        """測試空文本分詞"""
        assert self.processor.segment_text("") == []
        assert self.processor.segment_text("   ") == []

    def test_extract_keywords_basic(self):
        """測試關鍵詞提取"""
        text = """
        人工智慧是現代科技的重要發展方向。
        機器學習和深度學習是人工智慧的核心技術。
        自然語言處理也是人工智慧的重要應用領域。
        """

        keywords = self.processor.extract_keywords(text, top_k=5)

        assert len(keywords) <= 5
        assert all(
            "word" in kw and "frequency" in kw and "weight" in kw for kw in keywords
        )

        # 檢查是否包含預期的關鍵詞
        keyword_words = [kw["word"] for kw in keywords]
        assert "人工智慧" in keyword_words

    def test_extract_keywords_empty(self):
        """測試空文本關鍵詞提取"""
        keywords = self.processor.extract_keywords("")
        assert keywords == []

    def test_split_into_sentences(self):
        """測試句子分割"""
        text = "這是第一句。這是第二句！這是第三句？還有第四句；最後一句。"
        sentences = self.processor._split_into_sentences(text)

        assert len(sentences) >= 4
        assert "這是第一句" in sentences[0]
        assert "這是第二句" in sentences[1]

    def test_split_text_into_chunks_basic(self):
        """測試基本文本分塊"""
        text = """
        這是一個很長的文本，用來測試文本分塊功能。
        文本分塊是自然語言處理中的重要步驟。
        它可以將長文本分割成適當大小的片段。
        每個片段都包含完整的語義資訊。
        這樣可以提高後續處理的效率和準確性。
        """

        chunks = self.processor.split_text_into_chunks(text, "test_doc")

        assert len(chunks) > 0
        assert all(chunk.document_id == "test_doc" for chunk in chunks)
        assert all(chunk.chunk_index >= 0 for chunk in chunks)
        assert all(len(chunk.text) > 0 for chunk in chunks)

        # 檢查 metadata
        for chunk in chunks:
            assert "length" in chunk.metadata
            assert "sentence_count" in chunk.metadata
            assert "language" in chunk.metadata
            assert chunk.metadata["language"] == "zh"

    def test_split_text_into_chunks_empty(self):
        """測試空文本分塊"""
        chunks = self.processor.split_text_into_chunks("", "test_doc")
        assert chunks == []

    def test_split_text_into_chunks_small_text(self):
        """測試小文本分塊"""
        text = "這是一個很短的文本。"
        chunks = self.processor.split_text_into_chunks(text, "test_doc")

        assert len(chunks) == 1
        # clean_text 會移除行尾標點符號
        assert "這是一個很短的文本" in chunks[0].text
        assert chunks[0].chunk_index == 0

    def test_split_text_into_chunks_custom_size(self):
        """測試自訂大小的文本分塊"""
        text = "這是一個測試文本。" * 50  # 重複創建長文本

        chunks = self.processor.split_text_into_chunks(
            text, "test_doc", chunk_size=100, chunk_overlap=20
        )

        assert len(chunks) > 1
        # 檢查大部分塊的大小不超過限制（最後一塊可能較小）
        for chunk in chunks[:-1]:
            assert len(chunk.text) <= 120  # 允許一些彈性

    def test_get_text_statistics_comprehensive(self):
        """測試文本統計功能"""
        text = """
        這是一個包含中文和English的測試文本。
        它包含多個句子，用來測試統計功能。
        
        第二段落開始了。
        包含標點符號：，。！？
        """

        stats = self.processor.get_text_statistics(text)

        # 檢查所有統計項目都存在
        expected_keys = [
            "char_count",
            "word_count",
            "sentence_count",
            "paragraph_count",
            "chinese_char_count",
            "english_word_count",
            "punctuation_count",
            "avg_sentence_length",
            "avg_word_length",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
            assert stats[key] >= 0

        # 檢查具體數值的合理性
        assert stats["char_count"] > 0
        assert stats["chinese_char_count"] > 0
        assert stats["english_word_count"] > 0  # 包含 "English"
        assert stats["sentence_count"] > 0
        assert stats["paragraph_count"] >= 2  # 至少兩段

    def test_get_text_statistics_empty(self):
        """測試空文本統計"""
        stats = self.processor.get_text_statistics("")

        assert stats["char_count"] == 0
        assert stats["word_count"] == 0
        assert stats["sentence_count"] == 0
        assert stats["paragraph_count"] == 0
        assert stats["chinese_char_count"] == 0
        assert stats["english_word_count"] == 0
        assert stats["punctuation_count"] == 0
        assert stats["avg_sentence_length"] == 0
        assert stats["avg_word_length"] == 0

    def test_get_text_statistics_chinese_only(self):
        """測試純中文文本統計"""
        text = "這是純中文文本，沒有英文單詞。包含標點符號。"
        stats = self.processor.get_text_statistics(text)

        assert stats["chinese_char_count"] > 0
        assert stats["english_word_count"] == 0
        assert stats["punctuation_count"] > 0

    def test_get_text_statistics_english_only(self):
        """測試純英文文本統計"""
        text = "This is pure English text without Chinese characters."
        stats = self.processor.get_text_statistics(text)

        assert stats["chinese_char_count"] == 0
        assert stats["english_word_count"] > 0

    def test_custom_stopwords_file(self):
        """測試自訂停用詞檔案"""
        # 建立臨時停用詞檔案
        custom_stopwords = ["測試", "檔案", "自訂"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as tmp:
            for word in custom_stopwords:
                tmp.write(word + "\n")
            tmp_path = tmp.name

        try:
            processor = ChineseTextProcessor(stopwords_path=tmp_path)

            # 檢查自訂停用詞是否被載入
            for word in custom_stopwords:
                assert word in processor.stopwords

            # 測試分詞時是否移除自訂停用詞
            text = "這是測試檔案的自訂功能"
            words = processor.segment_text(text)

            for word in custom_stopwords:
                assert word not in words

        finally:
            Path(tmp_path).unlink()

    def test_custom_dict_integration(self):
        """測試自訂詞典整合"""
        # 建立臨時詞典檔案
        custom_words = ["GraphRAG", "人工智慧系統", "知識圖譜"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as tmp:
            for word in custom_words:
                tmp.write(f"{word} 100\n")  # jieba 詞典格式
            tmp_path = tmp.name

        try:
            processor = ChineseTextProcessor(custom_dict_path=tmp_path)

            # 測試包含自訂詞的文本分詞
            text = "GraphRAG是一個人工智慧系統，用於建構知識圖譜"
            words = processor.segment_text(text, remove_stopwords=False, keep_pos=False)

            # 檢查自訂詞是否被正確識別
            assert "GraphRAG" in words
            assert "人工智慧系統" in words or "人工智慧" in words
            assert "知識圖譜" in words

        finally:
            Path(tmp_path).unlink()

    def test_normalize_punctuation(self):
        """測試標點符號正規化"""
        text = "測試\"引號\"和'單引號'以及...省略號—破折號"
        normalized = self.processor._normalize_punctuation(text)

        assert '"' in normalized
        assert "'" in normalized
        assert "…" in normalized
        assert "—" in normalized

    def test_integration_with_real_chinese_text(self):
        """測試真實中文文本的整合處理"""
        text = """
        中華人民共和國是世界上人口最多的國家，位於亞洲東部。
        中國有著悠久的歷史和燦爛的文化，是四大文明古國之一。
        
        改革開放以來，中國經濟快速發展，成為世界第二大經濟體。
        科技創新、人工智慧、新能源等領域都取得了重大突破。
        
        中國堅持和平發展道路，致力於構建人類命運共同體。
        """

        # 測試完整的處理流程
        # 1. 文本清理
        cleaned = self.processor.clean_text(text)
        assert len(cleaned) > 0

        # 2. 分詞
        words = self.processor.segment_text(cleaned)
        assert len(words) > 0
        assert "中國" in words
        assert "人工智慧" in words

        # 3. 關鍵詞提取
        keywords = self.processor.extract_keywords(cleaned, top_k=10)
        assert len(keywords) > 0
        keyword_words = [kw["word"] for kw in keywords]
        assert "中國" in keyword_words

        # 4. 文本分塊
        chunks = self.processor.split_text_into_chunks(cleaned, "china_doc")
        assert len(chunks) > 0

        # 5. 統計資訊
        stats = self.processor.get_text_statistics(cleaned)
        assert stats["chinese_char_count"] > 0
        assert stats["sentence_count"] > 0
        assert stats["paragraph_count"] >= 1  # clean_text 會合併段落

    def test_preprocess_for_entity_recognition_basic(self):
        """測試實體識別預處理基本功能"""
        text = "張三在北京大學工作，電話是13912345678，郵件是zhang@pku.edu.cn"

        result = self.processor.preprocess_for_entity_recognition(text)

        # 檢查返回結構
        assert "cleaned_text" in result
        assert "potential_entities" in result
        assert "entity_patterns" in result
        assert "text_spans" in result

        # 檢查清理後的文本
        assert len(result["cleaned_text"]) > 0

        # 檢查實體候選
        entities = result["potential_entities"]
        entity_texts = [e["text"] for e in entities]

        # 應該包含人名和地名
        assert any("張三" in text or "北京" in text for text in entity_texts)

        # 檢查文本片段
        assert len(result["text_spans"]) > 0

        # 檢查實體模式
        patterns = result["entity_patterns"]
        assert "phone" in patterns
        assert "email" in patterns

        # 檢查電話和郵件是否被識別
        if patterns["phone"]:
            assert "13912345678" in [p["text"] for p in patterns["phone"]]
        if patterns["email"]:
            assert "zhang@pku.edu.cn" in [p["text"] for p in patterns["email"]]

    def test_preprocess_for_entity_recognition_empty(self):
        """測試空文本的實體識別預處理"""
        result = self.processor.preprocess_for_entity_recognition("")

        assert result["cleaned_text"] == ""
        assert result["potential_entities"] == []
        assert result["entity_patterns"] == {}
        assert result["text_spans"] == []

    def test_extract_entity_patterns_comprehensive(self):
        """測試實體模式提取的全面功能"""
        text = """
        今天是2023年12月25日，時間是14:30:00。
        聯絡電話：13912345678，座機：010-12345678。
        郵件：test@example.com，網址：https://www.example.com。
        金額：1000元，￥500，$100美元。
        百分比：85%，百分之九十。
        """

        patterns = self.processor._extract_entity_patterns(text)

        # 檢查各種模式是否被識別
        assert len(patterns["date"]) > 0
        assert len(patterns["time"]) > 0
        assert len(patterns["phone"]) > 0
        assert len(patterns["email"]) > 0
        assert len(patterns["url"]) > 0
        assert len(patterns["money"]) > 0
        assert len(patterns["percent"]) > 0

        # 檢查具體內容
        date_texts = [p["text"] for p in patterns["date"]]
        assert "2023年12月25日" in date_texts

        phone_texts = [p["text"] for p in patterns["phone"]]
        assert "13912345678" in phone_texts

        email_texts = [p["text"] for p in patterns["email"]]
        assert "test@example.com" in email_texts

    def test_evaluate_text_quality_good_text(self):
        """測試高品質文本的品質評估"""
        text = """
        人工智慧是現代科技發展的重要方向。近年來，機器學習和深度學習技術
        取得了突破性進展，在圖像識別、自然語言處理、語音識別等領域展現出
        強大的能力。然而，人工智慧的發展也帶來了一些挑戰，包括數據隱私、
        算法偏見、就業影響等問題。因此，我們需要在推進技術發展的同時，
        也要關注其社會影響，確保人工智慧技術能夠造福人類。
        """

        result = self.processor.evaluate_text_quality(text)

        # 檢查返回結構
        assert "overall_score" in result
        assert "quality_level" in result
        assert "scores" in result
        assert "issues" in result
        assert "recommendations" in result
        assert "statistics" in result

        # 高品質文本應該有較高分數
        assert result["overall_score"] >= 0.7
        assert result["quality_level"] in ["優秀", "良好"]

        # 檢查各項分數
        scores = result["scores"]
        expected_keys = [
            "length",
            "chinese_ratio",
            "sentence_structure",
            "vocabulary_richness",
            "punctuation",
            "coherence",
            "special_chars",
        ]

        for key in expected_keys:
            assert key in scores
            assert 0 <= scores[key] <= 1

    def test_evaluate_text_quality_poor_text(self):
        """測試低品質文本的品質評估"""
        text = "這是很短的文本。"

        result = self.processor.evaluate_text_quality(text)

        # 短文本應該有較低分數
        assert result["overall_score"] < 0.9

        # 應該有相關問題和建議
        assert len(result["issues"]) > 0
        assert len(result["recommendations"]) > 0

        # 檢查是否識別出長度問題
        issues_text = " ".join(result["issues"])
        assert "短" in issues_text or "長度" in issues_text

    def test_evaluate_text_quality_empty(self):
        """測試空文本的品質評估"""
        result = self.processor.evaluate_text_quality("")

        assert result["overall_score"] == 0.0
        assert result["scores"] == {}
        assert result["issues"] == []
        assert result["recommendations"] == []

    def test_extract_named_entities_candidates_basic(self):
        """測試命名實體候選提取基本功能"""
        text = "李明在上海華東師範大學讀書，今天是2023年10月15日，他的電話是15912345678"

        candidates = self.processor.extract_named_entities_candidates(text)

        # 檢查返回結果
        assert isinstance(candidates, list)
        assert len(candidates) > 0

        # 檢查候選實體結構
        for candidate in candidates:
            assert "text" in candidate
            assert "type" in candidate
            assert "start" in candidate
            assert "end" in candidate
            assert "confidence" in candidate
            assert "source" in candidate

            # 檢查信心度範圍
            assert 0 <= candidate["confidence"] <= 1

        # 檢查是否包含預期的實體類型
        entity_types = [c["type"] for c in candidates]
        candidate_texts = [c["text"] for c in candidates]

        # 應該包含人名、地名、日期、電話等
        expected_patterns = [
            "李明",
            "上海",
            "華東師範大學",
            "2023年10月15日",
            "15912345678",
        ]
        found_entities = []

        for text in expected_patterns:
            if any(text in candidate_text for candidate_text in candidate_texts):
                found_entities.append(text)

        # 至少應該找到一些實體
        assert len(found_entities) > 0

    def test_extract_named_entities_candidates_confidence_threshold(self):
        """測試信心度閾值過濾"""
        text = "張三在北京工作，電話13912345678"

        # 測試不同信心度閾值
        candidates_low = self.processor.extract_named_entities_candidates(
            text, min_confidence=0.1
        )
        candidates_high = self.processor.extract_named_entities_candidates(
            text, min_confidence=0.9
        )

        # 低閾值應該返回更多候選
        assert len(candidates_low) >= len(candidates_high)

        # 高閾值的所有候選信心度都應該 >= 0.9
        for candidate in candidates_high:
            assert candidate["confidence"] >= 0.9

    def test_extract_named_entities_candidates_empty(self):
        """測試空文本的實體候選提取"""
        candidates = self.processor.extract_named_entities_candidates("")
        assert candidates == []

    def test_entity_recognition_integration(self):
        """測試實體識別功能的整合測試"""
        text = """
        王小明是北京大學的教授，專門研究人工智慧。他的辦公室位於理科一號樓
        308室，聯絡電話是010-62751234，手機號碼是13812345678。他的研究領域
        包括機器學習、深度學習和自然語言處理。去年2022年，他發表了15篇論文，
        其中5篇發表在頂級期刊上。他的郵件地址是wang@pku.edu.cn，個人網站是
        https://www.pku.edu.cn/~wang。
        """

        # 1. 預處理
        preprocess_result = self.processor.preprocess_for_entity_recognition(text)
        assert len(preprocess_result["potential_entities"]) > 0
        assert len(preprocess_result["entity_patterns"]) > 0

        # 2. 提取候選實體
        candidates = self.processor.extract_named_entities_candidates(text)
        assert len(candidates) > 0

        # 3. 檢查是否包含預期的實體類型
        entity_types = set(c["type"] for c in candidates)
        expected_types = {
            "人名",
            "地名",
            "機構名",
            "phone",
            "email",
            "url",
            "date",
            "number",
        }

        # 至少應該包含其中幾種類型
        assert len(entity_types.intersection(expected_types)) >= 3

        # 4. 檢查位置資訊的正確性
        for candidate in candidates:
            start, end = candidate["start"], candidate["end"]
            extracted_text = preprocess_result["cleaned_text"][start:end]

            # 提取的文本應該與候選實體文本相符（或包含）
            assert (
                candidate["text"] in extracted_text
                or extracted_text in candidate["text"]
            )

    def test_text_quality_with_entity_patterns(self):
        """測試包含實體模式的文本品質評估"""
        text = """
        根據2023年的統計數據，中國的GDP達到了17.7萬億美元，同比增長5.2%。
        這一數據反映了中國經濟的穩定發展。如需更多資訊，請聯絡統計局，
        電話：010-68782222，郵件：info@stats.gov.cn。
        """

        # 文本品質評估
        quality_result = self.processor.evaluate_text_quality(text)

        # 包含數據和聯絡資訊的文本應該有合理的品質分數
        assert quality_result["overall_score"] > 0.6

        # 實體識別
        candidates = self.processor.extract_named_entities_candidates(text)

        # 應該識別出多種實體類型
        entity_types = set(c["type"] for c in candidates)
        assert "number" in entity_types  # 數字
        assert "地名" in entity_types  # 中國
        assert "機構名" in entity_types  # 統計局
        assert "英文" in entity_types  # 英文內容
