"""
中文文本預處理器

專門處理中文文本的分詞、清理和分塊功能
"""

import math
import re
from typing import Dict, List, Optional, Set, Tuple

import jieba
import jieba.posseg as pseg
from loguru import logger

from ..models.text_unit import TextUnit


class ChineseTextProcessor:
    """中文文本預處理器

    提供中文文本的分詞、停用詞過濾、文本清理和分塊功能
    """

    def __init__(
        self,
        custom_dict_path: Optional[str] = None,
        stopwords_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """初始化中文文本處理器

        Args:
            custom_dict_path: 自訂詞典路徑
            stopwords_path: 停用詞檔案路徑
            chunk_size: 文本塊大小（字符數）
            chunk_overlap: 文本塊重疊大小（字符數）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化 jieba
        self._init_jieba(custom_dict_path)

        # 載入停用詞
        self.stopwords = self._load_stopwords(stopwords_path)

        # 中文標點符號
        self.chinese_punctuation = {
            "，",
            "。",
            "！",
            "？",
            "；",
            "：",
            "「",
            "」",
            "『",
            "』",
            "（",
            "）",
            "【",
            "】",
            "《",
            "》",
            "〈",
            "〉",
            "、",
            "…",
            "—",
            "－",
            "·",
            "‧",
            "〔",
            "〕",
            "〖",
            "〗",
            "〘",
            "〙",
            "〚",
            "〛",
            "〜",
            "〝",
            "〞",
            "〟",
            "〰",
        }

        # 需要保留的詞性標籤（名詞、動詞、形容詞等實詞）
        self.keep_pos_tags = {
            "n",
            "nr",
            "ns",
            "nt",
            "nw",
            "nz",  # 名詞類
            "v",
            "vd",
            "vn",
            "vshi",
            "vyou",
            "vf",
            "vx",
            "vi",
            "vl",
            "vg",  # 動詞類
            "a",
            "ad",
            "an",
            "ag",
            "al",  # 形容詞類
            "r",
            "rr",
            "rz",
            "rzt",
            "rzs",
            "rzv",
            "ry",
            "ryt",
            "rys",
            "ryv",  # 代詞類
            "i",
            "j",
            "l",  # 成語、簡稱、習用語
            "eng",  # 英文
        }

        logger.info("中文文本處理器初始化完成")

    def _init_jieba(self, custom_dict_path: Optional[str] = None) -> None:
        """初始化 jieba 分詞器

        Args:
            custom_dict_path: 自訂詞典路徑
        """
        # 設定 jieba 為精確模式
        jieba.setLogLevel(20)  # 只顯示 INFO 以上的日誌

        # 載入自訂詞典
        if custom_dict_path:
            try:
                jieba.load_userdict(custom_dict_path)
                logger.info(f"載入自訂詞典: {custom_dict_path}")
            except Exception as e:
                logger.warning(f"載入自訂詞典失敗: {e}")

    def _load_stopwords(self, stopwords_path: Optional[str] = None) -> Set[str]:
        """載入停用詞列表

        Args:
            stopwords_path: 停用詞檔案路徑

        Returns:
            Set[str]: 停用詞集合
        """
        stopwords = set()

        # 預設中文停用詞
        default_stopwords = {
            "的",
            "了",
            "在",
            "是",
            "我",
            "有",
            "和",
            "就",
            "不",
            "人",
            "都",
            "一",
            "一個",
            "上",
            "也",
            "很",
            "到",
            "說",
            "要",
            "去",
            "你",
            "會",
            "著",
            "沒有",
            "看",
            "好",
            "自己",
            "這",
            "那",
            "他",
            "她",
            "它",
            "們",
            "這個",
            "那個",
            "這些",
            "那些",
            "什麼",
            "怎麼",
            "為什麼",
            "哪裡",
            "哪個",
            "誰",
            "什麼時候",
            "怎樣",
            "多少",
            "啊",
            "呢",
            "吧",
            "嗎",
            "呀",
            "哦",
            "哎",
            "唉",
            "嗯",
            "嘿",
            "但是",
            "可是",
            "然而",
            "不過",
            "雖然",
            "儘管",
            "如果",
            "假如",
            "因為",
            "所以",
            "由於",
            "因此",
            "而且",
            "並且",
            "或者",
            "還是",
            "以及",
            "以及",
            "等等",
            "之類",
            "比如",
            "例如",
            "譬如",
            "諸如",
        }
        stopwords.update(default_stopwords)

        # 從檔案載入停用詞
        if stopwords_path:
            try:
                with open(stopwords_path, "r", encoding="utf-8") as f:
                    file_stopwords = {line.strip() for line in f if line.strip()}
                    stopwords.update(file_stopwords)
                logger.info(
                    f"載入停用詞檔案: {stopwords_path}, 共 {len(file_stopwords)} 個詞"
                )
            except Exception as e:
                logger.warning(f"載入停用詞檔案失敗: {e}")

        logger.info(f"停用詞總數: {len(stopwords)}")
        return stopwords

    def clean_text(self, text: str) -> str:
        """清理文本

        Args:
            text: 原始文本

        Returns:
            str: 清理後的文本
        """
        if not text:
            return ""

        # 移除多餘的空白字符
        text = re.sub(r"\s+", " ", text)

        # 移除特殊字符（保留中文、英文、數字和基本標點）
        text = re.sub(
            r"[^\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
            r"a-zA-Z0-9\s，。！？；：「」『』（）【】《》〈〉、…—－·‧]",
            "",
            text,
        )

        # 正規化標點符號
        text = self._normalize_punctuation(text)

        # 移除重複的標點符號
        text = re.sub(r"([，。！？；：])\1+", r"\1", text)

        # 移除行首行尾的空白和標點
        text = text.strip(" \t\n\r，。！？；：")

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """正規化標點符號

        Args:
            text: 輸入文本

        Returns:
            str: 正規化後的文本
        """
        # 統一引號
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)

        # 統一破折號
        text = re.sub(r"[—－]", "—", text)

        # 統一省略號
        text = re.sub(r"\.{3,}", "…", text)

        return text

    def segment_text(
        self, text: str, remove_stopwords: bool = True, keep_pos: bool = True
    ) -> List[str]:
        """中文分詞

        Args:
            text: 輸入文本
            remove_stopwords: 是否移除停用詞
            keep_pos: 是否只保留特定詞性的詞

        Returns:
            List[str]: 分詞結果
        """
        if not text:
            return []

        # 清理文本
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []

        # 分詞
        if keep_pos:
            # 使用詞性標註分詞
            words = []
            for word, pos in pseg.cut(cleaned_text):
                # 過濾標點符號
                if word.strip() and word not in self.chinese_punctuation:
                    # 只保留特定詞性的詞
                    if any(pos.startswith(tag) for tag in self.keep_pos_tags):
                        words.append(word.strip())
        else:
            # 普通分詞
            words = [
                word.strip()
                for word in jieba.cut(cleaned_text)
                if word.strip() and word not in self.chinese_punctuation
            ]

        # 移除停用詞
        if remove_stopwords:
            words = [word for word in words if word not in self.stopwords]

        # 過濾長度過短的詞（單字符且非中文字符）
        words = [
            word
            for word in words
            if len(word) > 1 or re.match(r"[\u4e00-\u9fff]", word)
        ]

        return words

    def extract_keywords(
        self, text: str, top_k: int = 20, min_word_len: int = 2
    ) -> List[Dict[str, any]]:
        """提取關鍵詞

        Args:
            text: 輸入文本
            top_k: 返回前 k 個關鍵詞
            min_word_len: 最小詞長度

        Returns:
            List[Dict[str, any]]: 關鍵詞列表，包含詞和權重
        """
        # 分詞
        words = self.segment_text(text, remove_stopwords=True, keep_pos=True)

        # 過濾詞長度
        words = [word for word in words if len(word) >= min_word_len]

        # 計算詞頻
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # 按頻率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # 返回前 k 個關鍵詞
        keywords = []
        for word, freq in sorted_words[:top_k]:
            keywords.append(
                {
                    "word": word,
                    "frequency": freq,
                    "weight": freq / len(words) if words else 0,
                }
            )

        return keywords

    def split_text_into_chunks(
        self,
        text: str,
        document_id: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[TextUnit]:
        """將文本分割成塊

        Args:
            text: 輸入文本
            document_id: 文件 ID
            chunk_size: 塊大小（字符數）
            chunk_overlap: 重疊大小（字符數）

        Returns:
            List[TextUnit]: 文本塊列表
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap

        if not text:
            return []

        # 清理文本
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []

        # 按句子分割
        sentences = self._split_into_sentences(cleaned_text)
        if not sentences:
            return []

        # 組合句子成塊
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # 如果當前塊加上新句子超過大小限制
            if current_length + sentence_length > chunk_size and current_chunk:
                # 建立當前塊
                chunk_text = current_chunk.strip()
                if chunk_text:
                    text_unit = TextUnit(
                        text=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        metadata={
                            "length": len(chunk_text),
                            "sentence_count": len(
                                self._split_into_sentences(chunk_text)
                            ),
                            "language": "zh",
                        },
                    )
                    chunks.append(text_unit)
                    chunk_index += 1

                # 處理重疊
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    # 保留最後部分作為重疊
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                # 添加句子到當前塊
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length

        # 處理最後一個塊
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            text_unit = TextUnit(
                text=chunk_text,
                document_id=document_id,
                chunk_index=chunk_index,
                metadata={
                    "length": len(chunk_text),
                    "sentence_count": len(self._split_into_sentences(chunk_text)),
                    "language": "zh",
                },
            )
            chunks.append(text_unit)

        logger.info(f"文本分塊完成: {len(chunks)} 個塊")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """將文本分割成句子

        Args:
            text: 輸入文本

        Returns:
            List[str]: 句子列表
        """
        if not text:
            return []

        # 中文句子分割正則表達式
        sentence_pattern = r"[。！？；\n]+"

        # 分割句子
        sentences = re.split(sentence_pattern, text)

        # 清理和過濾句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) >= 2:  # 過濾太短的句子
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def get_text_statistics(self, text: str) -> Dict[str, any]:
        """取得文本統計資訊

        Args:
            text: 輸入文本

        Returns:
            Dict[str, any]: 統計資訊
        """
        if not text:
            return {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "chinese_char_count": 0,
                "english_word_count": 0,
                "punctuation_count": 0,
                "avg_sentence_length": 0,
                "avg_word_length": 0,
            }

        # 基本統計
        char_count = len(text)

        # 中文字符統計
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_char_count = len(chinese_chars)

        # 英文單詞統計
        english_words = re.findall(r"[a-zA-Z]+", text)
        english_word_count = len(english_words)

        # 標點符號統計
        punctuation_count = len([c for c in text if c in self.chinese_punctuation])

        # 分詞統計
        words = self.segment_text(text, remove_stopwords=False, keep_pos=False)
        word_count = len(words)

        # 句子統計
        sentences = self._split_into_sentences(text)
        sentence_count = len(sentences)

        # 段落統計
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        paragraph_count = len(paragraphs)

        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "chinese_char_count": chinese_char_count,
            "english_word_count": english_word_count,
            "punctuation_count": punctuation_count,
            "avg_sentence_length": (
                char_count / sentence_count if sentence_count > 0 else 0
            ),
            "avg_word_length": chinese_char_count / word_count if word_count > 0 else 0,
        }

    def preprocess_for_entity_recognition(
        self, text: str, extract_patterns: bool = True
    ) -> Dict[str, any]:
        """為實體識別進行文本預處理

        Args:
            text: 輸入文本
            extract_patterns: 是否提取預定義模式（如時間、地點等）

        Returns:
            Dict[str, any]: 預處理結果，包含清理後的文本和潛在實體候選
        """
        if not text:
            return {
                "cleaned_text": "",
                "potential_entities": [],
                "entity_patterns": {},
                "text_spans": [],
            }

        # 文本清理
        cleaned_text = self.clean_text(text)

        # 分詞並保留位置資訊
        words_with_pos = []
        start_pos = 0

        for word, pos in pseg.cut(cleaned_text):
            if word.strip() and word not in self.chinese_punctuation:
                end_pos = start_pos + len(word)
                words_with_pos.append(
                    {
                        "word": word.strip(),
                        "pos": pos,
                        "start": start_pos,
                        "end": end_pos,
                    }
                )
                start_pos = end_pos
            else:
                start_pos += len(word)

        # 識別潛在實體候選（基於詞性）
        potential_entities = []
        entity_pos_tags = {
            "nr": "人名",  # 人名
            "ns": "地名",  # 地名
            "nt": "機構名",  # 機構名
            "nz": "其他專名",  # 其他專名
            "nw": "作品名",  # 作品名
            "eng": "英文",  # 英文
        }

        for word_info in words_with_pos:
            pos = word_info["pos"]
            word = word_info["word"]

            # 根據詞性識別潛在實體
            for tag, entity_type in entity_pos_tags.items():
                if pos.startswith(tag):
                    potential_entities.append(
                        {
                            "text": word,
                            "type": entity_type,
                            "start": word_info["start"],
                            "end": word_info["end"],
                            "confidence": 0.8,
                            "source": "pos_tagging",
                        }
                    )
                    break

        # 提取預定義模式
        entity_patterns = {}
        if extract_patterns:
            entity_patterns = self._extract_entity_patterns(cleaned_text)

        return {
            "cleaned_text": cleaned_text,
            "potential_entities": potential_entities,
            "entity_patterns": entity_patterns,
            "text_spans": words_with_pos,
        }

    def _extract_entity_patterns(self, text: str) -> Dict[str, List[Dict[str, any]]]:
        """提取預定義實體模式

        Args:
            text: 輸入文本

        Returns:
            Dict[str, List[Dict[str, any]]]: 按類型分組的實體模式
        """
        patterns = {
            "date": [],  # 日期
            "time": [],  # 時間
            "number": [],  # 數字
            "phone": [],  # 電話號碼
            "email": [],  # 電子郵件
            "url": [],  # 網址
            "money": [],  # 金額
            "percent": [],  # 百分比
        }

        # 日期模式
        date_patterns = [
            r"\d{4}年\d{1,2}月\d{1,2}日",
            r"\d{4}-\d{1,2}-\d{1,2}",
            r"\d{4}/\d{1,2}/\d{1,2}",
            r"\d{1,2}月\d{1,2}日",
        ]

        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                patterns["date"].append(
                    {
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                )

        # 時間模式
        time_patterns = [
            r"\d{1,2}:\d{2}:\d{2}",
            r"\d{1,2}:\d{2}",
            r"\d{1,2}點\d{1,2}分",
            r"上午|下午|中午|凌晨",
        ]

        for pattern in time_patterns:
            for match in re.finditer(pattern, text):
                patterns["time"].append(
                    {
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                )

        # 數字模式
        number_patterns = [
            r"\d+\.?\d*",
            r"[一二三四五六七八九十百千萬億]+",
        ]

        for pattern in number_patterns:
            for match in re.finditer(pattern, text):
                patterns["number"].append(
                    {
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                )

        # 電話號碼模式
        phone_patterns = [
            r"1[3-9]\d{9}",
            r"\d{3,4}-\d{7,8}",
            r"\(\d{3,4}\)\d{7,8}",
        ]

        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                patterns["phone"].append(
                    {
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                )

        # 電子郵件模式
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        for match in re.finditer(email_pattern, text):
            patterns["email"].append(
                {
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "pattern": email_pattern,
                }
            )

        # 網址模式
        url_patterns = [
            r"https?://[^\s]+",
            r"www\.[^\s]+",
        ]

        for pattern in url_patterns:
            for match in re.finditer(pattern, text):
                patterns["url"].append(
                    {
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                )

        # 金額模式
        money_patterns = [
            r"\d+\.?\d*元",
            r"￥\d+\.?\d*",
            r"\$\d+\.?\d*",
            r"\d+\.?\d*萬元",
            r"\d+\.?\d*億元",
        ]

        for pattern in money_patterns:
            for match in re.finditer(pattern, text):
                patterns["money"].append(
                    {
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                )

        # 百分比模式
        percent_patterns = [
            r"\d+\.?\d*%",
            r"百分之\d+\.?\d*",
        ]

        for pattern in percent_patterns:
            for match in re.finditer(pattern, text):
                patterns["percent"].append(
                    {
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                )

        return patterns

    def evaluate_text_quality(self, text: str) -> Dict[str, any]:
        """評估中文文本品質

        Args:
            text: 輸入文本

        Returns:
            Dict[str, any]: 文本品質評估結果
        """
        if not text:
            return {
                "overall_score": 0.0,
                "scores": {},
                "issues": [],
                "recommendations": [],
            }

        # 取得文本統計
        stats = self.get_text_statistics(text)

        # 各項評估分數
        scores = {}
        issues = []
        recommendations = []

        # 1. 長度評估 (0-1分)
        char_count = stats["char_count"]
        if char_count < 50:
            scores["length"] = 0.3
            issues.append("文本過短，可能影響理解")
            recommendations.append("建議增加文本內容，至少50字符")
        elif char_count > 10000:
            scores["length"] = 0.8
            issues.append("文本較長，建議分段處理")
            recommendations.append("考慮將長文本分割成較小的段落")
        else:
            scores["length"] = 1.0

        # 2. 中文比例評估 (0-1分)
        chinese_ratio = (
            stats["chinese_char_count"] / char_count if char_count > 0 else 0
        )
        if chinese_ratio < 0.3:
            scores["chinese_ratio"] = 0.5
            issues.append("中文字符比例較低")
            recommendations.append("確認文本主要為中文內容")
        else:
            scores["chinese_ratio"] = min(1.0, chinese_ratio * 1.2)

        # 3. 句子結構評估 (0-1分)
        sentence_count = stats["sentence_count"]
        avg_sentence_length = stats["avg_sentence_length"]

        if sentence_count == 0:
            scores["sentence_structure"] = 0.0
            issues.append("缺少句子結構")
            recommendations.append("添加適當的句子分隔符號")
        elif avg_sentence_length < 5:
            scores["sentence_structure"] = 0.4
            issues.append("句子平均長度過短")
            recommendations.append("檢查句子完整性")
        elif avg_sentence_length > 100:
            scores["sentence_structure"] = 0.6
            issues.append("句子平均長度過長")
            recommendations.append("考慮將長句分解為短句")
        else:
            scores["sentence_structure"] = 1.0

        # 4. 詞彙豐富度評估 (0-1分)
        words = self.segment_text(text, remove_stopwords=True)
        unique_words = set(words)

        if len(words) == 0:
            scores["vocabulary_richness"] = 0.0
            issues.append("缺少有意義的詞彙")
        else:
            vocabulary_diversity = len(unique_words) / len(words)
            scores["vocabulary_richness"] = min(1.0, vocabulary_diversity * 2)

            if vocabulary_diversity < 0.3:
                issues.append("詞彙重複度較高")
                recommendations.append("增加詞彙的多樣性")

        # 5. 標點符號使用評估 (0-1分)
        punctuation_ratio = (
            stats["punctuation_count"] / char_count if char_count > 0 else 0
        )

        if punctuation_ratio < 0.02:
            scores["punctuation"] = 0.3
            issues.append("標點符號使用過少")
            recommendations.append("添加適當的標點符號")
        elif punctuation_ratio > 0.15:
            scores["punctuation"] = 0.6
            issues.append("標點符號使用過多")
            recommendations.append("檢查標點符號使用的合理性")
        else:
            scores["punctuation"] = 1.0

        # 6. 文本連貫性評估 (0-1分)
        # 簡單的連貫性檢查：檢查連接詞的使用
        coherence_words = [
            "但是",
            "然而",
            "因此",
            "所以",
            "而且",
            "並且",
            "另外",
            "此外",
            "首先",
            "其次",
            "最後",
            "總之",
        ]
        coherence_count = sum(1 for word in coherence_words if word in text)

        if sentence_count > 5:
            expected_coherence = sentence_count * 0.1  # 預期每10句話有一個連接詞
            coherence_ratio = (
                coherence_count / expected_coherence if expected_coherence > 0 else 0
            )
            scores["coherence"] = min(1.0, coherence_ratio)

            if coherence_ratio < 0.5:
                issues.append("文本連貫性可能不足")
                recommendations.append("適當使用連接詞增強文本連貫性")
        else:
            scores["coherence"] = 0.8  # 短文本不強制要求連接詞

        # 7. 特殊字符檢查 (0-1分)
        special_char_pattern = r"[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z0-9\s，。！？；：「」『』（）【】《》〈〉、…—－·‧]"
        special_chars = re.findall(special_char_pattern, text)

        if len(special_chars) > char_count * 0.05:  # 特殊字符超過5%
            scores["special_chars"] = 0.5
            issues.append("包含過多特殊字符")
            recommendations.append("清理或說明特殊字符的用途")
        else:
            scores["special_chars"] = 1.0

        # 計算總分（加權平均）
        weights = {
            "length": 0.15,
            "chinese_ratio": 0.15,
            "sentence_structure": 0.20,
            "vocabulary_richness": 0.15,
            "punctuation": 0.10,
            "coherence": 0.15,
            "special_chars": 0.10,
        }

        overall_score = sum(scores[key] * weights[key] for key in scores.keys())

        # 品質等級
        if overall_score >= 0.9:
            quality_level = "優秀"
        elif overall_score >= 0.7:
            quality_level = "良好"
        elif overall_score >= 0.5:
            quality_level = "一般"
        else:
            quality_level = "需要改進"

        return {
            "overall_score": round(overall_score, 3),
            "quality_level": quality_level,
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "issues": issues,
            "recommendations": recommendations,
            "statistics": stats,
        }

    def extract_named_entities_candidates(
        self, text: str, min_confidence: float = 0.6
    ) -> List[Dict[str, any]]:
        """提取命名實體候選

        Args:
            text: 輸入文本
            min_confidence: 最小信心度閾值

        Returns:
            List[Dict[str, any]]: 命名實體候選列表
        """
        # 獲取預處理結果
        preprocess_result = self.preprocess_for_entity_recognition(text)

        candidates = []

        # 從詞性標註結果中提取
        for entity in preprocess_result["potential_entities"]:
            if entity["confidence"] >= min_confidence:
                candidates.append(entity)

        # 從模式匹配結果中提取
        for pattern_type, entities in preprocess_result["entity_patterns"].items():
            for entity in entities:
                candidates.append(
                    {
                        "text": entity["text"],
                        "type": pattern_type,
                        "start": entity["start"],
                        "end": entity["end"],
                        "confidence": 0.9,  # 模式匹配通常有較高信心度
                        "source": "pattern_matching",
                    }
                )

        # 去重和排序
        seen = set()
        unique_candidates = []

        for candidate in sorted(
            candidates, key=lambda x: x["confidence"], reverse=True
        ):
            # 用文本內容和位置來去重
            key = (candidate["text"], candidate["start"], candidate["end"])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        return unique_candidates
