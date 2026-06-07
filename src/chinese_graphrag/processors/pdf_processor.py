"""
PDF 檔案處理器

使用 pypdf 處理 PDF 檔案
"""

import re
from pathlib import Path
from typing import Optional

from loguru import logger
from pypdf import PdfReader

from ..models.document import Document
from .base import BaseDocumentProcessor
from .exceptions import ContentExtractionError, FileCorruptionError


class PDFProcessor(BaseDocumentProcessor):
    """PDF 檔案處理器

    支援 .pdf 格式的檔案
    """

    def __init__(self):
        super().__init__()
        self.supported_extensions = {".pdf"}

    def extract_content(self, file_path: str) -> str:
        """從 PDF 檔案中提取文字內容

        Args:
            file_path: 檔案路徑

        Returns:
            str: 提取的文字內容

        Raises:
            ContentExtractionError: 內容提取失敗
            FileCorruptionError: PDF 檔案損壞
        """
        try:
            # 開啟 PDF 檔案
            reader = PdfReader(file_path)

            # 檢查 PDF 是否有頁面
            if len(reader.pages) == 0:
                raise ContentExtractionError(file_path, "PDF 檔案沒有頁面")

            # 提取所有頁面的文字
            text_content = []

            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(page_text)

                    logger.debug(f"成功提取第 {page_num} 頁內容")

                except Exception as e:
                    logger.warning(f"提取第 {page_num} 頁失敗: {str(e)}")
                    continue

            if not text_content:
                raise ContentExtractionError(file_path, "無法從 PDF 中提取任何文字內容")

            # 合併所有頁面的內容
            full_content = "\n\n".join(text_content)

            # 清理文字內容
            cleaned_content = self._clean_text(full_content)

            if not cleaned_content.strip():
                raise ContentExtractionError(file_path, "清理後的內容為空")

            return cleaned_content

        except Exception as e:
            if isinstance(e, (ContentExtractionError, FileCorruptionError)):
                raise

            # 檢查是否為 PDF 檔案損壞
            if "PDF" in str(e) or "corrupt" in str(e).lower():
                raise FileCorruptionError(file_path, f"PDF 檔案可能損壞: {str(e)}")

            raise ContentExtractionError(file_path, f"PDF 處理失敗: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """清理從 PDF 提取的文字（加強中文處理）

        Args:
            text: 原始文字

        Returns:
            str: 清理後的文字
        """
        # 移除多餘的空白字符
        text = re.sub(r"\s+", " ", text)

        # 修復 PDF 提取時常見的中文字符問題
        text = self._fix_chinese_character_issues(text)

        # 移除多餘的換行符和空格
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)  # 多個空格合併為一個

        # 移除頁首頁尾模式
        text = self._remove_header_footer_patterns(text)

        # 修復中文段落斷行
        text = self._fix_chinese_line_breaks(text)

        # 移除無關的特殊字符（保留中文標點符號）
        text = re.sub(
            r"[^\w\s\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u300c\u300d\u300e\u300f\uff08\uff09\u3010\u3011\u300a\u300b\u3008\u3009\u2014\u2013\u2026\uff0d\u0028\u0029\u002c\u002e\u0021\u003f\u003b\u003a\u0022\u0027]",
            "", text
        )

        return text.strip()

    def _fix_chinese_character_issues(self, text: str) -> str:
        """修復 PDF 提取時的中文字符問題
        
        Args:
            text: 原始文字
        
        Returns:
            str: 修復後的文字
        """
        # 移除異常的空格插入（PDF 常見問題）
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        
        # 修復中文標點符號前後的空格
        text = re.sub(r'\s+([\uff0c\u3002\uff01\uff1f\uff1b\uff1a])', r'\1', text)
        text = re.sub(r'([\u300c\u300e\uff08\u3010\u300a\u3008])\s+', r'\1', text)
        
        # 修復數字和中文之間的異常空格
        text = re.sub(r'(\d)\s+([\u4e00-\u9fff])', r'\1\2', text)
        text = re.sub(r'([\u4e00-\u9fff])\s+(\d)', r'\1\2', text)
        
        # 修復英文和中文之間的空格（保留必要的空格）
        text = re.sub(r'([a-zA-Z])\s{2,}([\u4e00-\u9fff])', r'\1 \2', text)
        text = re.sub(r'([\u4e00-\u9fff])\s{2,}([a-zA-Z])', r'\1 \2', text)
        
        return text

    def _remove_header_footer_patterns(self, text: str) -> str:
        """移除頁首頁尾模式
        
        Args:
            text: 原始文字
        
        Returns:
            str: 清理後的文字
        """
        # 移除各種頁碼模式
        patterns = [
            r'第\s*\d+\s*頁',  # 中文頁碼
            r'Page\s*\d+',  # 英文頁碼
            r'\d+\s*/\s*\d+',  # 數字頁碼
            r'-\s*\d+\s*-',  # 帶橫線的頁碼
            r'\[\s*\d+\s*\]',  # 方括號頁碼
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 移除日期時間模式
        text = re.sub(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?', '', text)
        text = re.sub(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', '', text)
        
        # 移除常見的版權資訊
        copyright_patterns = [
            r'[Cc]opyright\s*\u00a9?\s*\d{4}.*',
            r'©\s*\d{4}.*',
            r'版權所有.*',
            r'著作權.*',
            r'[Cc]onfidential.*',
            r'機密.*'
        ]
        
        for pattern in copyright_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text

    def _fix_chinese_line_breaks(self, text: str) -> str:
        """修復中文段落的強行斷行
        
        Args:
            text: 原始文字
        
        Returns:
            str: 修復後的文字
        """
        # 修復中文字符之間的異常換行
        # 如果一行末尾是中文字符，且下一行開頭也是中文字符，則合併
        text = re.sub(r'([\u4e00-\u9fff])\n([\u4e00-\u9fff])', r'\1\2', text)
        
        # 修復段落結尾標點符號後的換行
        text = re.sub(r'([\uff0c\u3002\uff01\uff1f\uff1b\uff1a])\n([\u4e00-\u9fff])', r'\1\2', text)
        
        return text

    def get_pdf_metadata(self, file_path: str) -> dict:
        """取得 PDF 檔案的元資料

        Args:
            file_path: 檔案路徑

        Returns:
            dict: PDF 元資料
        """
        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata

            if metadata:
                return {
                    "title": metadata.get("/Title", ""),
                    "author": metadata.get("/Author", ""),
                    "subject": metadata.get("/Subject", ""),
                    "creator": metadata.get("/Creator", ""),
                    "producer": metadata.get("/Producer", ""),
                    "creation_date": metadata.get("/CreationDate", ""),
                    "modification_date": metadata.get("/ModDate", ""),
                    "page_count": len(reader.pages),
                }

            return {"page_count": len(reader.pages)}

        except Exception as e:
            logger.warning(f"無法讀取 PDF 元資料: {str(e)}")
            return {}

    def process_file(self, file_path: str) -> Document:
        """處理 PDF 檔案

        Args:
            file_path: 檔案路徑

        Returns:
            Document: 文件物件
        """
        # 驗證檔案
        self.validate_file(file_path)

        # 提取內容
        content = self.extract_content(file_path)

        # 取得 PDF 元資料
        pdf_metadata = self.get_pdf_metadata(file_path)

        # 使用 PDF 標題作為文件標題（如果有的話）
        title = pdf_metadata.get("title", "").strip()
        if not title:
            title = Path(file_path).stem

        # 建立文件物件
        document = self.create_document(
            file_path=file_path,
            content=content,
            title=title,
            encoding="utf-8",  # PDF 內容已轉換為 Unicode
        )

        logger.debug(f"成功處理 PDF 檔案: {file_path}")
        return document
