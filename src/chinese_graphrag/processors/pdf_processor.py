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
        """清理從 PDF 提取的文字

        Args:
            text: 原始文字

        Returns:
            str: 清理後的文字
        """
        # 移除多餘的空白字符
        text = re.sub(r"\s+", " ", text)

        # 移除多餘的換行符
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # 移除頁首頁尾常見的模式（可根據需要調整）
        text = re.sub(r"第\s*\d+\s*頁", "", text)
        text = re.sub(r"Page\s*\d+", "", text, flags=re.IGNORECASE)

        # 移除多餘的標點符號
        text = re.sub(
            r"[^\w\s\u4e00-\u9fff，。！？；：「」『』（）【】《》〈〉]", "", text
        )

        return text.strip()

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
