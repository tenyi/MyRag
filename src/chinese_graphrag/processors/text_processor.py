"""
純文字檔案處理器

處理 .txt 和 .md 格式的檔案
"""

from pathlib import Path
from typing import Optional

import markdown
from loguru import logger

from ..models.document import Document
from .base import BaseDocumentProcessor
from .exceptions import ContentExtractionError


class TextProcessor(BaseDocumentProcessor):
    """純文字檔案處理器
    
    支援 .txt 格式的純文字檔案
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {".txt"}
    
    def extract_content(self, file_path: str) -> str:
        """從文字檔案中提取內容
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            str: 檔案內容
            
        Raises:
            ContentExtractionError: 內容提取失敗
        """
        try:
            # 檢測編碼
            encoding = self.detect_encoding(file_path)
            
            # 讀取檔案內容
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            
            # 移除多餘的空白字符
            content = content.strip()
            
            if not content:
                raise ContentExtractionError(
                    file_path, "檔案內容為空"
                )
            
            return content
            
        except UnicodeDecodeError as e:
            raise ContentExtractionError(
                file_path, f"編碼解碼失敗: {str(e)}"
            )
        except Exception as e:
            raise ContentExtractionError(
                file_path, f"讀取檔案失敗: {str(e)}"
            )
    
    def process_file(self, file_path: str) -> Document:
        """處理文字檔案
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            Document: 文件物件
        """
        # 驗證檔案
        self.validate_file(file_path)
        
        # 提取內容
        content = self.extract_content(file_path)
        encoding = self.detect_encoding(file_path)
        
        # 建立文件物件
        document = self.create_document(
            file_path=file_path,
            content=content,
            encoding=encoding
        )
        
        logger.debug(f"成功處理文字檔案: {file_path}")
        return document


class MarkdownProcessor(BaseDocumentProcessor):
    """Markdown 檔案處理器
    
    支援 .md 和 .markdown 格式的檔案
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {".md", ".markdown"}
        
        # 配置 Markdown 解析器
        self.md_parser = markdown.Markdown(
            extensions=[
                'extra',      # 支援表格、程式碼區塊等
                'codehilite', # 程式碼高亮
                'toc',        # 目錄
            ]
        )
    
    def extract_content(self, file_path: str) -> str:
        """從 Markdown 檔案中提取內容
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            str: 轉換為純文字的內容
            
        Raises:
            ContentExtractionError: 內容提取失敗
        """
        try:
            # 檢測編碼
            encoding = self.detect_encoding(file_path)
            
            # 讀取 Markdown 內容
            with open(file_path, "r", encoding=encoding) as f:
                markdown_content = f.read()
            
            if not markdown_content.strip():
                raise ContentExtractionError(
                    file_path, "檔案內容為空"
                )
            
            # 將 Markdown 轉換為 HTML，然後提取純文字
            html_content = self.md_parser.convert(markdown_content)
            
            # 簡單的 HTML 標籤移除（可以考慮使用 BeautifulSoup 做更完整的處理）
            import re
            text_content = re.sub(r'<[^>]+>', '', html_content)
            
            # 清理多餘的空白字符
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
            text_content = text_content.strip()
            
            if not text_content:
                raise ContentExtractionError(
                    file_path, "轉換後的內容為空"
                )
            
            return text_content
            
        except Exception as e:
            if isinstance(e, ContentExtractionError):
                raise
            raise ContentExtractionError(
                file_path, f"Markdown 處理失敗: {str(e)}"
            )
    
    def extract_raw_content(self, file_path: str) -> str:
        """提取原始 Markdown 內容（不轉換為純文字）
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            str: 原始 Markdown 內容
        """
        try:
            encoding = self.detect_encoding(file_path)
            
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            
            return content.strip()
            
        except Exception as e:
            raise ContentExtractionError(
                file_path, f"讀取 Markdown 檔案失敗: {str(e)}"
            )
    
    def process_file(self, file_path: str) -> Document:
        """處理 Markdown 檔案
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            Document: 文件物件
        """
        # 驗證檔案
        self.validate_file(file_path)
        
        # 提取內容（轉換為純文字）
        content = self.extract_content(file_path)
        encoding = self.detect_encoding(file_path)
        
        # 建立文件物件
        document = self.create_document(
            file_path=file_path,
            content=content,
            encoding=encoding
        )
        
        logger.debug(f"成功處理 Markdown 檔案: {file_path}")
        return document