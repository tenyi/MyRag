"""文件和文本處理模組

提供各種檔案格式的處理器和管理功能
"""

from .base import BaseDocumentProcessor, DocumentProcessorManager
from .chinese_text_processor import ChineseTextProcessor
from .docx_processor import DocxProcessor
from .exceptions import (
    ContentExtractionError,
    DocumentProcessingError,
    EncodingDetectionError,
    FileCorruptionError,
    FileNotFoundError,
    UnsupportedFileFormatError,
)
from .pdf_processor import PDFProcessor
from .text_processor import MarkdownProcessor, TextProcessor

__all__ = [
    # 基礎類別
    "BaseDocumentProcessor",
    "DocumentProcessorManager",
    # 具體處理器
    "TextProcessor",
    "MarkdownProcessor",
    "PDFProcessor",
    "DocxProcessor",
    "ChineseTextProcessor",
    # 例外類別
    "DocumentProcessingError",
    "UnsupportedFileFormatError",
    "FileCorruptionError",
    "FileNotFoundError",
    "EncodingDetectionError",
    "ContentExtractionError",
]


def create_default_processor_manager() -> DocumentProcessorManager:
    """建立預設的文件處理器管理器
    
    註冊所有可用的文件處理器
    
    Returns:
        DocumentProcessorManager: 配置好的處理器管理器
    """
    manager = DocumentProcessorManager()
    
    # 註冊各種處理器
    manager.register_processor("text", TextProcessor())
    manager.register_processor("markdown", MarkdownProcessor())
    manager.register_processor("pdf", PDFProcessor())
    manager.register_processor("docx", DocxProcessor())
    
    return manager