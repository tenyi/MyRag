"""
索引用文件處理器

為索引引擎提供統一的文件處理介面
"""

import logging
from pathlib import Path
from typing import List

from chinese_graphrag.config import GraphRAGConfig
from chinese_graphrag.models import Document
# from chinese_graphrag.processors import DocumentProcessorManager, create_default_processor_manager
# from chinese_graphrag.processors.chinese_text_processor import ChineseTextProcessor

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    索引用文件處理器
    
    整合各種文件處理器，為索引引擎提供統一介面
    """

    def __init__(self, config: GraphRAGConfig):
        """
        初始化文件處理器
        
        Args:
            config: GraphRAG 配置
        """
        self.config = config
        
        # 建立處理器管理器（暫時簡化）
        self.processor_manager = None
        logger.info("文件處理器管理器已暫時禁用")
        
        # 建立中文文本處理器（暫時禁用以避免初始化問題）
        self.chinese_processor = None
        logger.info("中文文本處理器已暫時禁用")

    def process_document(self, file_path: Path) -> Document:
        """
        處理單一文件
        
        Args:
            file_path: 文件路徑
            
        Returns:
            Document: 處理後的文件物件
        """
        logger.info(f"處理文件: {file_path}")
        
        try:
            # 使用處理器管理器處理文件（暫時簡化）
            # document = self.processor_manager.process_file(str(file_path))
            # 暫時建立一個簡單的文件物件
            import uuid
            from datetime import datetime
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            document = Document(
                id=str(uuid.uuid4()),
                title=file_path.name,
                content=content,
                file_path=file_path,
                created_at=datetime.now(),
                metadata={}
            )
            
            # 使用中文處理器進行文本預處理（暫時跳過）
            if document.content:
                # processed_content = self.chinese_processor.preprocess_text(document.content)
                # document.content = processed_content
                pass
            
            # 設定語言標記
            document.metadata["language"] = "zh"
            document.metadata["processed_by"] = "chinese_graphrag"
            
            logger.info(f"成功處理文件: {file_path}, 內容長度: {len(document.content)}")
            return document
            
        except Exception as e:
            logger.error(f"處理文件失敗 {file_path}: {e}")
            raise

    def batch_process(self, directory: Path) -> List[Document]:
        """
        批次處理目錄中的所有文件
        
        Args:
            directory: 目錄路徑
            
        Returns:
            List[Document]: 處理後的文件列表
        """
        logger.info(f"批次處理目錄: {directory}")
        
        if not directory.exists():
            raise FileNotFoundError(f"目錄不存在: {directory}")
        
        documents = []
        supported_formats = self.config.input.supported_formats
        
        # 遞迴搜尋文件
        if self.config.input.recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                # 檢查檔案格式
                if file_path.suffix.lower().lstrip('.') in supported_formats:
                    try:
                        document = self.process_document(file_path)
                        documents.append(document)
                    except Exception as e:
                        logger.warning(f"跳過文件 {file_path}: {e}")
                        continue
        
        logger.info(f"批次處理完成，共處理 {len(documents)} 個文件")
        return documents

    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        分割文本為塊
        
        Args:
            text: 要分割的文本
            chunk_size: 塊大小
            overlap: 重疊大小
            
        Returns:
            List[str]: 分割後的文本塊
        """
        # 暫時使用簡單的分割方法
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        return chunks