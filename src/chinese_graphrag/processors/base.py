"""
文件處理器基礎類別

定義文件處理器的抽象介面和共同功能
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set

import chardet
from loguru import logger

from ..models.document import Document
from .exceptions import (
    DocumentProcessingError,
    EncodingDetectionError,
    FileNotFoundError,
    UnsupportedFileFormatError,
)


class BaseDocumentProcessor(ABC):
    """文件處理器抽象基類

    定義所有文件處理器必須實作的介面
    """

    def __init__(self):
        self.supported_extensions: Set[str] = set()
        self.default_encoding = "utf-8"

    @abstractmethod
    def process_file(self, file_path: str) -> Document:
        """處理單一檔案

        Args:
            file_path: 檔案路徑

        Returns:
            Document: 處理後的文件物件

        Raises:
            DocumentProcessingError: 處理過程中發生錯誤
        """
        pass

    @abstractmethod
    def extract_content(self, file_path: str) -> str:
        """從檔案中提取文字內容

        Args:
            file_path: 檔案路徑

        Returns:
            str: 提取的文字內容

        Raises:
            DocumentProcessingError: 提取過程中發生錯誤
        """
        pass

    def can_process(self, file_path: str) -> bool:
        """檢查是否可以處理指定檔案

        Args:
            file_path: 檔案路徑

        Returns:
            bool: 是否可以處理
        """
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_extensions

    def validate_file(self, file_path: str) -> None:
        """驗證檔案是否存在且可讀取

        Args:
            file_path: 檔案路徑

        Raises:
            FileNotFoundError: 檔案不存在
            DocumentProcessingError: 檔案無法讀取
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        if not os.path.isfile(file_path):
            raise DocumentProcessingError("路徑不是檔案", file_path)

        if not os.access(file_path, os.R_OK):
            raise DocumentProcessingError("檔案無法讀取", file_path)

    def detect_encoding(self, file_path: str) -> str:
        """檢測檔案編碼

        Args:
            file_path: 檔案路徑

        Returns:
            str: 檢測到的編碼

        Raises:
            EncodingDetectionError: 編碼檢測失敗
        """
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read(10000)  # 讀取前 10KB 進行檢測

            result = chardet.detect(raw_data)
            if result["encoding"] is None:
                logger.warning(f"無法檢測檔案編碼，使用預設編碼: {file_path}")
                return self.default_encoding

            confidence = result["confidence"]
            encoding = result["encoding"]

            # 如果信心度太低，使用預設編碼
            if confidence < 0.7:
                logger.warning(
                    f"編碼檢測信心度較低 ({confidence:.2f})，使用預設編碼: {file_path}"
                )
                return self.default_encoding

            logger.debug(f"檢測到檔案編碼: {encoding} (信心度: {confidence:.2f})")
            return encoding

        except Exception as e:
            raise EncodingDetectionError(file_path, f"編碼檢測失敗: {str(e)}")

    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """取得檔案基本資訊

        Args:
            file_path: 檔案路徑

        Returns:
            Dict[str, any]: 檔案資訊
        """
        path_obj = Path(file_path)
        stat = path_obj.stat()

        return {
            "file_name": path_obj.name,
            "file_extension": path_obj.suffix.lower(),
            "file_size": stat.st_size,
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime,
        }

    def create_document(
        self,
        file_path: str,
        content: str,
        title: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> Document:
        """建立文件物件

        Args:
            file_path: 檔案路徑
            content: 文件內容
            title: 文件標題（可選）
            encoding: 檔案編碼（可選）

        Returns:
            Document: 文件物件
        """
        file_info = self.get_file_info(file_path)

        # 如果沒有提供標題，使用檔案名稱（不含副檔名）
        if title is None:
            title = Path(file_path).stem

        # 如果沒有提供編碼，使用預設編碼
        if encoding is None:
            encoding = self.default_encoding

        return Document(
            title=title,
            content=content,
            file_path=file_path,
            file_type=file_info["file_extension"],
            file_size=file_info["file_size"],
            encoding=encoding,
        )


class DocumentProcessorManager:
    """文件處理器管理器

    管理多個文件處理器，根據檔案類型選擇適當的處理器
    """

    def __init__(self):
        self.processors: Dict[str, BaseDocumentProcessor] = {}
        self.extension_mapping: Dict[str, str] = {}

    def register_processor(self, name: str, processor: BaseDocumentProcessor) -> None:
        """註冊文件處理器

        Args:
            name: 處理器名稱
            processor: 處理器實例
        """
        self.processors[name] = processor

        # 建立副檔名到處理器的映射
        for ext in processor.supported_extensions:
            self.extension_mapping[ext] = name

        logger.info(f"註冊文件處理器: {name}")

    def get_processor(self, file_path: str) -> Optional[BaseDocumentProcessor]:
        """根據檔案路徑取得適當的處理器

        Args:
            file_path: 檔案路徑

        Returns:
            Optional[BaseDocumentProcessor]: 處理器實例，如果沒有找到則返回 None
        """
        file_extension = Path(file_path).suffix.lower()
        processor_name = self.extension_mapping.get(file_extension)

        if processor_name:
            return self.processors[processor_name]

        return None

    def can_process(self, file_path: str) -> bool:
        """檢查是否可以處理指定檔案

        Args:
            file_path: 檔案路徑

        Returns:
            bool: 是否可以處理
        """
        return self.get_processor(file_path) is not None

    def process_file(self, file_path: str) -> Document:
        """處理單一檔案

        Args:
            file_path: 檔案路徑

        Returns:
            Document: 處理後的文件物件

        Raises:
            UnsupportedFileFormatError: 不支援的檔案格式
            DocumentProcessingError: 處理過程中發生錯誤
        """
        processor = self.get_processor(file_path)
        if processor is None:
            file_extension = Path(file_path).suffix.lower()
            raise UnsupportedFileFormatError(file_extension, file_path)

        return processor.process_file(file_path)

    def batch_process(self, directory: str, recursive: bool = True) -> List[Document]:
        """批次處理目錄中的檔案

        Args:
            directory: 目錄路徑
            recursive: 是否遞迴處理子目錄

        Returns:
            List[Document]: 處理後的文件列表
        """
        documents = []
        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(directory)

        if not directory_path.is_dir():
            raise DocumentProcessingError("路徑不是目錄", directory)

        # 取得所有檔案
        if recursive:
            files = directory_path.rglob("*")
        else:
            files = directory_path.glob("*")

        # 過濾出可處理的檔案
        processable_files = [
            f for f in files if f.is_file() and self.can_process(str(f))
        ]

        logger.info(f"找到 {len(processable_files)} 個可處理的檔案")

        # 處理每個檔案
        for file_path in processable_files:
            try:
                document = self.process_file(str(file_path))
                documents.append(document)
                logger.debug(f"成功處理檔案: {file_path}")
            except Exception as e:
                logger.error(f"處理檔案失敗: {file_path}, 錯誤: {str(e)}")
                # 繼續處理其他檔案，不中斷整個批次處理
                continue

        logger.info(f"批次處理完成，成功處理 {len(documents)} 個檔案")
        return documents

    def get_supported_extensions(self) -> Set[str]:
        """取得所有支援的檔案副檔名

        Returns:
            Set[str]: 支援的副檔名集合
        """
        return set(self.extension_mapping.keys())

    def get_processor_info(self) -> Dict[str, Dict[str, any]]:
        """取得所有處理器的資訊

        Returns:
            Dict[str, Dict[str, any]]: 處理器資訊
        """
        info = {}
        for name, processor in self.processors.items():
            info[name] = {
                "supported_extensions": list(processor.supported_extensions),
                "class_name": processor.__class__.__name__,
            }
        return info
