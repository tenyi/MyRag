"""
文件處理器

負責處理各種格式的文件，進行預處理和分塊
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from chinese_graphrag.config import GraphRAGConfig
from chinese_graphrag.models import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文件處理器"""

    def __init__(self, config: GraphRAGConfig):
        """
        初始化文件處理器

        Args:
            config: GraphRAG 配置
        """
        self.config = config
        self.supported_extensions = {".txt", ".md", ".json", ".csv"}

    def process_document(self, file_path: Path) -> Document:
        """
        處理單個文件

        Args:
            file_path: 文件路徑

        Returns:
            Document: 處理後的文件對象
        """
        try:
            # 讀取文件內容
            content = self._read_file(file_path)

            # 創建文件對象
            document = Document(
                id=str(file_path),
                title=file_path.stem,
                content=content,
                file_path=file_path,
                created_at=datetime.now(),
                metadata={
                    "file_size": file_path.stat().st_size,
                    "file_extension": file_path.suffix,
                    "language": "zh",
                },
            )

            logger.info(f"成功處理文件: {file_path}")
            return document

        except Exception as e:
            logger.error(f"處理文件 {file_path} 失敗: {e}")
            raise

    def batch_process(self, input_path: Path) -> List[Document]:
        """
        批次處理文件

        Args:
            input_path: 輸入路徑（文件或目錄）

        Returns:
            List[Document]: 處理後的文件列表
        """
        documents = []

        if input_path.is_file():
            # 處理單個文件
            if input_path.suffix in self.supported_extensions:
                documents.append(self.process_document(input_path))
        else:
            # 處理目錄中的所有文件
            for file_path in input_path.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix in self.supported_extensions
                ):
                    try:
                        documents.append(self.process_document(file_path))
                    except Exception as e:
                        logger.warning(f"跳過文件 {file_path}: {e}")

        logger.info(f"批次處理完成，共處理 {len(documents)} 個文件")
        return documents

    def split_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
        """
        分割文本為塊

        Args:
            text: 要分割的文本
            chunk_size: 塊大小
            overlap: 重疊大小

        Returns:
            List[str]: 分割後的文本塊
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # 如果不是最後一塊，嘗試在句號處分割
            if end < len(text):
                last_period = chunk.rfind("。")
                if last_period > chunk_size // 2:  # 確保塊不會太小
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1

            if chunk.strip():
                chunks.append(chunk.strip())

            # 計算下一個開始位置，考慮重疊
            start = max(start + 1, end - overlap)

            # 避免無限循環
            if start >= len(text):
                break

        return chunks

    def _read_file(self, file_path: Path) -> str:
        """
        讀取文件內容

        Args:
            file_path: 文件路徑

        Returns:
            str: 文件內容
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # 嘗試其他編碼
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        提取文件元資料

        Args:
            file_path: 文件路徑

        Returns:
            Dict[str, Any]: 元資料字典
        """
        stat = file_path.stat()

        return {
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "file_extension": file_path.suffix,
            "created_time": datetime.fromtimestamp(stat.st_ctime),
            "modified_time": datetime.fromtimestamp(stat.st_mtime),
            "language": "zh",  # 預設為中文
        }
