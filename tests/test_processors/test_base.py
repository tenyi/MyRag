"""
測試基礎文件處理器功能
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from chinese_graphrag.processors.base import (
    BaseDocumentProcessor,
    DocumentProcessorManager,
)
from chinese_graphrag.processors.exceptions import (
    DocumentProcessingError,
    FileNotFoundError,
    UnsupportedFileFormatError,
)


class MockProcessor(BaseDocumentProcessor):
    """測試用的模擬處理器"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {".mock"}

    def extract_content(self, file_path: str) -> str:
        return "模擬內容"

    def process_file(self, file_path: str):
        self.validate_file(file_path)
        content = self.extract_content(file_path)
        return self.create_document(file_path, content)


class TestBaseDocumentProcessor:
    """測試基礎文件處理器"""

    def setup_method(self):
        """設定測試環境"""
        self.processor = MockProcessor()

    def test_can_process_supported_extension(self):
        """測試支援的檔案格式檢查"""
        assert self.processor.can_process("test.mock")
        assert not self.processor.can_process("test.txt")

    def test_validate_file_not_exists(self):
        """測試檔案不存在的驗證"""
        with pytest.raises(FileNotFoundError):
            self.processor.validate_file("nonexistent.mock")

    def test_validate_file_not_readable(self):
        """測試檔案無法讀取的驗證"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name

        try:
            # 移除讀取權限
            os.chmod(tmp_path, 0o000)

            with pytest.raises(DocumentProcessingError):
                self.processor.validate_file(tmp_path)
        finally:
            # 恢復權限並刪除檔案
            os.chmod(tmp_path, 0o644)
            os.unlink(tmp_path)

    def test_detect_encoding_utf8(self):
        """測試 UTF-8 編碼檢測"""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write("測試中文內容")
            tmp_path = tmp.name

        try:
            encoding = self.processor.detect_encoding(tmp_path)
            assert encoding in ["utf-8", "UTF-8"]
        finally:
            os.unlink(tmp_path)

    def test_get_file_info(self):
        """測試檔案資訊取得"""
        with tempfile.NamedTemporaryFile(suffix=".mock", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name

        try:
            file_info = self.processor.get_file_info(tmp_path)

            assert "file_name" in file_info
            assert "file_extension" in file_info
            assert "file_size" in file_info
            assert file_info["file_extension"] == ".mock"
            assert file_info["file_size"] > 0
        finally:
            os.unlink(tmp_path)

    def test_create_document(self):
        """測試文件物件建立"""
        with tempfile.NamedTemporaryFile(suffix=".mock", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name

        try:
            document = self.processor.create_document(
                file_path=tmp_path, content="測試內容", title="測試標題"
            )

            assert document.title == "測試標題"
            assert document.content == "測試內容"
            assert document.file_path == tmp_path
            assert document.file_type == ".mock"
        finally:
            os.unlink(tmp_path)


class TestDocumentProcessorManager:
    """測試文件處理器管理器"""

    def setup_method(self):
        """設定測試環境"""
        self.manager = DocumentProcessorManager()
        self.mock_processor = MockProcessor()

    def test_register_processor(self):
        """測試處理器註冊"""
        self.manager.register_processor("mock", self.mock_processor)

        assert "mock" in self.manager.processors
        assert ".mock" in self.manager.extension_mapping
        assert self.manager.extension_mapping[".mock"] == "mock"

    def test_get_processor(self):
        """測試處理器取得"""
        self.manager.register_processor("mock", self.mock_processor)

        processor = self.manager.get_processor("test.mock")
        assert processor is self.mock_processor

        processor = self.manager.get_processor("test.txt")
        assert processor is None

    def test_can_process(self):
        """測試檔案處理能力檢查"""
        self.manager.register_processor("mock", self.mock_processor)

        assert self.manager.can_process("test.mock")
        assert not self.manager.can_process("test.txt")

    def test_process_file_unsupported_format(self):
        """測試處理不支援的檔案格式"""
        with pytest.raises(UnsupportedFileFormatError):
            self.manager.process_file("test.unsupported")

    def test_process_file_success(self):
        """測試成功處理檔案"""
        self.manager.register_processor("mock", self.mock_processor)

        with tempfile.NamedTemporaryFile(suffix=".mock", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name

        try:
            document = self.manager.process_file(tmp_path)
            assert document.content == "模擬內容"
            assert document.file_path == tmp_path
        finally:
            os.unlink(tmp_path)

    def test_batch_process_directory_not_exists(self):
        """測試批次處理不存在的目錄"""
        with pytest.raises(FileNotFoundError):
            self.manager.batch_process("nonexistent_directory")

    def test_batch_process_success(self):
        """測試成功批次處理"""
        self.manager.register_processor("mock", self.mock_processor)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 建立測試檔案
            mock_file1 = Path(tmp_dir) / "test1.mock"
            mock_file2 = Path(tmp_dir) / "test2.mock"
            unsupported_file = Path(tmp_dir) / "test.unsupported"

            mock_file1.write_text("content1")
            mock_file2.write_text("content2")
            unsupported_file.write_text("unsupported")

            documents = self.manager.batch_process(tmp_dir)

            # 應該只處理支援的檔案
            assert len(documents) == 2
            assert all(doc.content == "模擬內容" for doc in documents)

    def test_get_supported_extensions(self):
        """測試取得支援的副檔名"""
        self.manager.register_processor("mock", self.mock_processor)

        extensions = self.manager.get_supported_extensions()
        assert ".mock" in extensions

    def test_get_processor_info(self):
        """測試取得處理器資訊"""
        self.manager.register_processor("mock", self.mock_processor)

        info = self.manager.get_processor_info()
        assert "mock" in info
        assert "supported_extensions" in info["mock"]
        assert "class_name" in info["mock"]
        assert ".mock" in info["mock"]["supported_extensions"]
