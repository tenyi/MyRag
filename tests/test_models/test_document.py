"""
文件資料模型測試
"""

import pytest
from pydantic import ValidationError

from chinese_graphrag.models.document import Document


class TestDocument:
    """測試文件資料模型"""

    def test_create_document_with_required_fields(self):
        """測試使用必要欄位建立文件"""
        doc = Document(
            title="測試文件", content="這是測試內容", file_path="/path/to/test.txt"
        )

        assert doc.title == "測試文件"
        assert doc.content == "這是測試內容"
        assert doc.file_path == "/path/to/test.txt"
        assert doc.language == "zh"  # 預設值
        assert doc.encoding == "utf-8"  # 預設值

    def test_create_document_with_all_fields(self):
        """測試使用所有欄位建立文件"""
        doc = Document(
            title="完整測試文件",
            content="這是完整的測試內容",
            file_path="/path/to/complete.pdf",
            language="en",
            file_type="pdf",
            file_size=1024,
            encoding="utf-8",
        )

        assert doc.title == "完整測試文件"
        assert doc.content == "這是完整的測試內容"
        assert doc.file_path == "/path/to/complete.pdf"
        assert doc.language == "en"
        assert doc.file_type == "pdf"
        assert doc.file_size == 1024
        assert doc.encoding == "utf-8"

    def test_title_validation(self):
        """測試標題驗證"""
        # 空標題應該失敗
        with pytest.raises(ValidationError):
            Document(title="", content="內容", file_path="/path")

        # 只有空白的標題應該失敗
        with pytest.raises(ValidationError):
            Document(title="   ", content="內容", file_path="/path")

        # 太長的標題應該失敗
        with pytest.raises(ValidationError):
            Document(title="x" * 501, content="內容", file_path="/path")

        # 正常標題應該成功，並且會被 strip
        doc = Document(title="  正常標題  ", content="內容", file_path="/path")
        assert doc.title == "正常標題"

    def test_content_validation(self):
        """測試內容驗證"""
        # 空內容應該失敗
        with pytest.raises(ValidationError):
            Document(title="標題", content="", file_path="/path")

        # 只有空白的內容應該失敗
        with pytest.raises(ValidationError):
            Document(title="標題", content="   ", file_path="/path")

        # 正常內容應該成功，並且會被 strip
        doc = Document(title="標題", content="  正常內容  ", file_path="/path")
        assert doc.content == "正常內容"

    def test_file_path_validation(self):
        """測試檔案路徑驗證"""
        # 空路徑應該失敗
        with pytest.raises(ValidationError):
            Document(title="標題", content="內容", file_path="")

        # 只有空白的路徑應該失敗
        with pytest.raises(ValidationError):
            Document(title="標題", content="內容", file_path="   ")

        # 正常路徑應該成功，並且會被 strip
        doc = Document(title="標題", content="內容", file_path="  /path/to/file.txt  ")
        assert doc.file_path == "/path/to/file.txt"

    def test_language_validation(self):
        """測試語言驗證"""
        # 正確的語言代碼應該成功
        doc = Document(title="標題", content="內容", file_path="/path", language="en")
        assert doc.language == "en"

        # 錯誤的語言代碼格式應該失敗
        with pytest.raises(ValidationError):
            Document(
                title="標題", content="內容", file_path="/path", language="english"
            )

        with pytest.raises(ValidationError):
            Document(title="標題", content="內容", file_path="/path", language="e")

    def test_file_size_validation(self):
        """測試檔案大小驗證"""
        # 正數應該成功
        doc = Document(title="標題", content="內容", file_path="/path", file_size=1024)
        assert doc.file_size == 1024

        # 零應該成功
        doc = Document(title="標題", content="內容", file_path="/path", file_size=0)
        assert doc.file_size == 0

        # 負數應該失敗
        with pytest.raises(ValidationError):
            Document(title="標題", content="內容", file_path="/path", file_size=-1)

    def test_file_name_property(self):
        """測試檔案名稱屬性"""
        doc = Document(title="標題", content="內容", file_path="/path/to/test.txt")
        assert doc.file_name == "test.txt"

        doc = Document(title="標題", content="內容", file_path="simple.pdf")
        assert doc.file_name == "simple.pdf"

    def test_file_extension_property(self):
        """測試檔案副檔名屬性"""
        doc = Document(title="標題", content="內容", file_path="/path/to/test.txt")
        assert doc.file_extension == ".txt"

        doc = Document(title="標題", content="內容", file_path="/path/to/test.PDF")
        assert doc.file_extension == ".pdf"  # 應該轉為小寫

        doc = Document(title="標題", content="內容", file_path="/path/to/noext")
        assert doc.file_extension == ""

    def test_content_length_property(self):
        """測試內容長度屬性"""
        doc = Document(title="標題", content="測試內容", file_path="/path")
        assert doc.content_length == len("測試內容")

    def test_get_summary(self):
        """測試取得摘要"""
        # 短內容應該返回完整內容
        short_content = "這是短內容"
        doc = Document(title="標題", content=short_content, file_path="/path")
        assert doc.get_summary() == short_content

        # 長內容應該被截斷
        long_content = "這是一個很長的內容" * 50
        doc = Document(title="標題", content=long_content, file_path="/path")
        summary = doc.get_summary(max_length=50)
        assert len(summary) <= 53  # 50 + "..."
        assert summary.endswith("...")

        # 自訂長度
        summary = doc.get_summary(max_length=20)
        assert len(summary) <= 23  # 20 + "..."
        assert summary.endswith("...")

    def test_serialization(self):
        """測試序列化"""
        doc = Document(
            title="測試文件",
            content="測試內容",
            file_path="/path/to/test.txt",
            file_size=1024,
        )

        # 測試轉換為字典
        data = doc.to_dict()
        assert data["title"] == "測試文件"
        assert data["content"] == "測試內容"
        assert data["file_path"] == "/path/to/test.txt"
        assert data["file_size"] == 1024

        # 測試從字典重建
        new_doc = Document.from_dict(data)
        assert new_doc.title == doc.title
        assert new_doc.content == doc.content
        assert new_doc.file_path == doc.file_path
        assert new_doc.file_size == doc.file_size
