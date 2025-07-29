"""
測試文字檔案處理器
"""

import tempfile
from pathlib import Path

import pytest

from chinese_graphrag.processors.exceptions import ContentExtractionError
from chinese_graphrag.processors.text_processor import MarkdownProcessor, TextProcessor


class TestTextProcessor:
    """測試純文字處理器"""
    
    def setup_method(self):
        """設定測試環境"""
        self.processor = TextProcessor()
    
    def test_supported_extensions(self):
        """測試支援的檔案格式"""
        assert ".txt" in self.processor.supported_extensions
        assert self.processor.can_process("test.txt")
        assert not self.processor.can_process("test.pdf")
    
    def test_extract_content_success(self):
        """測試成功提取內容"""
        test_content = "這是測試內容\n包含多行文字\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name
        
        try:
            content = self.processor.extract_content(tmp_path)
            assert "這是測試內容" in content
            assert "包含多行文字" in content
        finally:
            Path(tmp_path).unlink()
    
    def test_extract_content_empty_file(self):
        """測試空檔案處理"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write("")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ContentExtractionError):
                self.processor.extract_content(tmp_path)
        finally:
            Path(tmp_path).unlink()
    
    def test_extract_content_whitespace_only(self):
        """測試只有空白字符的檔案"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write("   \n\n\t  ")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ContentExtractionError):
                self.processor.extract_content(tmp_path)
        finally:
            Path(tmp_path).unlink()
    
    def test_process_file_success(self):
        """測試成功處理檔案"""
        test_content = "測試文件內容"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name
        
        try:
            document = self.processor.process_file(tmp_path)
            
            assert document.content == test_content
            assert document.file_path == tmp_path
            assert document.file_type == ".txt"
            assert document.language == "zh"
        finally:
            Path(tmp_path).unlink()
    
    def test_process_file_with_chinese_content(self):
        """測試處理中文內容"""
        chinese_content = """
        這是一個包含中文的測試檔案。
        
        內容包括：
        - 繁體中文字符
        - 簡體中文字符
        - 標點符號：，。！？；：「」『』
        
        測試完成。
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write(chinese_content)
            tmp_path = tmp.name
        
        try:
            document = self.processor.process_file(tmp_path)
            
            assert "繁體中文字符" in document.content
            assert "簡體中文字符" in document.content
            assert "標點符號" in document.content
        finally:
            Path(tmp_path).unlink()


class TestMarkdownProcessor:
    """測試 Markdown 處理器"""
    
    def setup_method(self):
        """設定測試環境"""
        self.processor = MarkdownProcessor()
    
    def test_supported_extensions(self):
        """測試支援的檔案格式"""
        assert ".md" in self.processor.supported_extensions
        assert ".markdown" in self.processor.supported_extensions
        assert self.processor.can_process("test.md")
        assert self.processor.can_process("test.markdown")
        assert not self.processor.can_process("test.txt")
    
    def test_extract_content_simple_markdown(self):
        """測試簡單 Markdown 內容提取"""
        markdown_content = """
# 標題

這是一個段落。

## 子標題

- 項目一
- 項目二

**粗體文字**和*斜體文字*。
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write(markdown_content)
            tmp_path = tmp.name
        
        try:
            content = self.processor.extract_content(tmp_path)
            
            # 檢查 HTML 標籤已被移除
            assert "<h1>" not in content
            assert "<p>" not in content
            assert "<strong>" not in content
            
            # 檢查內容保留
            assert "標題" in content
            assert "這是一個段落" in content
            assert "項目一" in content
            assert "粗體文字" in content
        finally:
            Path(tmp_path).unlink()
    
    def test_extract_content_with_code_blocks(self):
        """測試包含程式碼區塊的 Markdown"""
        markdown_content = """
# 程式碼範例

```python
def hello():
    print("Hello, World!")
```

這是說明文字。
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write(markdown_content)
            tmp_path = tmp.name
        
        try:
            content = self.processor.extract_content(tmp_path)
            
            assert "程式碼範例" in content
            assert "def hello" in content
            assert "這是說明文字" in content
        finally:
            Path(tmp_path).unlink()
    
    def test_extract_raw_content(self):
        """測試提取原始 Markdown 內容"""
        markdown_content = "# 標題\n\n**粗體**文字"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write(markdown_content)
            tmp_path = tmp.name
        
        try:
            raw_content = self.processor.extract_raw_content(tmp_path)
            
            # 原始內容應保留 Markdown 語法
            assert "# 標題" in raw_content
            assert "**粗體**" in raw_content
        finally:
            Path(tmp_path).unlink()
    
    def test_process_file_success(self):
        """測試成功處理 Markdown 檔案"""
        markdown_content = """
# 測試文件

這是一個測試用的 Markdown 文件。

## 功能測試

- 列表項目
- 另一個項目
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write(markdown_content)
            tmp_path = tmp.name
        
        try:
            document = self.processor.process_file(tmp_path)
            
            assert "測試文件" in document.content
            assert "功能測試" in document.content
            assert "列表項目" in document.content
            assert document.file_type == ".md"
            assert document.language == "zh"
        finally:
            Path(tmp_path).unlink()
    
    def test_extract_content_empty_markdown(self):
        """測試空 Markdown 檔案"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', 
                                       encoding='utf-8', delete=False) as tmp:
            tmp.write("")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ContentExtractionError):
                self.processor.extract_content(tmp_path)
        finally:
            Path(tmp_path).unlink()