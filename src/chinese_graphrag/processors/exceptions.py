"""
文件處理相關的例外類別

定義文件處理過程中可能發生的各種錯誤
"""

from typing import Optional


class DocumentProcessingError(Exception):
    """文件處理基礎錯誤類別"""

    def __init__(self, message: str, file_path: Optional[str] = None):
        self.message = message
        self.file_path = file_path
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.file_path:
            return f"文件處理錯誤 ({self.file_path}): {self.message}"
        return f"文件處理錯誤: {self.message}"


class UnsupportedFileFormatError(DocumentProcessingError):
    """不支援的檔案格式錯誤"""

    def __init__(self, file_format: str, file_path: Optional[str] = None):
        message = f"不支援的檔案格式: {file_format}"
        super().__init__(message, file_path)
        self.file_format = file_format


class FileCorruptionError(DocumentProcessingError):
    """檔案損壞錯誤"""

    def __init__(self, file_path: str, details: Optional[str] = None):
        message = "檔案損壞或無法讀取"
        if details:
            message += f": {details}"
        super().__init__(message, file_path)


class FileNotFoundError(DocumentProcessingError):
    """檔案不存在錯誤"""

    def __init__(self, file_path: str):
        message = "檔案不存在"
        super().__init__(message, file_path)


class EncodingDetectionError(DocumentProcessingError):
    """編碼檢測錯誤"""

    def __init__(self, file_path: str, details: Optional[str] = None):
        message = "無法檢測檔案編碼"
        if details:
            message += f": {details}"
        super().__init__(message, file_path)


class ContentExtractionError(DocumentProcessingError):
    """內容提取錯誤"""

    def __init__(self, file_path: str, details: Optional[str] = None):
        message = "無法提取檔案內容"
        if details:
            message += f": {details}"
        super().__init__(message, file_path)
