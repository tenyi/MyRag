"""
統一日誌管理系統

提供結構化日誌記錄、多輸出目標和動態配置功能
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


@dataclass
class LogConfig:
    """日誌配置類別"""

    level: str = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )
    rotation: str = "10 MB"
    retention: str = "30 days"
    compression: str = "gz"
    backtrace: bool = True
    diagnose: bool = True
    enqueue: bool = True
    catch: bool = True
    colorize: bool = True
    serialize: bool = False

    # 輸出目標配置
    console_enabled: bool = True
    file_enabled: bool = True
    json_enabled: bool = False

    # 檔案路徑配置
    log_dir: str = "logs"
    log_file: str = "chinese_graphrag.log"
    error_file: str = "error.log"
    json_file: str = "structured.json"

    # 過濾器配置
    modules_filter: Dict[str, str] = field(default_factory=dict)
    exclude_modules: list = field(default_factory=list)


class StructuredLogRecord(BaseModel):
    """結構化日誌記錄模型"""

    timestamp: datetime
    level: str
    module: str
    function: str
    line: int
    message: str
    extra: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class LoggerManager:
    """日誌管理器"""

    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self._setup_complete = False
        self._handlers = []

    def setup_logging(self) -> None:
        """設定日誌系統"""
        if self._setup_complete:
            return

        # 移除預設處理器
        logger.remove()

        # 建立日誌目錄
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 設定控制台輸出
        if self.config.console_enabled:
            self._add_console_handler()

        # 設定檔案輸出
        if self.config.file_enabled:
            self._add_file_handlers()

        # 設定 JSON 輸出
        if self.config.json_enabled:
            self._add_json_handler()

        # 設定模組過濾器
        self._setup_module_filters()

        # 設定全域異常處理
        if self.config.catch:
            logger.catch(reraise=True)

        self._setup_complete = True
        logger.info("日誌系統初始化完成")

    def _add_console_handler(self) -> None:
        """添加控制台處理器"""
        handler_id = logger.add(
            sys.stderr,
            format=self.config.format,
            level=self.config.level,
            colorize=self.config.colorize,
            backtrace=self.config.backtrace,
            diagnose=self.config.diagnose,
            enqueue=self.config.enqueue,
            filter=self._create_filter(),
        )
        self._handlers.append(("console", handler_id))

    def _add_file_handlers(self) -> None:
        """添加檔案處理器"""
        log_dir = Path(self.config.log_dir)

        # 一般日誌檔案
        general_handler = logger.add(
            log_dir / self.config.log_file,
            format=self.config.format,
            level=self.config.level,
            rotation=self.config.rotation,
            retention=self.config.retention,
            compression=self.config.compression,
            backtrace=self.config.backtrace,
            diagnose=self.config.diagnose,
            enqueue=self.config.enqueue,
            filter=self._create_filter(),
        )
        self._handlers.append(("file_general", general_handler))

        # 錯誤日誌檔案
        error_handler = logger.add(
            log_dir / self.config.error_file,
            format=self.config.format,
            level="ERROR",
            rotation=self.config.rotation,
            retention=self.config.retention,
            compression=self.config.compression,
            backtrace=self.config.backtrace,
            diagnose=self.config.diagnose,
            enqueue=self.config.enqueue,
            filter=lambda record: record["level"].no >= logger.level("ERROR").no,
        )
        self._handlers.append(("file_error", error_handler))

    def _add_json_handler(self) -> None:
        """添加 JSON 處理器"""
        log_dir = Path(self.config.log_dir)

        def json_formatter(record):
            """JSON 格式化器"""
            structured_record = StructuredLogRecord(
                timestamp=datetime.fromtimestamp(record["time"].timestamp()),
                level=record["level"].name,
                module=record["name"],
                function=record["function"],
                line=record["line"],
                message=record["message"],
                extra=record.get("extra", {}),
            )
            return structured_record.json() + "\n"

        json_handler = logger.add(
            log_dir / self.config.json_file,
            format=json_formatter,
            level=self.config.level,
            rotation=self.config.rotation,
            retention=self.config.retention,
            compression=self.config.compression,
            serialize=True,
            enqueue=self.config.enqueue,
            filter=self._create_filter(),
        )
        self._handlers.append(("json", json_handler))

    def _create_filter(self):
        """建立日誌過濾器"""

        def filter_func(record):
            module_name = record["name"]

            # 排除指定模組
            if any(excluded in module_name for excluded in self.config.exclude_modules):
                return False

            # 模組級別過濾
            for module_pattern, level in self.config.modules_filter.items():
                if module_pattern in module_name:
                    return record["level"].no >= logger.level(level).no

            return True

        return filter_func

    def _setup_module_filters(self) -> None:
        """設定模組過濾器"""
        # 預設模組過濾設定
        default_filters = {
            "httpx": "WARNING",
            "urllib3": "WARNING",
            "requests": "WARNING",
            "asyncio": "WARNING",
            "concurrent.futures": "WARNING",
        }

        # 合併使用者設定
        self.config.modules_filter.update(default_filters)

    def get_logger(self, name: str) -> "logger":
        """取得指定名稱的日誌器"""
        if not self._setup_complete:
            self.setup_logging()
        return logger.bind(name=name)

    def update_level(self, level: str) -> None:
        """動態更新日誌級別"""
        self.config.level = level
        # 重新設定所有處理器
        for handler_type, handler_id in self._handlers:
            if handler_type != "file_error":  # 錯誤日誌保持 ERROR 級別
                logger.remove(handler_id)

        # 重新添加處理器
        self._handlers.clear()
        self._setup_complete = False
        self.setup_logging()

    def add_context(self, **kwargs) -> "logger":
        """添加上下文資訊"""
        return logger.bind(**kwargs)

    def shutdown(self) -> None:
        """關閉日誌系統"""
        for _, handler_id in self._handlers:
            logger.remove(handler_id)
        self._handlers.clear()
        self._setup_complete = False


# 全域日誌管理器
_logger_manager: Optional[LoggerManager] = None


def setup_logging(config: Optional[LogConfig] = None) -> None:
    """設定全域日誌系統"""
    global _logger_manager
    _logger_manager = LoggerManager(config)
    _logger_manager.setup_logging()


def get_logger(name: str = None) -> "logger":
    """取得日誌器實例"""
    global _logger_manager
    if _logger_manager is None:
        setup_logging()

    if name:
        return _logger_manager.get_logger(name)
    else:
        return logger


def update_log_level(level: str) -> None:
    """動態更新日誌級別"""
    global _logger_manager
    if _logger_manager:
        _logger_manager.update_level(level)


def add_log_context(**kwargs) -> "logger":
    """添加日誌上下文"""
    global _logger_manager
    if _logger_manager:
        return _logger_manager.add_context(**kwargs)
    return logger.bind(**kwargs)


def shutdown_logging() -> None:
    """關閉日誌系統"""
    global _logger_manager
    if _logger_manager:
        _logger_manager.shutdown()
        _logger_manager = None
