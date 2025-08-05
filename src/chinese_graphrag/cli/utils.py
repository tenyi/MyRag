#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI 工具模組

提供進度顯示、互動功能和 UI 元件等通用工具。
"""

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

# 全域 console 實例
console = Console()


class ProgressManager:
    """管理 CLI 進度顯示和用戶互動的工具類。"""

    def __init__(self, console_instance: Console = None):
        """初始化進度管理器。

        Args:
            console_instance: Rich Console 實例，預設使用全域 console
        """
        self.console = console_instance or console
        self._current_task = None
        self._progress = None
        self._tasks = {}

    def start_progress(self):
        """啟動進度顯示。"""
        if not self._progress:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
            )
            self._progress.start()

    def add_task(self, description: str, total: int = None) -> str:
        """新增一個進度任務。

        Args:
            description: 任務描述
            total: 總步驟數（可選）

        Returns:
            任務 ID
        """
        if not self._progress:
            self.start_progress()

        task_id = self._progress.add_task(description, total=total)
        self._tasks[task_id] = {
            "description": description,
            "total": total,
            "completed": 0,
        }
        self._current_task = task_id
        return task_id

    def update_task(
        self, task_id: str = None, advance: int = 1, description: str = None
    ):
        """更新任務進度。

        Args:
            task_id: 任務 ID，預設使用當前任務
            advance: 前進步數
            description: 新的描述（可選）
        """
        if not self._progress:
            return

        task_id = task_id or self._current_task
        if task_id is not None:
            self._progress.update(task_id, advance=advance, description=description)
            if task_id in self._tasks:
                self._tasks[task_id]["completed"] += advance
                if description:
                    self._tasks[task_id]["description"] = description

    def complete_task(self, task_id: str = None):
        """完成任務。

        Args:
            task_id: 任務 ID，預設使用當前任務
        """
        if not self._progress:
            return

        task_id = task_id or self._current_task
        if task_id is not None:
            # 確保進度條完成
            if task_id in self._tasks:
                total = self._tasks[task_id]["total"]
                completed = self._tasks[task_id]["completed"]
                if total and completed < total:
                    self._progress.update(task_id, completed=total)
            else:
                self._progress.update(task_id, completed=True)

    def stop_progress(self):
        """停止進度顯示。"""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._current_task = None
            self._tasks.clear()


class InteractivePrompt:
    """處理互動式用戶輸入的工具類。"""

    def __init__(self, console_instance: Console = None):
        """初始化互動提示器。

        Args:
            console_instance: Rich Console 實例，預設使用全域 console
        """
        self.console = console_instance or console

    def confirm(self, message: str, default: bool = True) -> bool:
        """顯示確認對話框。

        Args:
            message: 確認訊息
            default: 預設值

        Returns:
            用戶選擇結果
        """
        try:
            return Confirm.ask(message, default=default, console=self.console)
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]操作已取消[/yellow]")
            return False

    def select(self, message: str, choices: List[str], default: int = 0) -> str:
        """顯示選擇對話框。

        Args:
            message: 選擇訊息
            choices: 選項列表
            default: 預設選項索引

        Returns:
            選擇的選項
        """
        self.console.print(f"[bold blue]?[/bold blue] {message}")

        for i, choice in enumerate(choices):
            marker = ">" if i == default else " "
            self.console.print(f"  {marker} {i + 1}. {choice}")

        try:
            while True:
                response = Prompt.ask(
                    "\n[bold blue]請選擇[/bold blue]",
                    choices=[str(i + 1) for i in range(len(choices))],
                    default=str(default + 1),
                    console=self.console,
                )

                try:
                    index = int(response) - 1
                    if 0 <= index < len(choices):
                        return choices[index]
                except ValueError:
                    pass

        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]操作已取消[/yellow]")
            return choices[default]

    def input_text(
        self, message: str, default: str = None, validation: Callable = None
    ) -> str:
        """獲取文字輸入。

        Args:
            message: 輸入提示訊息
            default: 預設值
            validation: 驗證函數

        Returns:
            用戶輸入的文字
        """
        try:
            while True:
                response = Prompt.ask(message, default=default, console=self.console)

                if validation:
                    try:
                        validation(response)
                        break
                    except ValueError as e:
                        self.console.print(f"[red]{e}[/red]")
                        continue
                else:
                    break

            return response

        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]操作已取消[/yellow]")
            return default or ""


def create_status_table(title: str, data: Dict[str, Any]) -> Table:
    """創建狀態顯示表格。

    Args:
        title: 表格標題
        data: 狀態資料字典

    Returns:
        Rich Table 物件
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("項目", style="cyan", width=20)
    table.add_column("狀態", style="green", width=30)
    table.add_column("詳情", style="white", width=40)

    for key, value in data.items():
        if isinstance(value, dict):
            status = value.get("status", "未知")
            detail = value.get("detail", "")

            # 根據狀態設定顏色
            if status in ["正常", "成功", "完成", "OK"]:
                status = f"[green]{status}[/green]"
            elif status in ["警告", "部分完成", "Warning"]:
                status = f"[yellow]{status}[/yellow]"
            elif status in ["錯誤", "失敗", "Error", "Failed"]:
                status = f"[red]{status}[/red]"
            else:
                status = f"[white]{status}[/white]"

            table.add_row(key, status, str(detail))
        else:
            # 簡單值處理
            if isinstance(value, bool):
                status = "[green]是[/green]" if value else "[red]否[/red]"
                table.add_row(key, status, "")
            elif isinstance(value, (int, float)):
                table.add_row(key, f"[cyan]{value}[/cyan]", "")
            else:
                table.add_row(key, str(value), "")

    return table


def create_info_panel(title: str, content: str, style: str = "blue") -> Panel:
    """創建資訊面板。

    Args:
        title: 面板標題
        content: 面板內容
        style: 面板樣式

    Returns:
        Rich Panel 物件
    """
    return Panel.fit(content, title=title, style=style)


def show_welcome_message():
    """顯示歡迎訊息和使用提示。"""
    panel = Panel.fit(
        "[bold blue]歡迎使用 Chinese GraphRAG 系統[/bold blue]\n\n"
        "這是一個針對中文文件優化的知識圖譜檢索增強生成系統。\n"
        "支援中文文本處理、向量化、索引建構和智慧查詢。\n\n"
        "[dim]使用 --help 參數查看詳細說明\n"
        "使用 doctor 命令檢查系統狀態[/dim]",
        style="blue",
        title="Chinese GraphRAG",
    )
    console.print(panel)


def show_completion_message(
    operation: str, duration: float = None, details: str = None
):
    """顯示操作完成訊息。

    Args:
        operation: 操作名稱
        duration: 執行時間（秒）
        details: 詳細資訊
    """
    message = f"[green]✓[/green] {operation} 已完成"
    if duration:
        message += f" [dim]({duration:.2f}s)[/dim]"

    console.print(message)

    if details:
        console.print(f"[dim]{details}[/dim]")


def show_error_message(operation: str, error: str, suggestions: List[str] = None):
    """顯示錯誤訊息。

    Args:
        operation: 操作名稱
        error: 錯誤訊息
        suggestions: 建議解決方案列表
    """
    console.print(f"[red]✗[/red] {operation} 失敗")
    console.print(f"[red]錯誤：{error}[/red]")

    if suggestions:
        console.print("\n[yellow]建議解決方案：[/yellow]")
        for i, suggestion in enumerate(suggestions, 1):
            console.print(f"  {i}. {suggestion}")


def format_file_size(size_bytes: int) -> str:
    """格式化檔案大小顯示。

    Args:
        size_bytes: 檔案大小（位元組）

    Returns:
        格式化後的大小字串
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)

    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """格式化時間長度顯示。

    Args:
        seconds: 秒數

    Returns:
        格式化後的時間字串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


class StatusDisplay:
    """即時狀態顯示工具。"""

    def __init__(self, console_instance: Console = None):
        """初始化狀態顯示器。

        Args:
            console_instance: Rich Console 實例
        """
        self.console = console_instance or console
        self._live = None
        self._layout = None

    def start_live_display(self, refresh_per_second: int = 4):
        """啟動即時顯示。

        Args:
            refresh_per_second: 每秒刷新次數
        """
        if not self._live:
            self._layout = Layout()
            self._live = Live(
                self._layout,
                console=self.console,
                refresh_per_second=refresh_per_second,
                transient=True,
            )
            self._live.start()

    def update_display(self, content):
        """更新顯示內容。

        Args:
            content: 要顯示的內容
        """
        if self._layout:
            self._layout.update(content)

    def stop_live_display(self):
        """停止即時顯示。"""
        if self._live:
            self._live.stop()
            self._live = None
            self._layout = None


# 全域實例
progress_manager = ProgressManager()
interactive_prompt = InteractivePrompt()
status_display = StatusDisplay()


__all__ = [
    "ProgressManager",
    "InteractivePrompt",
    "StatusDisplay",
    "create_status_table",
    "create_info_panel",
    "show_welcome_message",
    "show_completion_message",
    "show_error_message",
    "format_file_size",
    "format_duration",
    "progress_manager",
    "interactive_prompt",
    "status_display",
    "console",
]
