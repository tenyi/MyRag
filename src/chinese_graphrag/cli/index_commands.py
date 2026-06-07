"""Chinese GraphRAG CLI 索引命令。

提供文件索引和知識圖譜建構功能。
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..indexing import IndexingEngine
from ..monitoring import get_error_tracker, get_logger, get_metrics_collector

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="輸入文件目錄或檔案路徑",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    help="輸出目錄路徑（預設使用配置中的設定）",
)
@click.option("--resume", is_flag=True, help="從上次中斷的地方繼續索引")
@click.option(
    "--incremental", is_flag=True, help="增量索引模式（只處理新增或修改的檔案）"
)
@click.option("--dry-run", is_flag=True, help="試運行模式（不實際執行索引）")
@click.option("--parallel-jobs", "-j", type=int, help="並行作業數量（覆蓋配置設定）")
@click.option("--chunk-size", type=int, help="文本分塊大小（覆蓋配置設定）")
@click.option(
    "--skip-extraction",
    type=click.Choice(["entities", "relationships", "communities"]),
    multiple=True,
    help="跳過指定的提取步驟",
)
@click.option(
    "--format",
    type=click.Choice(["auto", "txt", "pdf", "docx", "md", "jpg", "png", "tiff"]),
    default="auto",
    help="強制指定輸入檔案格式",
)
@click.pass_context
def index(
    ctx: click.Context,
    input_path: Path,
    output_path: Optional[Path],
    resume: bool,
    incremental: bool,
    dry_run: bool,
    parallel_jobs: Optional[int],
    chunk_size: Optional[int],
    skip_extraction: tuple,
    format: str,
):
    """對文件進行索引，建構知識圖譜。

    使用範例:

    \b
    # 索引單一文件
    chinese-graphrag index -i document.txt

    \b
    # 索引整個目錄
    chinese-graphrag index -i ./documents -o ./output

    \b
    # 增量索引
    chinese-graphrag index -i ./documents --incremental

    \b
    # 指定並行作業數
    chinese-graphrag index -i ./documents -j 8
    """
    if not ctx.obj["config"]:
        console.print(
            "[red]未找到配置檔案。請先執行 'chinese-graphrag init' 初始化系統。[/red]"
        )
        sys.exit(1)

    config = ctx.obj["config"]

    try:
        # 覆蓋配置設定
        if output_path:
            config.storage.base_dir = str(output_path)
        if parallel_jobs:
            config.parallelization.num_threads = parallel_jobs
        if chunk_size:
            config.chunks.size = chunk_size

        # 處理跳過的提取步驟
        if "entities" in skip_extraction:
            config.indexing.enable_entity_extraction = False
        if "relationships" in skip_extraction:
            config.indexing.enable_relationship_extraction = False
        if "communities" in skip_extraction:
            config.indexing.enable_community_detection = False

        # 建立索引引擎
        indexing_engine = IndexingEngine(config)

        # 檢查輸入路徑
        if not input_path.exists():
            console.print(f"[red]輸入路徑不存在: {input_path}[/red]")
            sys.exit(1)

        # 收集要處理的檔案
        files_to_process = _collect_files(
            input_path, format, config.input.supported_formats
        )

        if not files_to_process:
            console.print("[yellow]未找到可處理的檔案[/yellow]")
            sys.exit(0)

        # 顯示處理摘要
        if not ctx.obj["quiet"]:
            _show_processing_summary(
                files_to_process, config, dry_run, resume, incremental
            )

        if dry_run:
            console.print("[yellow]試運行模式已完成[/yellow]")
            return

        # 執行索引
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=ctx.obj["quiet"],
        ) as progress:

            # 建立進度任務
            total_task = progress.add_task("總進度", total=len(files_to_process))
            current_task = progress.add_task("當前檔案", total=100)

            # 設定進度回調
            def update_progress(
                current_file: str, file_progress: float, overall_progress: int
            ):
                progress.update(
                    current_task,
                    description=f"處理: {Path(current_file).name}",
                    completed=file_progress,
                )
                progress.update(total_task, completed=overall_progress)

            # 執行索引
            result = asyncio.run(
                indexing_engine.process_documents(
                    files_to_process,
                    progress_callback=update_progress,
                    resume=resume,
                    incremental=incremental,
                )
            )

        # 計算處理時間
        elapsed_time = time.time() - start_time

        # 顯示結果
        if not ctx.obj["quiet"]:
            _show_indexing_results(result, elapsed_time)

        # 記錄指標
        metrics_collector = get_metrics_collector()
        metrics_collector.record_counter("indexing.completed", 1)
        metrics_collector.record_gauge(
            "indexing.documents_processed", len(files_to_process)
        )
        metrics_collector.record_timer("indexing.total_time", elapsed_time)

        logger.info(
            f"索引完成，處理了 {len(files_to_process)} 個檔案，耗時 {elapsed_time:.2f} 秒"
        )

    except Exception as e:
        # 記錄錯誤
        import traceback

        full_traceback = traceback.format_exc()
        error_tracker = get_error_tracker()
        from ..monitoring.error_tracker import ErrorCategory, ErrorSeverity

        error_tracker.track_error(
            e, category=ErrorCategory.PROCESSING, severity=ErrorSeverity.HIGH
        )

        logger.error(f"索引失敗: {e}")
        logger.error(f"完整錯誤追踪: {full_traceback}")
        if not ctx.obj["quiet"]:
            console.print(f"[red]索引失敗: {e}[/red]")
            console.print(f"[red]完整錯誤追踪:[/red]")
            console.print(full_traceback)
        sys.exit(1)


def _collect_files(
    input_path: Path, format_filter: str, supported_formats: list
) -> list[Path]:
    """收集要處理的檔案。"""
    files = []

    if input_path.is_file():
        # 單一檔案
        if format_filter == "auto":
            if input_path.suffix.lstrip(".") in supported_formats:
                files.append(input_path)
        else:
            if input_path.suffix.lstrip(".") == format_filter:
                files.append(input_path)
    else:
        # 目錄
        for ext in supported_formats:
            if format_filter == "auto" or format_filter == ext:
                files.extend(input_path.rglob(f"*.{ext}"))

    return sorted(files)


def _show_processing_summary(
    files: list[Path], config, dry_run: bool, resume: bool, incremental: bool
):
    """顯示處理摘要。"""
    table = Table(title="索引設定摘要")
    table.add_column("項目", style="cyan")
    table.add_column("值", style="green")

    table.add_row("檔案數量", str(len(files)))
    table.add_row(
        "模式",
        (
            "試運行"
            if dry_run
            else ("恢復" if resume else ("增量" if incremental else "完整"))
        ),
    )
    table.add_row("並行作業", str(config.parallelization.num_threads))
    table.add_row("文本分塊大小", str(config.chunks.size))
    table.add_row("輸出目錄", config.storage.base_dir)
    table.add_row("實體提取", "✓" if config.indexing.enable_entity_extraction else "×")
    table.add_row(
        "關係提取", "✓" if config.indexing.enable_relationship_extraction else "×"
    )
    table.add_row(
        "社群偵測", "✓" if config.indexing.enable_community_detection else "×"
    )

    console.print(table)
    console.print()


def _show_indexing_results(result: dict, elapsed_time: float):
    """顯示索引結果。"""
    console.print("\n[bold green]🎉 索引完成！[/bold green]")

    # 結果統計表
    table = Table(title="處理結果")
    table.add_column("項目", style="cyan")
    table.add_column("數量", style="green")

    table.add_row("處理檔案", str(result.get("documents_processed", 0)))
    table.add_row("文本塊", str(result.get("chunks_created", 0)))
    table.add_row("提取實體", str(result.get("entities_extracted", 0)))
    table.add_row("發現關係", str(result.get("relationships_found", 0)))
    table.add_row("識別社群", str(result.get("communities_detected", 0)))
    table.add_row("處理時間", f"{elapsed_time:.2f} 秒")

    console.print(table)

    # 輸出檔案位置
    if result.get("output_files"):
        console.print("\n[bold]輸出檔案:[/bold]")
        for file_type, file_path in result["output_files"].items():
            console.print(f"  {file_type}: {file_path}")

    console.print("\n[bold]下一步:[/bold]")
    console.print('執行查詢: chinese-graphrag query "您的問題"')


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="索引結果目錄路徑",
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "detailed"]),
    default="table",
    help="輸出格式",
)
@click.pass_context
def show_index(ctx: click.Context, input_path: Path, format: str):
    """顯示索引資訊和統計。"""
    try:
        from ..indexing import IndexAnalyzer

        # 建立索引分析器
        analyzer = IndexAnalyzer(input_path)

        # 獲取索引資訊
        index_info = analyzer.get_index_info()

        if format == "table":
            # 基本統計表
            table = Table(title="索引統計")
            table.add_column("項目", style="cyan")
            table.add_column("數量", style="green")

            for key, value in index_info["statistics"].items():
                table.add_row(key, str(value))

            console.print(table)

        elif format == "json":
            import json

            console.print(json.dumps(index_info, indent=2, ensure_ascii=False))

        elif format == "detailed":
            # 詳細資訊
            console.print(f"[bold]索引目錄:[/bold] {input_path}")
            console.print(f"[bold]建立時間:[/bold] {index_info['created_at']}")
            console.print(f"[bold]最後更新:[/bold] {index_info['updated_at']}")

            # 統計表
            table = Table(title="詳細統計")
            table.add_column("類別", style="cyan")
            table.add_column("項目", style="yellow")
            table.add_column("數量", style="green")

            for category, items in index_info["detailed_stats"].items():
                for item, count in items.items():
                    table.add_row(category, item, str(count))

            console.print(table)

    except Exception as e:
        logger.error(f"顯示索引資訊失敗: {e}")
        if not ctx.obj["quiet"]:
            console.print(f"[red]顯示索引資訊失敗: {e}[/red]")
        sys.exit(1)


@click.command()
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="要匯入的檔案路徑",
)
@click.option(
    "--format",
    type=click.Choice(["auto", "txt", "pdf", "docx", "md", "jpg", "png", "tiff"]),
    default="auto",
    help="強制指定檔案格式",
)
@click.option(
    "--preview", is_flag=True, help="預覽模式（只顯示前 1000 個字符）"
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    help="輸出處理後的文字到檔案",
)
@click.pass_context
def import_file(
    ctx: click.Context,
    file_path: Path,
    format: str,
    preview: bool,
    output_file: Optional[Path],
):
    """匯入並處理 Word、PDF 或圖像檔案。

    使用範例:

    \b
    # 匯入 Word 檔案
    chinese-graphrag import-file -f document.docx

    \b
    # 匯入 PDF 並預覽
    chinese-graphrag import-file -f document.pdf --preview

    \b
    # 匯入圖像檔案 (OCR 識別)
    chinese-graphrag import-file -f image.jpg --preview

    \b
    # 匯入並輸出到檔案
    chinese-graphrag import-file -f document.docx -o output.txt
    """
    try:
        from ..processors import create_default_processor_manager

        # 建立處理器管理器
        processor_manager = create_default_processor_manager()

        # 檢查是否支援此檔案格式
        if not processor_manager.can_process(str(file_path)):
            file_ext = file_path.suffix.lower()
            supported_exts = processor_manager.get_supported_extensions()
            console.print(f"[red]不支援的檔案格式: {file_ext}[/red]")
            console.print(f"[yellow]支援的格式: {', '.join(supported_exts)}[/yellow]")
            sys.exit(1)

        # 顯示處理資訊
        if not ctx.obj["quiet"]:
            console.print(f"[cyan]正在處理檔案: {file_path}[/cyan]")
            console.print(f"[cyan]檔案大小: {file_path.stat().st_size / 1024:.2f} KB[/cyan]")

        # 處理檔案
        document = processor_manager.process_file(str(file_path))

        # 顯示處理結果
        content = document.content
        word_count = len(content)
        char_count = len(content.replace(" ", ""))

        if not ctx.obj["quiet"]:
            table = Table(title="檔案處理結果")
            table.add_column("項目", style="cyan")
            table.add_column("值", style="green")

            table.add_row("檔案名稱", document.title)
            table.add_row("檔案類型", document.file_type)
            table.add_row("檔案大小", f"{document.file_size / 1024:.2f} KB")
            table.add_row("編碼", document.encoding)
            table.add_row("文字長度", f"{word_count} 字符")
            table.add_row("中文字數", f"{char_count} 字")

            console.print(table)

        # 預覽模式
        if preview:
            preview_text = content[:1000] + "..." if len(content) > 1000 else content
            console.print("\n[bold]內容預覽:[/bold]")
            console.print(f"[dim]{preview_text}[/dim]")
        elif not output_file and not ctx.obj["quiet"]:
            # 如果沒有輸出檔案且不是靜默模式，顯示完整內容
            console.print("\n[bold]提取的內容:[/bold]")
            console.print(content)

        # 輸出到檔案
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            if not ctx.obj["quiet"]:
                console.print(f"\n[green]✓ 內容已保存到: {output_file}[/green]")

        logger.info(f"成功處理檔案: {file_path}")

    except Exception as e:
        logger.error(f"處理檔案失敗: {e}")
        if not ctx.obj["quiet"]:
            console.print(f"[red]處理檔案失敗: {e}[/red]")
        sys.exit(1)


@click.command()
@click.option(
    "--directory",
    "-d",
    "dir_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="要掃描的目錄路徑",
)
@click.option(
    "--recursive", "-r", is_flag=True, help="遞迴掃描子目錄"
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "detailed"]),
    default="table",
    help="輸出格式",
)
@click.pass_context
def scan_files(
    ctx: click.Context,
    dir_path: Path,
    recursive: bool,
    format: str,
):
    """掃描目錄中的 Word、PDF 和圖像檔案。

    使用範例:

    \b
    # 掃描目錄
    chinese-graphrag scan-files -d ./documents

    \b
    # 遞迴掃描
    chinese-graphrag scan-files -d ./documents --recursive

    \b
    # JSON 格式輸出
    chinese-graphrag scan-files -d ./documents --format json
    """
    try:
        from ..processors import create_default_processor_manager

        # 建立處理器管理器
        processor_manager = create_default_processor_manager()
        supported_extensions = processor_manager.get_supported_extensions()

        # 掃描檔案
        if not ctx.obj["quiet"]:
            console.print(f"[cyan]正在掃描目錄: {dir_path}[/cyan]")

        files_info = []
        
        # 獲取檔案列表
        if recursive:
            files = dir_path.rglob("*")
        else:
            files = dir_path.glob("*")

        # 過濾支援的檔案
        for file_path in files:
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                if file_ext in supported_extensions:
                    try:
                        stat = file_path.stat()
                        files_info.append({
                            "name": file_path.name,
                            "path": str(file_path),
                            "size": stat.st_size,
                            "type": file_ext,
                            "modified": stat.st_mtime,
                            "can_process": processor_manager.can_process(str(file_path))
                        })
                    except Exception as e:
                        logger.warning(f"無法獲取檔案資訊: {file_path}, 錯誤: {e}")

        # 排序
        files_info.sort(key=lambda x: x["modified"], reverse=True)

        # 顯示結果
        if format == "table":
            if files_info:
                table = Table(title=f"找到 {len(files_info)} 個可處理的檔案")
                table.add_column("檔案名", style="cyan")
                table.add_column("類型", style="yellow")
                table.add_column("大小", style="green")
                table.add_column("可處理", style="blue")

                for file_info in files_info[:20]:  # 只顯示前20個
                    size_kb = file_info["size"] / 1024
                    size_str = f"{size_kb:.2f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
                    can_process = "✓" if file_info["can_process"] else "×"
                    
                    table.add_row(
                        file_info["name"],
                        file_info["type"],
                        size_str,
                        can_process
                    )

                console.print(table)
                
                if len(files_info) > 20:
                    console.print(f"[dim]... 還有 {len(files_info) - 20} 個檔案[/dim]")
            else:
                console.print("[yellow]未找到可處理的檔案[/yellow]")

        elif format == "json":
            import json
            console.print(json.dumps(files_info, indent=2, ensure_ascii=False))

        elif format == "detailed":
            if files_info:
                for i, file_info in enumerate(files_info[:10], 1):
                    console.print(f"\n[bold]{i}. {file_info['name']}[/bold]")
                    console.print(f"   路徑: {file_info['path']}")
                    console.print(f"   類型: {file_info['type']}")
                    console.print(f"   大小: {file_info['size'] / 1024:.2f} KB")
                    console.print(f"   可處理: {'是' if file_info['can_process'] else '否'}")
                    
                    import time
                    modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_info['modified']))
                    console.print(f"   修改時間: {modified_time}")

                if len(files_info) > 10:
                    console.print(f"\n[dim]... 還有 {len(files_info) - 10} 個檔案[/dim]")
            else:
                console.print("[yellow]未找到可處理的檔案[/yellow]")

        # 統計資訊
        if not ctx.obj["quiet"] and files_info:
            total_size = sum(f["size"] for f in files_info)
            console.print(f"\n[bold]統計資訊:[/bold]")
            console.print(f"  總檔案數: {len(files_info)}")
            console.print(f"  總大小: {total_size / 1024 / 1024:.2f} MB")
            
            # 按類型統計
            type_stats = {}
            for file_info in files_info:
                file_type = file_info["type"]
                if file_type not in type_stats:
                    type_stats[file_type] = 0
                type_stats[file_type] += 1
            
            console.print("  檔案類型分布:")
            for file_type, count in type_stats.items():
                console.print(f"    {file_type}: {count} 個")

        logger.info(f"掃描完成，找到 {len(files_info)} 個可處理檔案")

    except Exception as e:
        logger.error(f"掃描檔案失敗: {e}")
        if not ctx.obj["quiet"]:
            console.print(f"[red]掃描檔案失敗: {e}[/red]")
        sys.exit(1)


# 將索引命令導出到 main.py 使用
__all__ = ["index", "show_index", "import_file", "scan_files"]
