"""Chinese GraphRAG CLI 主程式。

提供完整的命令列介面，支援：
- 系統初始化和配置管理
- 文件索引和知識圖譜建構
- 查詢和檢索功能
- 系統管理和監控
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from ..config import create_default_config, load_config, validate_config as config_validator
from ..monitoring import get_logger, setup_logging
from ..monitoring.logger import LogConfig

# 建立控制台輸出物件
console = Console()

# 建立日誌器
logger = get_logger(__name__)


@click.group()
@click.option(
    "--config", "-c", type=click.Path(exists=True, path_type=Path), help="配置檔案路徑"
)
@click.option("--verbose", "-v", is_flag=True, help="啟用詳細輸出")
@click.option("--quiet", "-q", is_flag=True, help="靜默模式")
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool, quiet: bool):
    """Chinese GraphRAG - 中文知識圖譜檢索增強生成系統。

    使用範例:

    \b
    # 初始化系統
    chinese-graphrag init

    \b
    # 索引文件
    chinese-graphrag index --input ./documents --output ./data

    \b
    # 執行查詢
    chinese-graphrag query "您的中文問題"

    \b
    # 驗證配置
    chinese-graphrag validate-config
    """
    # 確保上下文物件存在
    ctx.ensure_object(dict)

    # 設定日誌級別
    if verbose:
        log_level = "DEBUG"
    elif quiet:
        log_level = "ERROR"
    else:
        log_level = "INFO"

    # 初始化日誌系統
    log_config = LogConfig(level=log_level)
    setup_logging(log_config)

    # 載入配置
    try:
        if config:
            ctx.obj["config"] = load_config(config)
            ctx.obj["config_path"] = config
        else:
            # 嘗試載入預設配置
            default_configs = [
                Path("config/settings.yaml"),
                Path("settings.yaml"),
                Path(".chinese-graphrag/config.yaml"),
            ]

            config_loaded = False
            for config_path in default_configs:
                if config_path.exists():
                    ctx.obj["config"] = load_config(config_path)
                    ctx.obj["config_path"] = config_path
                    config_loaded = True
                    break

            if not config_loaded:
                ctx.obj["config"] = None
                ctx.obj["config_path"] = None

    except Exception as e:
        if not quiet:
            console.print(f"[red]載入配置檔案失敗: {e}[/red]")
        ctx.obj["config"] = None
        ctx.obj["config_path"] = None

    # 儲存選項
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="config/settings.yaml",
    help="配置檔案輸出路徑",
)
@click.option("--env-template", is_flag=True, help="同時建立環境變數範本檔案")
@click.pass_context
def init(ctx: click.Context, output: Path, env_template: bool):
    """初始化 Chinese GraphRAG 系統。

    建立預設配置檔案和目錄結構。
    """
    try:
        # 建立輸出目錄
        output.parent.mkdir(parents=True, exist_ok=True)

        # 建立預設配置
        config_path = create_default_config(output)

        if not ctx.obj["quiet"]:
            console.print(f"[green]✓ 配置檔案已建立: {config_path}[/green]")

        # 建立環境變數範本
        if env_template:
            from ..config.env import SYSTEM_ENV_VARS, env_manager

            env_template_path = output.parent / ".env.template"
            env_manager.create_env_template(SYSTEM_ENV_VARS, env_template_path)

            if not ctx.obj["quiet"]:
                console.print(
                    f"[green]✓ 環境變數範本已建立: {env_template_path}[/green]"
                )

        # 建立必要目錄
        directories = ["input", "output", "cache", "logs", "data"]

        for dir_name in directories:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)

            if not ctx.obj["quiet"]:
                console.print(f"[green]✓ 目錄已建立: {dir_path}[/green]")

        if not ctx.obj["quiet"]:
            console.print(
                "\n[bold green]🎉 Chinese GraphRAG 系統初始化完成！[/bold green]"
            )
            console.print("\n下一步:")
            console.print("1. 編輯配置檔案並設定 API 金鑰")
            console.print("2. 將文件放入 input/ 目錄")
            console.print("3. 執行索引: chinese-graphrag index")

    except Exception as e:
        logger.error(f"初始化失敗: {e}")
        if not ctx.obj["quiet"]:
            console.print(f"[red]初始化失敗: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="要驗證的配置檔案路徑",
)
@click.pass_context
def validate_config(ctx: click.Context, config_path: Optional[Path]):
    """驗證配置檔案。

    檢查配置檔案的語法和內容正確性。
    """
    try:
        # 決定要驗證的配置
        if config_path:
            config = load_config(config_path)
            path_to_show = config_path
        elif ctx.obj["config"]:
            config = ctx.obj["config"]
            path_to_show = ctx.obj["config_path"]
        else:
            if not ctx.obj["quiet"]:
                console.print(
                    "[red]未找到配置檔案。請指定配置路徑或先執行 init 命令。[/red]"
                )
            sys.exit(1)

        # 執行驗證
        config_validator(config)

        if not ctx.obj["quiet"]:
            console.print(f"[green]✓ 配置檔案驗證通過: {path_to_show}[/green]")

            # 顯示配置摘要
            console.print("\n[bold]配置摘要:[/bold]")
            console.print(f"  模型數量: {len(config.models)}")
            console.print(f"  預設 LLM: {config.model_selection.default_llm}")
            console.print(
                f"  預設 Embedding: {config.model_selection.default_embedding}"
            )
            console.print(f"  向量資料庫: {config.vector_store.type}")
            console.print(f"  日誌級別: {config.logging.level}")

    except Exception as e:
        logger.error(f"配置驗證失敗: {e}")
        if not ctx.obj["quiet"]:
            console.print(f"[red]配置驗證失敗: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def version(ctx: click.Context):
    """顯示版本資訊。"""
    try:
        # 嘗試從 pyproject.toml 獲取版本
        import importlib.metadata

        version = importlib.metadata.version("chinese-graphrag")
    except importlib.metadata.PackageNotFoundError:
        version = "開發版本"

    if not ctx.obj["quiet"]:
        console.print(f"[bold]Chinese GraphRAG[/bold] {version}")
        console.print("中文知識圖譜檢索增強生成系統")


@cli.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="輸出格式",
)
@click.pass_context
def status(ctx: click.Context, output_format: str):
    """顯示系統狀態。"""
    try:
        from ..monitoring import (
            get_error_tracker,
            get_metrics_collector,
            get_system_monitor,
        )

        # 獲取系統狀態
        system_monitor = get_system_monitor()
        metrics_collector = get_metrics_collector()
        error_tracker = get_error_tracker()

        # 收集狀態資訊
        system_stats = system_monitor.collect_system_stats()
        metrics_summary = metrics_collector.get_metrics_summary()
        error_stats = error_tracker.get_error_stats()

        if output_format == "table":
            from rich.table import Table

            # 系統資源表
            table = Table(title="系統狀態")
            table.add_column("項目", style="cyan")
            table.add_column("值", style="green")

            table.add_row("CPU 使用率", f"{system_stats.cpu_percent:.1f}%")
            table.add_row("記憶體使用率", f"{system_stats.memory_percent:.1f}%")
            table.add_row("磁碟使用率", f"{system_stats.disk_percent:.1f}%")
            table.add_row("程序數量", str(system_stats.process_count))
            table.add_row("錯誤總數", str(error_stats.total_errors))

            console.print(table)

        elif output_format == "json":
            import json

            status_data = {
                "system": system_stats.to_dict(),
                "metrics": metrics_summary,
                "errors": {
                    "total": error_stats.total_errors,
                    "rate": error_stats.error_rate,
                },
            }
            console.print(json.dumps(status_data, indent=2, ensure_ascii=False))

        elif output_format == "yaml":
            import yaml

            status_data = {
                "system": system_stats.to_dict(),
                "metrics": metrics_summary,
                "errors": {
                    "total": error_stats.total_errors,
                    "rate": error_stats.error_rate,
                },
            }
            console.print(yaml.dump(status_data, allow_unicode=True))

    except Exception as e:
        logger.error(f"獲取系統狀態失敗: {e}")
        if not ctx.obj["quiet"]:
            console.print(f"[red]獲取系統狀態失敗: {e}[/red]")
        sys.exit(1)


def setup_warning_filters():
    """設置警告過濾器以抑制已知的棄用警告"""
    import warnings

    # 抑制 jieba 的 pkg_resources 棄用警告
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pkg_resources",
        message=".*pkg_resources is deprecated.*",
    )

    # 可以在此添加其他警告過濾器


def main():
    """CLI 主入口函數。"""
    # 設置警告過濾器
    setup_warning_filters()

    try:
        # 註冊索引命令
        from .index_commands import index, show_index, import_file, scan_files

        cli.add_command(index)
        cli.add_command(show_index)
        cli.add_command(import_file, name="import-file")
        cli.add_command(scan_files, name="scan-files")

        # 註冊查詢命令
        from .query_commands import batch_query, query, test_llm_segmentation

        cli.add_command(query)
        cli.add_command(batch_query)
        cli.add_command(test_llm_segmentation)

        # 註冊管理命令
        from .management_commands import clean, config_cmd, doctor

        cli.add_command(clean)
        cli.add_command(doctor)
        cli.add_command(config_cmd, name="config")  # 避免與 Click 內建的 config 衝突

        # 註冊 API 命令
        from .api_commands import api

        cli.add_command(api)

        # 啟動 CLI
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]操作已取消[/yellow]")
        sys.exit(1)
    except Exception as e:
        import traceback

        full_traceback = traceback.format_exc()
        console.print(f"[red]發生未預期的錯誤: {e}[/red]")
        console.print(f"[red]完整錯誤追踪:[/red]")
        console.print(full_traceback)
        sys.exit(1)


# 匯入 CLI 工具模組
from .utils import (
    console,
    interactive_prompt,
    progress_manager,
    show_completion_message,
    show_error_message,
    show_welcome_message,
)

if __name__ == "__main__":
    main()
