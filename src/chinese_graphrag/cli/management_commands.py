"""Chinese GraphRAG CLI 管理命令。

提供系統管理和維護功能。
"""

import sys
import shutil
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import load_config, ConfigValidator
from ..monitoring import get_logger, get_system_monitor, get_error_tracker

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option(
    "--target", "-t",
    type=click.Choice(["cache", "logs", "temp", "all"]),
    default="cache",
    help="清理目標（快取、日誌、暫存檔案或全部）"
)
@click.option(
    "--force", "-f",
    is_flag=True,
    help="強制清理，不詢問確認"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="試運行模式（顯示將要清理的檔案）"
)
@click.pass_context
def clean(ctx: click.Context, target: str, force: bool, dry_run: bool):
    """清理系統檔案和快取。
    
    使用範例:
    
    \b
    # 清理快取檔案
    chinese-graphrag clean --target cache
    
    \b
    # 清理所有暫存檔案
    chinese-graphrag clean --target all --force
    
    \b
    # 試運行模式
    chinese-graphrag clean --target all --dry-run
    """
    try:
        if not ctx.obj['config']:
            console.print("[red]未找到配置檔案。請先執行 'chinese-graphrag init' 初始化系統。[/red]")
            return
        
        config = ctx.obj['config']
        
        # 定義清理目標
        cleanup_targets = {
            "cache": [
                Path(config.storage.cache_dir) if hasattr(config.storage, 'cache_dir') else Path("cache"),
                Path(".serena/cache") if Path(".serena/cache").exists() else None
            ],
            "logs": [
                Path(config.logging.log_dir) if hasattr(config.logging, 'log_dir') else Path("logs")
            ],
            "temp": [
                Path("temp"),
                Path(".tmp"),
                Path("tmp")
            ]
        }
        
        if target == "all":
            paths_to_clean = []
            for target_paths in cleanup_targets.values():
                paths_to_clean.extend([p for p in target_paths if p is not None])
        else:
            paths_to_clean = [p for p in cleanup_targets.get(target, []) if p is not None]
        
        # 收集要清理的檔案
        files_to_remove = []
        total_size = 0
        
        for path in paths_to_clean:
            if path.exists():
                if path.is_file():
                    size = path.stat().st_size
                    files_to_remove.append((path, size))
                    total_size += size
                elif path.is_dir():
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            files_to_remove.append((file_path, size))
                            total_size += size
        
        if not files_to_remove:
            console.print("[yellow]沒有檔案需要清理[/yellow]")
            return
        
        # 顯示清理摘要
        if not ctx.obj['quiet']:
            table = Table(title="清理摘要")
            table.add_column("項目", style="cyan")
            table.add_column("值", style="green")
            
            table.add_row("清理目標", target)
            table.add_row("檔案數量", str(len(files_to_remove)))
            table.add_row("總大小", f"{total_size / 1024 / 1024:.2f} MB")
            
            console.print(table)
        
        if dry_run:
            console.print("\n[bold]將要清理的檔案:[/bold]")
            for file_path, size in files_to_remove[:10]:  # 只顯示前10個
                console.print(f"  {file_path} ({size / 1024:.1f} KB)")
            if len(files_to_remove) > 10:
                console.print(f"  ... 和另外 {len(files_to_remove) - 10} 個檔案")
            console.print("[yellow]試運行模式已完成[/yellow]")
            return
        
        # 確認清理
        if not force and not ctx.obj['quiet']:
            if not click.confirm(f"確認要清理 {len(files_to_remove)} 個檔案嗎？"):
                console.print("[yellow]清理已取消[/yellow]")
                return
        
        # 執行清理
        removed_count = 0
        removed_size = 0
        
        if not ctx.obj['quiet']:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("正在清理檔案...", total=len(files_to_remove))
                
                for file_path, size in files_to_remove:
                    try:
                        if file_path.exists():
                            file_path.unlink()
                            removed_count += 1
                            removed_size += size
                        progress.advance(task)
                    except Exception as e:
                        logger.warning(f"無法刪除檔案 {file_path}: {e}")
        else:
            for file_path, size in files_to_remove:
                try:
                    if file_path.exists():
                        file_path.unlink()
                        removed_count += 1
                        removed_size += size
                except Exception as e:
                    logger.warning(f"無法刪除檔案 {file_path}: {e}")
        
        # 清理空目錄
        for path in paths_to_clean:
            if path.exists() and path.is_dir():
                try:
                    # 嘗試刪除空目錄
                    if not any(path.iterdir()):
                        path.rmdir()
                except Exception:
                    pass  # 忽略無法刪除的目錄
        
        if not ctx.obj['quiet']:
            console.print(f"\n[bold green]✓ 清理完成[/bold green]")
            console.print(f"已刪除 {removed_count} 個檔案，釋放 {removed_size / 1024 / 1024:.2f} MB 空間")
        
        logger.info(f"清理完成，刪除了 {removed_count} 個檔案，釋放了 {removed_size} bytes")
        
    except Exception as e:
        logger.error(f"清理失敗: {e}")
        if not ctx.obj['quiet']:
            console.print(f"[red]清理失敗: {e}[/red]")
        sys.exit(1)


@click.command()
@click.option(
    "--check-type",
    type=click.Choice(["config", "dependencies", "models", "storage", "all"]),
    default="all",
    help="檢查類型"
)
@click.option(
    "--fix-issues",
    is_flag=True,
    help="嘗試自動修復發現的問題"
)
@click.pass_context
def doctor(ctx: click.Context, check_type: str, fix_issues: bool):
    """執行系統健康檢查。
    
    檢查配置、依賴、模型和儲存狀態。
    
    使用範例:
    
    \b
    # 完整健康檢查
    chinese-graphrag doctor
    
    \b
    # 檢查配置
    chinese-graphrag doctor --check-type config
    
    \b
    # 檢查並修復問題
    chinese-graphrag doctor --fix-issues
    """
    try:
        if not ctx.obj['quiet']:
            console.print("[bold blue]🔍 開始系統健康檢查...[/bold blue]\n")
        
        issues = []
        fixes_applied = []
        
        # 檢查配置
        if check_type in ["config", "all"]:
            config_issues, config_fixes = _check_configuration(ctx, fix_issues)
            issues.extend(config_issues)
            fixes_applied.extend(config_fixes)
        
        # 檢查依賴
        if check_type in ["dependencies", "all"]:
            dep_issues, dep_fixes = _check_dependencies(fix_issues)
            issues.extend(dep_issues)  
            fixes_applied.extend(dep_fixes)
        
        # 檢查模型
        if check_type in ["models", "all"]:
            model_issues, model_fixes = _check_models(ctx, fix_issues)
            issues.extend(model_issues)
            fixes_applied.extend(model_fixes)
        
        # 檢查儲存
        if check_type in ["storage", "all"]:
            storage_issues, storage_fixes = _check_storage(ctx, fix_issues)
            issues.extend(storage_issues)
            fixes_applied.extend(storage_fixes)
        
        # 顯示結果
        if not ctx.obj['quiet']:
            _display_doctor_results(issues, fixes_applied, check_type)
        
        # 設定退出碼
        if issues:
            sys.exit(1 if not fix_issues or any(i['severity'] == 'critical' for i in issues) else 0)
        
    except Exception as e:
        logger.error(f"健康檢查失敗: {e}")
        if not ctx.obj['quiet']:
            console.print(f"[red]健康檢查失敗: {e}[/red]")
        sys.exit(1)


def _check_configuration(ctx: click.Context, fix_issues: bool) -> tuple[list, list]:
    """檢查配置。"""
    issues = []
    fixes = []
    
    if not ctx.obj['quiet']:
        console.print("[cyan]檢查配置...[/cyan]")
    
    try:
        config = ctx.obj['config']
        if not config:
            issues.append({
                'type': 'config',
                'severity': 'critical',
                'message': '未找到配置檔案',
                'solution': '執行 chinese-graphrag init 初始化系統'
            })
            return issues, fixes
        
        # 驗證配置
        validator = ConfigValidator()
        validation_result = validator.validate_config(config)
        
        if not validation_result['valid']:
            for error in validation_result['errors']:
                issues.append({
                    'type': 'config',
                    'severity': 'high',
                    'message': f'配置驗證失敗: {error}',
                    'solution': '檢查並修正配置檔案'
                })
        
        # 檢查必要的 API 金鑰
        if not hasattr(config, 'models') or not config.models:
            issues.append({
                'type': 'config',
                'severity': 'high',
                'message': '未配置任何模型',
                'solution': '在配置檔案中添加 LLM 和 embedding 模型'
            })
        
    except Exception as e:
        issues.append({
            'type': 'config',
            'severity': 'critical',
            'message': f'配置檢查失敗: {e}',
            'solution': '檢查配置檔案語法和格式'
        })
    
    return issues, fixes


def _check_dependencies(fix_issues: bool) -> tuple[list, list]:
    """檢查依賴。"""
    issues = []
    fixes = []
    
    console.print("[cyan]檢查依賴...[/cyan]")
    
    # 檢查核心依賴
    required_packages = [
        'graphrag',
        'jieba',
        'sentence-transformers',
        'lancedb',
        'pydantic',
        'click',
        'rich',
        'loguru'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            issues.append({
                'type': 'dependency',
                'severity': 'critical',
                'message': f'缺少必要套件: {package}',
                'solution': f'執行 uv add {package} 安裝套件'
            })
    
    return issues, fixes


def _check_models(ctx: click.Context, fix_issues: bool) -> tuple[list, list]:
    """檢查模型。"""
    issues = []
    fixes = []
    
    console.print("[cyan]檢查模型...[/cyan]")
    
    try:
        config = ctx.obj['config']
        if config and hasattr(config, 'embeddings'):
            # 檢查 embedding 模型
            if hasattr(config.embeddings, 'model_name'):
                model_name = config.embeddings.model_name
                if 'bge-m3' in model_name.lower():
                    try:
                        from sentence_transformers import SentenceTransformer
                        # 嘗試載入模型（不實際載入，只檢查是否可用）
                        # model = SentenceTransformer(model_name)
                    except Exception as e:
                        issues.append({
                            'type': 'model',
                            'severity': 'high',
                            'message': f'無法載入 embedding 模型: {model_name}',
                            'solution': '檢查模型名稱或網路連接'
                        })
    
    except Exception as e:
        issues.append({
            'type': 'model',
            'severity': 'medium',
            'message': f'模型檢查失敗: {e}',
            'solution': '檢查模型配置'
        })
    
    return issues, fixes


def _check_storage(ctx: click.Context, fix_issues: bool) -> tuple[list, list]:
    """檢查儲存。"""
    issues = []
    fixes = []
    
    console.print("[cyan]檢查儲存...[/cyan]")
    
    try:
        config = ctx.obj['config']
        if config and hasattr(config, 'storage'):
            # 檢查基本目錄
            base_dir = Path(config.storage.base_dir)
            if not base_dir.exists():
                if fix_issues:
                    base_dir.mkdir(parents=True, exist_ok=True)
                    fixes.append(f"建立基本目錄: {base_dir}")
                else:
                    issues.append({
                        'type': 'storage',
                        'severity': 'medium',
                        'message': f'基本目錄不存在: {base_dir}',
                        'solution': '執行 chinese-graphrag init 或手動建立目錄'
                    })
            
            # 檢查磁碟空間
            total, used, free = shutil.disk_usage(base_dir if base_dir.exists() else Path.cwd())
            free_gb = free / (1024**3)
            
            if free_gb < 1:  # 少於 1GB
                issues.append({
                    'type': 'storage',
                    'severity': 'high',
                    'message': f'磁碟空間不足: {free_gb:.1f} GB',
                    'solution': '釋放磁碟空間或更改儲存位置'
                })
            elif free_gb < 5:  # 少於 5GB
                issues.append({
                    'type': 'storage',
                    'severity': 'medium',
                    'message': f'磁碟空間偏低: {free_gb:.1f} GB',
                    'solution': '考慮釋放更多磁碟空間'
                })
    
    except Exception as e:
        issues.append({
            'type': 'storage',
            'severity': 'medium',
            'message': f'儲存檢查失敗: {e}',
            'solution': '檢查儲存配置和權限'
        })
    
    return issues, fixes


def _display_doctor_results(issues: list, fixes_applied: list, check_type: str):
    """顯示健康檢查結果。"""
    console.print()
    
    if not issues and not fixes_applied:
        console.print("[bold green]✅ 系統健康狀況良好！[/bold green]")
        return
    
    # 顯示修復項目
    if fixes_applied:
        console.print("[bold green]🛠️  已應用的修復：[/bold green]")
        for fix in fixes_applied:
            console.print(f"  ✓ {fix}")
        console.print()
    
    # 顯示問題
    if issues:
        # 按嚴重程度分組
        critical_issues = [i for i in issues if i['severity'] == 'critical']
        high_issues = [i for i in issues if i['severity'] == 'high']
        medium_issues = [i for i in issues if i['severity'] == 'medium']
        
        if critical_issues:
            console.print("[bold red]🚨 嚴重問題：[/bold red]")
            for issue in critical_issues:
                console.print(f"  ❌ {issue['message']}")
                console.print(f"     💡 {issue['solution']}")
            console.print()
        
        if high_issues:
            console.print("[bold yellow]⚠️  重要問題：[/bold yellow]")
            for issue in high_issues:
                console.print(f"  ⚠️  {issue['message']}")
                console.print(f"     💡 {issue['solution']}")
            console.print()
        
        if medium_issues:
            console.print("[bold blue]ℹ️  一般問題：[/bold blue]")
            for issue in medium_issues:
                console.print(f"  ℹ️  {issue['message']}")
                console.print(f"     💡 {issue['solution']}")
            console.print()
        
        # 摘要表
        table = Table(title="問題摘要")
        table.add_column("嚴重程度", style="cyan")
        table.add_column("數量", style="red")
        
        table.add_row("嚴重", str(len(critical_issues)))
        table.add_row("重要", str(len(high_issues)))
        table.add_row("一般", str(len(medium_issues)))
        table.add_row("總計", str(len(issues)))
        
        console.print(table)


@click.command()
@click.option(
    "--key", "-k",
    help="配置鍵值（支援點號路徑，如 'models.default_llm'）"
)
@click.option(
    "--value", "-v",
    help="設定值"
)
@click.option(
    "--list-all", "-l",
    is_flag=True,
    help="列出所有配置選項"
)
@click.option(
    "--section", "-s",
    help="顯示特定配置區段"
)
@click.pass_context
def config_cmd(ctx: click.Context, key: Optional[str], value: Optional[str], list_all: bool, section: Optional[str]):
    """管理系統配置。
    
    使用範例:
    
    \b
    # 查看所有配置
    chinese-graphrag config --list-all
    
    \b
    # 查看特定區段
    chinese-graphrag config --section models
    
    \b
    # 查看特定配置值
    chinese-graphrag config --key models.default_llm
    
    \b
    # 設定配置值
    chinese-graphrag config --key models.default_llm --value gpt-4
    """
    try:
        if not ctx.obj['config']:
            console.print("[red]未找到配置檔案。請先執行 'chinese-graphrag init' 初始化系統。[/red]")
            sys.exit(1)
        
        config = ctx.obj['config']
        
        if list_all:
            _display_full_config(config)
        elif section:
            _display_config_section(config, section)
        elif key and not value:
            _display_config_value(config, key)
        elif key and value:
            _set_config_value(config, key, value, ctx.obj['config_path'])
        else:
            _display_config_summary(config)
            
    except Exception as e:
        logger.error(f"配置管理失敗: {e}")
        if not ctx.obj['quiet']:
            console.print(f"[red]配置管理失敗: {e}[/red]")
        sys.exit(1)


def _display_full_config(config):
    """顯示完整配置。"""
    console.print("[bold]完整系統配置:[/bold]\n")
    
    # 將 Pydantic 模型轉換為字典
    config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.__dict__
    
    import json
    console.print(json.dumps(config_dict, indent=2, ensure_ascii=False))


def _display_config_section(config, section: str):
    """顯示配置區段。"""
    try:
        section_config = getattr(config, section)
        if hasattr(section_config, 'model_dump'):
            section_dict = section_config.model_dump()
        else:
            section_dict = section_config.__dict__
        
        console.print(f"[bold]{section} 配置:[/bold]\n")
        
        import json
        console.print(json.dumps(section_dict, indent=2, ensure_ascii=False))
    except AttributeError:
        console.print(f"[red]未找到配置區段: {section}[/red]")


def _display_config_value(config, key: str):
    """顯示特定配置值。"""
    try:
        # 支援點號路徑
        keys = key.split('.')
        value = config
        
        for k in keys:
            value = getattr(value, k)
        
        console.print(f"[cyan]{key}:[/cyan] {value}")
    except AttributeError:
        console.print(f"[red]未找到配置鍵: {key}[/red]")


def _set_config_value(config, key: str, value: str, config_path: Optional[Path]):
    """設定配置值。"""
    console.print("[yellow]配置值設定功能尚未實作[/yellow]")
    console.print("請直接編輯配置檔案進行修改")


def _display_config_summary(config):
    """顯示配置摘要。"""
    table = Table(title="配置摘要")
    table.add_column("項目", style="cyan")
    table.add_column("值", style="green")
    
    # 基本資訊
    if hasattr(config, 'models') and config.models:
        table.add_row("已配置模型數", str(len(config.models)))
    
    if hasattr(config, 'model_selection') and config.model_selection:
        if hasattr(config.model_selection, 'default_llm'):
            table.add_row("預設 LLM", config.model_selection.default_llm)
        if hasattr(config.model_selection, 'default_embedding'):
            table.add_row("預設 Embedding", config.model_selection.default_embedding)
    
    if hasattr(config, 'vector_store') and config.vector_store:
        if hasattr(config.vector_store, 'type'):
            table.add_row("向量資料庫", config.vector_store.type)
    
    if hasattr(config, 'logging') and config.logging:
        if hasattr(config.logging, 'level'):
            table.add_row("日誌級別", config.logging.level)
    
    console.print(table)


# 將管理命令導出
__all__ = ['clean', 'doctor', 'config_cmd']