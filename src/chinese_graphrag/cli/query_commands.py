"""Chinese GraphRAG CLI 查詢命令。

提供中文問答和檢索功能。
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..monitoring import get_error_tracker, get_logger, get_metrics_collector
from ..query import QueryEngine

console = Console()
logger = get_logger(__name__)


def _get_default_llm_config(config, logger):
    """
    獲取默認 LLM 配置，優先使用用戶指定的默認模型

    Args:
        config: GraphRAG 配置對象
        logger: 日誌記錄器

    Returns:
        List[LLMConfig]: LLM 配置列表
    """
    from ..query.manager import LLMConfig, LLMProvider

    # 優先使用配置中指定的默認 LLM 模型
    default_llm_name = config.model_selection.default_llm
    default_llm_config_obj = config.get_llm_config(default_llm_name)

    llm_configs = []

    if default_llm_config_obj:
        # 檢查 API 金鑰（Ollama 模型不需要）
        api_key = getattr(default_llm_config_obj, "api_key", None)
        model_type_str = str(default_llm_config_obj.type).lower()

        # Ollama 模型不需要 API 金鑰
        if api_key or "ollama" in model_type_str:
            # 映射模型類型到 LLMProvider
            if "openai" in model_type_str:
                provider = LLMProvider.OPENAI
            elif "ollama" in model_type_str:
                provider = LLMProvider.OLLAMA
            else:
                provider = LLMProvider.MOCK

            default_config = LLMConfig(
                provider=provider,
                model=default_llm_config_obj.model,
                config={
                    "model": default_llm_config_obj.model,
                    "api_key": api_key,
                    "base_url": getattr(default_llm_config_obj, "api_base", None),
                    "temperature": getattr(default_llm_config_obj, "temperature", 0.7),
                },
                max_tokens=getattr(default_llm_config_obj, "max_tokens", 4000),
                temperature=getattr(default_llm_config_obj, "temperature", 0.7),
            )
            llm_configs.append(default_config)
            logger.info(
                f"使用默認 LLM 模型: {default_llm_name} ({default_llm_config_obj.model})"
            )
            return llm_configs
        else:
            logger.warning(f"默認 LLM 模型 {default_llm_name} 缺少 API 金鑰")

    # 如果默認模型不可用，嘗試備用模型
    if hasattr(config.model_selection, "fallback_models"):
        fallback_name = config.model_selection.fallback_models.get(default_llm_name)
        if fallback_name:
            fallback_config = config.get_llm_config(fallback_name)
            if fallback_config:
                logger.info(f"使用備用 LLM 模型: {fallback_name}")
                api_key = getattr(fallback_config, "api_key", None)
                model_type_str = str(fallback_config.type).lower()

                if api_key or "ollama" in model_type_str:
                    if "openai" in model_type_str:
                        provider = LLMProvider.OPENAI
                    elif "ollama" in model_type_str:
                        provider = LLMProvider.OLLAMA
                    else:
                        provider = LLMProvider.MOCK

                    fallback_llm_config = LLMConfig(
                        provider=provider,
                        model=fallback_config.model,
                        config={
                            "model": fallback_config.model,
                            "api_key": api_key,
                            "base_url": getattr(fallback_config, "api_base", None),
                            "temperature": getattr(fallback_config, "temperature", 0.7),
                        },
                        max_tokens=getattr(fallback_config, "max_tokens", 4000),
                        temperature=getattr(fallback_config, "temperature", 0.7),
                    )
                    llm_configs.append(fallback_llm_config)
                    return llm_configs

    # 如果都不可用，創建一個默認的測試配置
    logger.warning("未找到可用的 LLM 配置，使用測試模型")
    default_llm_config = LLMConfig(
        provider=LLMProvider.MOCK,
        model="test_model",
        config={},
        max_tokens=4000,
        temperature=0.7,
    )
    llm_configs.append(default_llm_config)
    return llm_configs


@click.command()
@click.argument("question", type=str)
@click.option(
    "--search-type",
    "-t",
    type=click.Choice(["auto", "global", "local"]),
    default="auto",
    help="搜尋類型（自動、全域、本地）",
)
@click.option("--max-tokens", type=int, help="回答最大 token 數（覆蓋配置設定）")
@click.option(
    "--community-level", type=int, help="社群層級（全域搜尋使用，覆蓋配置設定）"
)
@click.option(
    "--response-type",
    type=click.Choice(
        ["multiple_paragraphs", "single_paragraph", "real_time", "single_sentence"]
    ),
    help="回答格式類型",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["rich", "plain", "json", "markdown"]),
    default="rich",
    help="輸出格式",
)
@click.option("--show-sources", is_flag=True, help="顯示資料來源")
@click.option("--show-reasoning", is_flag=True, help="顯示推理過程")
@click.option("--interactive", "-i", is_flag=True, help="進入互動模式")
@click.option(
    "--enable-global-search",
    is_flag=True,
    help="啟用全域搜尋（預設關閉，僅使用本地搜尋）",
)
@click.option(
    "--use-jieba-segmentation",
    is_flag=True,
    help="使用 jieba 分詞（預設使用 LLM 分詞）",
)
@click.option("--simple", "-s", is_flag=True, help="返回精簡答案（不包含詳細推理過程）")
@click.pass_context
def query(
    ctx: click.Context,
    question: str,
    search_type: str,
    max_tokens: Optional[int],
    community_level: Optional[int],
    response_type: Optional[str],
    output_format: str,
    show_sources: bool,
    show_reasoning: bool,
    interactive: bool,
    enable_global_search: bool,
    use_jieba_segmentation: bool,
    simple: bool,
):
    """執行中文問答查詢。

    使用範例:

    \b
    # 基本查詢（使用 LLM 分詞）
    chinese-graphrag query "什麼是人工智慧？"

    \b
    # 使用 jieba 分詞
    chinese-graphrag query "小明姓什麼" --use-jieba-segmentation

    \b
    # 啟用全域搜尋
    chinese-graphrag query "介紹機器學習的應用" --enable-global-search

    \b
    # 指定搜尋類型
    chinese-graphrag query "深度學習的歷史" --search-type local

    \b
    # 顯示資料來源
    chinese-graphrag query "深度學習的歷史" --show-sources

    \b
    # 互動模式
    chinese-graphrag query "" --interactive
    """
    if not ctx.obj["config"]:
        console.print(
            "[red]未找到配置檔案。請先執行 'chinese-graphrag init' 初始化系統。[/red]"
        )
        sys.exit(1)

    config = ctx.obj["config"]

    try:
        # 覆蓋配置設定
        if max_tokens:
            config.query.max_tokens = max_tokens
        if community_level:
            config.query.community_level = community_level
        if response_type:
            config.query.response_type = response_type

        # 根據 CLI 選項覆蓋全域搜尋設定
        if enable_global_search:
            config.query.enable_global_search = True
        else:
            config.query.enable_global_search = False

        # 建立查詢引擎
        from ..config.models import VectorStoreType
        from ..indexing import GraphRAGIndexer
        from ..vector_stores import VectorStoreManager

        # 初始化索引器
        indexer = GraphRAGIndexer(config)

        # 初始化向量存儲
        vector_store_type = config.vector_store.type
        if isinstance(vector_store_type, str):
            vector_store_type = VectorStoreType(vector_store_type)
        vector_store = VectorStoreManager(vector_store_type)

        # 創建 QueryEngineConfig
        from ..query.engine import QueryEngineConfig
        from ..query.manager import LLMConfig, LLMProvider

        # 使用新的輔助函數獲取默認 LLM 配置
        llm_configs = _get_default_llm_config(config, logger)

        query_engine_config = QueryEngineConfig(
            llm_configs=llm_configs,
            enable_global_search=getattr(config.query, "enable_global_search", False),
            enable_local_search=getattr(config.query, "enable_local_search", True),
            enable_drift_search=getattr(config.query, "enable_drift_search", False),
            max_global_communities=getattr(config.query, "max_global_communities", 5),
            max_local_entities=getattr(config.query, "max_local_entities", 10),
            max_text_units=getattr(config.query, "max_text_units", 20),
        )

        query_engine = QueryEngine(query_engine_config, config, indexer, vector_store)

        # 決定分詞方法：默認使用 LLM 分詞，除非用戶明確指定使用 jieba
        use_llm_segmentation = not use_jieba_segmentation

        # 互動模式
        if interactive or not question.strip():
            _run_interactive_mode(
                query_engine,
                search_type,
                show_sources,
                show_reasoning,
                output_format,
                ctx.obj["quiet"],
                use_llm_segmentation,
            )
            return

        # 單次查詢
        _execute_single_query(
            query_engine,
            question,
            search_type,
            show_sources,
            show_reasoning,
            output_format,
            ctx.obj["quiet"],
            use_llm_segmentation,
        )

    except Exception as e:
        # 記錄錯誤
        error_tracker = get_error_tracker()
        from ..monitoring.error_tracker import ErrorCategory, ErrorSeverity

        error_tracker.track_error(
            e, category=ErrorCategory.PROCESSING, severity=ErrorSeverity.HIGH
        )

        logger.error(f"查詢失敗: {e}")
        if not ctx.obj["quiet"]:
            console.print(f"[red]查詢失敗: {e}[/red]")
        sys.exit(1)


@click.command()
@click.argument("question", required=True)
@click.pass_context
def test_llm_segmentation(ctx: click.Context, question: str):
    """測試 LLM 分詞功能

    使用範例:
    chinese-graphrag test-llm-segmentation "小明姓什麼"
    """
    if not ctx.obj["config"]:
        console.print(
            "[red]未找到配置檔案。請先執行 'chinese-graphrag init' 初始化系統。[/red]"
        )
        sys.exit(1)

    config = ctx.obj["config"]

    try:
        # 使用新的輔助函數獲取默認 LLM 配置
        from ..query.manager import LLMManager
        from ..query.processor import ChineseTextNormalizer

        llm_configs = _get_default_llm_config(config, logger)

        if not llm_configs:
            console.print("[red]未找到可用的 LLM 配置[/red]")
            sys.exit(1)

        # 初始化 LLM 管理器
        llm_manager = LLMManager(llm_configs)

        # 測試分詞
        async def run_test():
            console.print(f"[bold blue]測試查詢: {question}[/bold blue]\n")

            # 先測試不使用 LLM 的情況
            normalizer_no_llm = ChineseTextNormalizer()

            # 對比 jieba 分詞
            import jieba

            jieba_segments = list(jieba.cut(question, cut_all=False))
            jieba_keywords = normalizer_no_llm.extract_keywords(question)

            console.print("[cyan]jieba 分詞結果:[/cyan]")
            console.print(f"  分詞: {jieba_segments}")
            console.print(f"  關鍵詞: {jieba_keywords}\n")

            # 測試 LLM 分詞
            normalizer = ChineseTextNormalizer(llm_manager)

            try:
                with console.status("[bold green]正在使用 LLM 分詞..."):
                    llm_segments = await normalizer.llm_segment_text(question)
                    llm_keywords = await normalizer.extract_keywords_with_llm(question)

                console.print("[cyan]LLM 分詞結果:[/cyan]")
                console.print(f"  分詞: {llm_segments}")
                console.print(f"  關鍵詞: {llm_keywords}")

            except Exception as e:
                console.print(f"[red]LLM 分詞失敗: {e}[/red]")

        # 執行測試
        asyncio.run(run_test())

    except Exception as e:
        logger.error(f"測試失敗: {e}")
        console.print(f"[red]測試失敗: {e}[/red]")
        sys.exit(1)


def _execute_single_query(
    query_engine: QueryEngine,
    question: str,
    search_type: str,
    show_sources: bool,
    show_reasoning: bool,
    output_format: str,
    quiet: bool,
    use_llm_segmentation: bool = True,  # 新增參數，默認使用 LLM 分詞
    simple: bool = False,  # 新增精簡模式參數
):
    """執行單次查詢。"""
    start_time = time.time()

    if not quiet and output_format == "rich":
        console.print(f"\n[bold blue]🤔 問題: {question}[/bold blue]")

        with console.status("[bold green]正在思考..."):
            # 根據參數選擇使用 LLM 分詞或普通分詞
            if use_llm_segmentation:
                unified_result = asyncio.run(
                    query_engine.query_with_llm_segmentation(
                        question, search_type=search_type
                    )
                )
            else:
                unified_result = asyncio.run(
                    query_engine.query(question, search_type=search_type)
                )
        result = unified_result.to_dict()
        # 添加 CLI 期望的字段映射
        result["response"] = unified_result.answer
        result["reasoning"] = unified_result.reasoning_path
    else:
        # 根據參數選擇使用 LLM 分詞或普通分詞
        if use_llm_segmentation:
            unified_result = asyncio.run(
                query_engine.query_with_llm_segmentation(
                    question, search_type=search_type
                )
            )
        else:
            unified_result = asyncio.run(
                query_engine.query(question, search_type=search_type)
            )
        result = unified_result.to_dict()
        # 添加 CLI 期望的字段映射
        result["response"] = unified_result.answer
        result["reasoning"] = unified_result.reasoning_path

    elapsed_time = time.time() - start_time

    # 如果是精簡模式，直接輸出答案
    if simple:
        _display_simple_result(result, output_format, elapsed_time, quiet)
    else:
        # 格式化輸出
        _display_query_result(
            result, show_sources, show_reasoning, output_format, elapsed_time, quiet
        )

    # 記錄指標
    metrics_collector = get_metrics_collector()
    metrics_collector.record_counter("query.completed", 1)
    metrics_collector.record_timer("query.response_time", elapsed_time)
    metrics_collector.record_gauge(
        "query.search_type", 1, labels={"type": result.get("search_type", "unknown")}
    )

    # 在日誌中標註使用的分詞方法
    segmentation_method = "LLM分詞" if use_llm_segmentation else "jieba分詞"
    mode = "精簡模式" if simple else "完整模式"
    logger.info(
        f"查詢完成，耗時 {elapsed_time:.2f} 秒，搜尋類型: {result.get('search_type', 'unknown')}，分詞方法: {segmentation_method}，模式: {mode}"
    )


def _run_interactive_mode(
    query_engine: QueryEngine,
    default_search_type: str,
    show_sources: bool,
    show_reasoning: bool,
    output_format: str,
    quiet: bool,
    use_llm_segmentation: bool = True,  # 新增參數
):
    """運行互動模式。"""
    if not quiet:
        console.print("\n[bold green]🎯 進入互動查詢模式[/bold green]")
        segmentation_method = "LLM分詞" if use_llm_segmentation else "jieba分詞"
        console.print(f"當前分詞方法: [cyan]{segmentation_method}[/cyan]")
        console.print("輸入 'exit' 或 'quit' 退出，輸入 'help' 查看幫助")
        console.print("-" * 50)

    while True:
        try:
            # 獲取用戶輸入
            question = console.input(
                "\n[bold cyan]請輸入您的問題: [/bold cyan]"
            ).strip()

            if not question:
                continue

            # 處理特殊命令
            if question.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]再見！[/yellow]")
                break
            elif question.lower() in ["help", "h"]:
                _show_interactive_help()
                continue
            elif question.lower().startswith("set "):
                _handle_settings_command(question)
                continue

            # 執行查詢
            _execute_single_query(
                query_engine,
                question,
                default_search_type,
                show_sources,
                show_reasoning,
                output_format,
                quiet,
                use_llm_segmentation,
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]再見！[/yellow]")
            break
        except EOFError:
            console.print("\n[yellow]再見！[/yellow]")
            break


def _show_interactive_help():
    """顯示互動模式幫助。"""
    help_text = """
[bold]互動模式命令:[/bold]

[cyan]查詢命令:[/cyan]
  直接輸入問題                     - 執行查詢
  
[cyan]設定命令:[/cyan]
  set search-type <type>          - 設定搜尋類型 (auto/global/local)
  set sources <on/off>            - 開啟/關閉來源顯示
  set reasoning <on/off>          - 開啟/關閉推理過程顯示
  set segmentation <llm/jieba>    - 設定分詞方法 (LLM分詞/jieba分詞)
  
[cyan]系統命令:[/cyan]
  help, h                         - 顯示此幫助
  exit, quit, q                   - 退出互動模式

[yellow]注意:[/yellow] 分詞方法設定功能尚未實作，請使用命令行參數 --use-jieba-segmentation
"""
    console.print(Panel(help_text, title="幫助", expand=False))


def _handle_settings_command(command: str):
    """處理設定命令。"""
    # 這裡可以實作設定變更邏輯
    # 暫時顯示說明
    console.print("[yellow]設定功能尚未實作[/yellow]")


def _display_query_result(
    result: dict,
    show_sources: bool,
    show_reasoning: bool,
    output_format: str,
    elapsed_time: float,
    quiet: bool,
):
    """顯示查詢結果。"""
    if output_format == "json":
        import json

        console.print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if output_format == "plain":
        console.print(result.get("response", ""))
        return

    if output_format == "markdown":
        markdown_output = f"# 查詢結果\n\n{result.get('response', '')}"
        if show_sources and result.get("sources"):
            markdown_output += "\n\n## 資料來源\n\n"
            for i, source in enumerate(result["sources"], 1):
                markdown_output += f"{i}. {source}\n"
        console.print(Syntax(markdown_output, "markdown"))
        return

    # Rich 格式（預設）
    if not quiet:
        # 主要回答
        response_text = result.get("response", "未找到回答")
        console.print(
            Panel(
                response_text,
                title="[bold green]📝 回答",
                expand=False,
                border_style="green",
            )
        )

        # 搜尋類型和統計
        stats_table = Table(show_header=False, box=None, pad_edge=False)
        stats_table.add_column("項目", style="cyan", width=12)
        stats_table.add_column("值", style="green")

        stats_table.add_row("搜尋類型", result.get("search_type", "unknown"))
        stats_table.add_row("回應時間", f"{elapsed_time:.2f} 秒")
        if result.get("tokens_used"):
            stats_table.add_row("使用 Token", str(result["tokens_used"]))

        console.print("\n[bold]查詢統計:[/bold]")
        console.print(stats_table)

        # 資料來源
        if show_sources and result.get("sources"):
            console.print("\n[bold]📚 資料來源:[/bold]")
            sources_table = Table(show_header=False)
            sources_table.add_column("#", style="dim", width=3)
            sources_table.add_column("來源", style="blue")

            for i, source in enumerate(result["sources"], 1):
                sources_table.add_row(str(i), source)

            console.print(sources_table)

        # 推理過程
        if show_reasoning and result.get("reasoning"):
            console.print("\n[bold]🧠 推理過程:[/bold]")
            reasoning_text = result["reasoning"]
            console.print(Panel(reasoning_text, border_style="blue", expand=False))
    else:
        # 靜默模式只輸出回答
        console.print(result.get("response", ""))


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
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    help="查詢結果輸出檔案路徑",
)
@click.option(
    "--questions-file",
    type=click.Path(exists=True, path_type=Path),
    help="批次查詢問題檔案（每行一個問題）",
)
@click.option(
    "--search-type",
    "-t",
    type=click.Choice(["auto", "global", "local"]),
    default="auto",
    help="搜尋類型",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "markdown", "txt"]),
    default="json",
    help="輸出格式",
)
@click.pass_context
def batch_query(
    ctx: click.Context,
    input_path: Path,
    output_path: Optional[Path],
    questions_file: Optional[Path],
    search_type: str,
    output_format: str,
):
    """批次查詢處理。

    從檔案讀取問題列表並批次執行查詢。
    """
    if not ctx.obj["config"]:
        console.print(
            "[red]未找到配置檔案。請先執行 'chinese-graphrag init' 初始化系統。[/red]"
        )
        sys.exit(1)

    if not questions_file:
        console.print("[red]請指定問題檔案路徑（--questions-file）[/red]")
        sys.exit(1)

    try:
        # 讀取問題列表
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]

        if not questions:
            console.print("[yellow]未找到有效問題[/yellow]")
            return

        config = ctx.obj["config"]
        from ..config.models import VectorStoreType
        from ..indexing import GraphRAGIndexer
        from ..vector_stores import VectorStoreManager

        # 初始化索引器
        indexer = GraphRAGIndexer(config)

        # 初始化向量存儲
        vector_store_type = config.vector_store.type
        if isinstance(vector_store_type, str):
            vector_store_type = VectorStoreType(vector_store_type)
        vector_store = VectorStoreManager(vector_store_type)

        # 創建 QueryEngineConfig
        from ..query.engine import QueryEngineConfig
        from ..query.manager import LLMConfig, LLMProvider

        # 使用新的輔助函數獲取默認 LLM 配置
        llm_configs = _get_default_llm_config(config, logger)

        query_engine_config = QueryEngineConfig(
            llm_configs=llm_configs,
            enable_global_search=getattr(config.query, "enable_global_search", False),
            enable_local_search=getattr(config.query, "enable_local_search", True),
            enable_drift_search=getattr(config.query, "enable_drift_search", False),
            max_global_communities=getattr(config.query, "max_global_communities", 5),
            max_local_entities=getattr(config.query, "max_local_entities", 10),
            max_text_units=getattr(config.query, "max_text_units", 20),
        )

        query_engine = QueryEngine(query_engine_config, config, indexer, vector_store)

        results = []

        if not ctx.obj["quiet"]:
            console.print(f"[bold]開始批次查詢，共 {len(questions)} 個問題[/bold]")

        # 執行批次查詢
        for i, question in enumerate(questions, 1):
            if not ctx.obj["quiet"]:
                console.print(f"\n[cyan]{i}/{len(questions)}[/cyan] {question}")

            start_time = time.time()
            unified_result = asyncio.run(
                query_engine.query(question, search_type=search_type)
            )
            result = unified_result.to_dict()
            # 添加 CLI 期望的字段映射
            result["response"] = unified_result.answer
            result["reasoning"] = unified_result.reasoning_path

            elapsed_time = time.time() - start_time

            result["question"] = question
            result["query_time"] = elapsed_time
            result["index"] = i
            results.append(result)

            if not ctx.obj["quiet"]:
                console.print(f"[green]✓ 完成 ({elapsed_time:.2f}s)[/green]")

        # 保存結果
        if output_path:
            _save_batch_results(results, output_path, output_format)
            if not ctx.obj["quiet"]:
                console.print(
                    f"\n[bold green]✓ 結果已保存至: {output_path}[/bold green]"
                )
        else:
            # 顯示結果摘要
            if not ctx.obj["quiet"]:
                _display_batch_summary(results)

        # 記錄指標
        metrics_collector = get_metrics_collector()
        metrics_collector.record_counter("batch_query.completed", 1)
        metrics_collector.record_gauge("batch_query.questions_count", len(questions))

        logger.info(f"批次查詢完成，處理了 {len(questions)} 個問題")

    except Exception as e:
        logger.error(f"批次查詢失敗: {e}")
        if not ctx.obj["quiet"]:
            console.print(f"[red]批次查詢失敗: {e}[/red]")
        sys.exit(1)


def _save_batch_results(results: List[dict], output_path: Path, format: str):
    """保存批次查詢結果。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    elif format == "csv":
        import csv

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            if results:
                fieldnames = [
                    "index",
                    "question",
                    "response",
                    "search_type",
                    "query_time",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    writer.writerow(
                        {
                            "index": result.get("index"),
                            "question": result.get("question"),
                            "response": result.get("response", "").replace("\n", " "),
                            "search_type": result.get("search_type"),
                            "query_time": result.get("query_time", 0),
                        }
                    )

    elif format == "markdown":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# 批次查詢結果\n\n")
            for result in results:
                f.write(f"## {result.get('index')}. {result.get('question')}\n\n")
                f.write(f"{result.get('response', '')}\n\n")
                f.write(f"**搜尋類型**: {result.get('search_type')}\n")
                f.write(f"**查詢時間**: {result.get('query_time', 0):.2f} 秒\n\n")
                f.write("---\n\n")

    elif format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"問題 {result.get('index')}: {result.get('question')}\n")
                f.write(f"回答: {result.get('response', '')}\n")
                f.write(f"搜尋類型: {result.get('search_type')}\n")
                f.write(f"查詢時間: {result.get('query_time', 0):.2f} 秒\n")
                f.write("=" * 50 + "\n\n")


def _display_batch_summary(results: List[dict]):
    """顯示批次查詢摘要。"""
    total_time = sum(r.get("query_time", 0) for r in results)
    avg_time = total_time / len(results) if results else 0

    search_type_counts = {}
    for result in results:
        search_type = result.get("search_type", "unknown")
        search_type_counts[search_type] = search_type_counts.get(search_type, 0) + 1

    # 摘要表
    table = Table(title="批次查詢摘要")
    table.add_column("項目", style="cyan")
    table.add_column("值", style="green")

    table.add_row("總問題數", str(len(results)))
    table.add_row("總處理時間", f"{total_time:.2f} 秒")
    table.add_row("平均處理時間", f"{avg_time:.2f} 秒")

    for search_type, count in search_type_counts.items():
        table.add_row(f"{search_type} 查詢", str(count))

    console.print(table)


# 將查詢命令導出到 main.py 使用
__all__ = ["query", "batch_query"]
