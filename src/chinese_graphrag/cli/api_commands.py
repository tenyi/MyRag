#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API 相關的 CLI 命令

提供 API 伺服器管理、文件生成和驗證功能。
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from ..monitoring import get_logger

# 建立控制台輸出物件
console = Console()

# 建立日誌器
logger = get_logger(__name__)


@click.group()
def api():
    """API 相關命令。

    管理 REST API 伺服器、生成文件和執行驗證。
    """
    pass


@api.command()
@click.option("--host", "-h", default="0.0.0.0", help="綁定主機地址 (預設: 0.0.0.0)")
@click.option("--port", "-p", type=int, default=8000, help="綁定端口 (預設: 8000)")
@click.option("--reload", is_flag=True, help="啟用自動重載（開發模式）")
@click.option("--workers", "-w", type=int, default=1, help="工作進程數 (預設: 1)")
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
    help="日誌級別 (預設: info)",
)
def server(host: str, port: int, reload: bool, workers: int, log_level: str):
    """啟動 API 伺服器。

    使用範例:

    \b
    chinese-graphrag api server                    # 預設設定啟動
    chinese-graphrag api server --reload           # 開發模式（自動重載）
    chinese-graphrag api server --port 8080        # 指定端口
    chinese-graphrag api server --workers 4        # 多進程模式
    """
    console.print("🚀 [bold]正在啟動 Chinese GraphRAG API 伺服器...[/bold]")
    console.print(f"   地址: [link]http://{host}:{port}[/link]")
    console.print(f"   API 文件: [link]http://{host}:{port}/api/v1/docs[/link]")
    console.print(f"   ReDoc: [link]http://{host}:{port}/api/v1/redoc[/link]")

    if reload:
        console.print("   模式: [yellow]開發模式（自動重載）[/yellow]")
    else:
        console.print(f"   工作進程: {workers}")

    console.print(f"   日誌級別: {log_level}")
    console.print()

    try:
        import uvicorn

        # 啟動伺服器
        uvicorn.run(
            "chinese_graphrag.api.app:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level=log_level,
            access_log=True,
        )

    except ImportError:
        console.print("[red]❌ 需要安裝 uvicorn：[/red]")
        console.print("[yellow]   uv add uvicorn[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"啟動伺服器失敗: {e}")
        console.print(f"[red]❌ 啟動伺服器失敗: {e}[/red]")
        sys.exit(1)


@api.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="docs/api",
    help="文件輸出目錄 (預設: docs/api)",
)
@click.option(
    "--format",
    "formats",
    type=click.Choice(["json", "yaml", "all"]),
    default="all",
    help="OpenAPI 規格格式 (預設: all)",
)
def docs(output: Path, formats: str):
    """生成 API 文件。

    生成完整的 API 文件，包括：
    - OpenAPI 3.0 規格 (JSON/YAML)
    - 客戶端程式碼範例 (Python, JavaScript, cURL)
    - Postman 測試集合
    - 使用說明文件

    使用範例:

    \b
    chinese-graphrag api docs                       # 生成所有文件
    chinese-graphrag api docs --format json        # 只生成 JSON 格式
    chinese-graphrag api docs --output ./api-docs  # 指定輸出目錄
    """
    console.print("📚 [bold]正在生成 API 文件...[/bold]")

    try:
        from ..api.app import create_app
        from ..api.docs import APIDocumentationGenerator

        # 創建 FastAPI 應用程式
        console.print("   正在初始化 API 應用程式...")
        app = create_app()

        # 創建文件生成器
        console.print(f"   輸出目錄: {output}")
        generator = APIDocumentationGenerator(app, str(output))

        # 生成 OpenAPI 規格
        if formats in ["json", "all"]:
            console.print("   生成 OpenAPI JSON 規格...")
            json_path = generator.save_openapi_spec("json")
            console.print(f"   ✅ {json_path}")

        if formats in ["yaml", "all"]:
            console.print("   生成 OpenAPI YAML 規格...")
            yaml_path = generator.save_openapi_spec("yaml")
            console.print(f"   ✅ {yaml_path}")

        if formats == "all":
            # 生成客戶端範例
            console.print("   生成客戶端程式碼範例...")
            examples = generator.generate_client_examples()
            for lang in examples.keys():
                console.print(f"   ✅ {lang} 範例")

            # 生成 Postman 集合
            console.print("   生成 Postman 測試集合...")
            generator.generate_postman_collection()
            console.print("   ✅ Postman 集合")

            # 生成 README
            console.print("   生成說明文件...")
            generator._generate_readme()
            console.print("   ✅ README.md")

        console.print(f"\\n🎉 [bold green]API 文件生成完成！[/bold green]")
        console.print(f"📁 查看文件: {output}")

    except Exception as e:
        logger.error(f"生成 API 文件失敗: {e}")
        console.print(f"[red]❌ 生成 API 文件失敗: {e}[/red]")
        sys.exit(1)


@api.command()
@click.option(
    "--url",
    "-u",
    default="http://localhost:8000",
    help="API 基礎 URL (預設: http://localhost:8000)",
)
@click.option(
    "--report", "-r", type=click.Path(path_type=Path), help="驗證報告檔案路徑"
)
@click.option(
    "--timeout", "-t", type=float, default=30.0, help="請求超時時間（秒） (預設: 30.0)"
)
def validate(url: str, report: Optional[Path], timeout: float):
    """驗證 API 端點。

    測試所有 API 端點的功能性和正確性，包括：
    - 健康檢查端點
    - 索引管理端點
    - 查詢服務端點
    - 配置管理端點
    - 監控管理端點

    使用範例:

    \b
    chinese-graphrag api validate                                   # 驗證本地 API
    chinese-graphrag api validate --url http://api.example.com     # 驗證遠端 API
    chinese-graphrag api validate --report validation.md           # 生成驗證報告
    """
    console.print("🔍 [bold]正在驗證 API 端點...[/bold]")
    console.print(f"   目標 URL: {url}")
    console.print(f"   超時時間: {timeout}秒")

    try:
        import asyncio

        from ..api.validation import APIValidator

        async def run_validation():
            async with APIValidator(url, timeout) as validator:
                # 執行驗證
                results = await validator.validate_all_endpoints()

                # 生成報告
                report_path = report or Path("docs/api/validation-report.md")
                report_content = validator.generate_validation_report(
                    results, str(report_path)
                )

                # 計算總體統計
                total_tests = sum(r.total for r in results.values())
                total_passed = sum(r.passed for r in results.values())
                success_rate = (
                    (total_passed / total_tests * 100) if total_tests > 0 else 0
                )

                # 顯示結果摘要
                console.print(f"\\n📊 [bold]驗證結果摘要:[/bold]")
                console.print(f"   總測試數: {total_tests}")
                console.print(f"   通過數: {total_passed}")
                console.print(f"   失敗數: {total_tests - total_passed}")
                console.print(f"   成功率: {success_rate:.1f}%")

                if report_path.exists():
                    console.print(f"   報告檔案: {report_path}")

                # 顯示各類別結果
                console.print(f"\\n📋 [bold]詳細結果:[/bold]")
                for category, result in results.items():
                    status_icon = "✅" if result.success_rate >= 80 else "❌"
                    console.print(
                        f"   {status_icon} {category.title()}: {result.success_rate:.1f}% ({result.passed}/{result.total})"
                    )

                # 整體評價
                if success_rate >= 90:
                    console.print(
                        "\\n🎉 [bold green]API 驗證通過！系統運作良好。[/bold green]"
                    )
                elif success_rate >= 80:
                    console.print(
                        "\\n👍 [bold yellow]API 大部分功能正常，建議檢查失敗項目。[/bold yellow]"
                    )
                else:
                    console.print(
                        "\\n⚠️ [bold red]API 存在問題，建議優先修復。[/bold red]"
                    )
                    sys.exit(1)

        # 執行異步驗證
        asyncio.run(run_validation())

    except Exception as e:
        logger.error(f"API 驗證失敗: {e}")
        console.print(f"[red]❌ API 驗證失敗: {e}[/red]")
        sys.exit(1)


@api.command()
@click.option(
    "--url",
    "-u",
    default="http://localhost:8000",
    help="API 基礎 URL (預設: http://localhost:8000)",
)
@click.option("--endpoint", "-e", default="/health", help="測試端點 (預設: /health)")
@click.option("--concurrent", "-c", type=int, default=10, help="並發請求數 (預設: 10)")
@click.option(
    "--duration", "-d", type=int, default=30, help="測試持續時間（秒） (預設: 30)"
)
def perf(url: str, endpoint: str, concurrent: int, duration: int):
    """執行 API 效能測試。

    測試 API 端點的效能表現，包括：
    - 回應時間統計
    - 吞吐量測量
    - 錯誤率分析
    - 並發處理能力

    使用範例:

    \b
    chinese-graphrag api perf                                       # 基本效能測試
    chinese-graphrag api perf --concurrent 20 --duration 60        # 高強度測試
    chinese-graphrag api perf --endpoint /api/v1/query             # 測試特定端點
    """
    console.print("🚀 [bold]正在執行 API 效能測試...[/bold]")
    console.print(f"   目標 URL: {url}")
    console.print(f"   測試端點: {endpoint}")
    console.print(f"   並發數: {concurrent}")
    console.print(f"   持續時間: {duration}秒")
    console.print()

    try:
        import asyncio

        from ..api.validation import APIValidator

        async def run_performance_test():
            async with APIValidator(url) as validator:
                results = await validator.performance_test(
                    endpoint=endpoint, concurrent_requests=concurrent, duration=duration
                )

                # 顯示詳細結果
                console.print(f"\\n📊 [bold]效能測試結果:[/bold]")
                console.print(f"   總請求數: {results['total_requests']}")
                console.print(f"   成功請求數: {results['successful_requests']}")
                console.print(f"   失敗請求數: {results['failed_requests']}")
                console.print(f"   成功率: {results['success_rate']:.1f}%")
                console.print(f"   每秒請求數: {results['requests_per_second']:.1f}")
                console.print(f"   平均回應時間: {results['avg_response_time']:.3f}秒")
                console.print(f"   最短回應時間: {results['min_response_time']:.3f}秒")
                console.print(f"   最長回應時間: {results['max_response_time']:.3f}秒")

                # 效能評價
                if results["success_rate"] >= 99 and results["avg_response_time"] < 1.0:
                    console.print("\\n🎉 [bold green]優秀的效能表現！[/bold green]")
                elif (
                    results["success_rate"] >= 95 and results["avg_response_time"] < 3.0
                ):
                    console.print("\\n👍 [bold yellow]良好的效能表現。[/bold yellow]")
                else:
                    console.print("\\n⚠️ [bold red]效能需要改進。[/bold red]")

        # 執行異步效能測試
        asyncio.run(run_performance_test())

    except Exception as e:
        logger.error(f"效能測試失敗: {e}")
        console.print(f"[red]❌ 效能測試失敗: {e}[/red]")
        sys.exit(1)


@api.command()
@click.option(
    "--url",
    "-u",
    default="http://localhost:8000",
    help="API 基礎 URL (預設: http://localhost:8000)",
)
def deploy_check(url: str):
    """執行部署前檢查。

    執行全面的部署前檢查，包括：
    - API 端點驗證
    - 基本效能測試
    - 必要檔案檢查
    - 配置驗證

    這個命令適合在 CI/CD 管道中使用。

    使用範例:

    \b
    chinese-graphrag api deploy-check                               # 檢查本地部署
    chinese-graphrag api deploy-check --url http://staging.com     # 檢查測試環境
    """
    console.print("🔍 [bold]正在執行部署前檢查...[/bold]")
    console.print(f"   目標 URL: {url}")
    console.print()

    try:
        import asyncio

        from ..api.validation import APIValidator

        async def run_deploy_check():
            success = True

            async with APIValidator(url) as validator:
                # 1. API 端點驗證
                console.print("1️⃣ [bold]驗證 API 端點...[/bold]")
                results = await validator.validate_all_endpoints()

                total_tests = sum(r.total for r in results.values())
                total_passed = sum(r.passed for r in results.values())
                success_rate = (
                    (total_passed / total_tests * 100) if total_tests > 0 else 0
                )

                if success_rate >= 80:
                    console.print(f"   ✅ API 驗證通過 ({success_rate:.1f}%)")
                else:
                    console.print(f"   ❌ API 驗證失敗 ({success_rate:.1f}%)")
                    success = False

                # 2. 基本效能測試
                console.print("\\n2️⃣ [bold]執行效能測試...[/bold]")
                perf_results = await validator.performance_test(
                    endpoint="/health", concurrent_requests=5, duration=10
                )

                if perf_results["success_rate"] >= 95:
                    console.print(
                        f"   ✅ 效能測試通過 ({perf_results['success_rate']:.1f}%)"
                    )
                else:
                    console.print(
                        f"   ⚠️ 效能測試成功率偏低 ({perf_results['success_rate']:.1f}%)"
                    )

            # 3. 檢查必要檔案
            console.print("\\n3️⃣ [bold]檢查必要檔案...[/bold]")
            required_files = [
                "src/chinese_graphrag/api/app.py",
                "src/chinese_graphrag/api/models.py",
                "src/chinese_graphrag/config/settings.py",
            ]

            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)

            if not missing_files:
                console.print("   ✅ 所有必要檔案存在")
            else:
                console.print(f"   ❌ 缺少檔案: {missing_files}")
                success = False

            # 4. 整體結果
            console.print("\\n" + "=" * 50)
            if success:
                console.print(
                    "🎉 [bold green]部署前檢查通過！系統準備就緒。[/bold green]"
                )
            else:
                console.print(
                    "❌ [bold red]部署前檢查失敗！請修復問題後重試。[/bold red]"
                )
                sys.exit(1)

        # 執行異步檢查
        asyncio.run(run_deploy_check())

    except Exception as e:
        logger.error(f"部署前檢查失敗: {e}")
        console.print(f"[red]❌ 部署前檢查失敗: {e}[/red]")
        sys.exit(1)
