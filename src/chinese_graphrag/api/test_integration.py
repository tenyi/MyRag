#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API 整合測試腳本

執行完整的 API 功能測試，驗證所有端點和功能。
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


async def run_integration_tests():
    """執行整合測試。"""
    console.print("🚀 [bold]Chinese GraphRAG API 整合測試[/bold]")
    console.print("=" * 60)

    test_results = []

    try:
        from chinese_graphrag.api.validation import APIValidator

        # 測試配置
        base_url = "http://localhost:8000"
        timeout = 30.0

        console.print(f"📡 測試目標: {base_url}")
        console.print(f"⏱️ 超時設定: {timeout}秒")
        console.print()

        async with APIValidator(base_url, timeout) as validator:

            # 1. 健康檢查測試
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:

                # 健康檢查
                task = progress.add_task("🔍 測試健康檢查端點...", total=None)
                health_results = await validator.validate_health_endpoints()
                test_results.append(("健康檢查", health_results))
                progress.update(task, completed=1)

                # 配置管理
                task = progress.add_task("⚙️ 測試配置管理端點...", total=None)
                config_results = await validator.validate_config_endpoints()
                test_results.append(("配置管理", config_results))
                progress.update(task, completed=1)

                # 監控管理
                task = progress.add_task("📊 測試監控管理端點...", total=None)
                monitoring_results = await validator.validate_monitoring_endpoints()
                test_results.append(("監控管理", monitoring_results))
                progress.update(task, completed=1)

                # 查詢服務（基本測試）
                task = progress.add_task("🔎 測試查詢服務端點...", total=None)
                query_results = await validator.validate_query_endpoints()
                test_results.append(("查詢服務", query_results))
                progress.update(task, completed=1)

                # 索引管理（基本測試）
                task = progress.add_task("📚 測試索引管理端點...", total=None)
                index_results = await validator.validate_index_endpoints()
                test_results.append(("索引管理", index_results))
                progress.update(task, completed=1)

        # 2. 顯示測試結果
        console.print("\n📋 [bold]測試結果摘要[/bold]")

        # 創建結果表格
        table = Table(title="API 端點測試結果")
        table.add_column("類別", style="cyan", no_wrap=True)
        table.add_column("測試數", justify="center")
        table.add_column("通過數", justify="center", style="green")
        table.add_column("失敗數", justify="center", style="red")
        table.add_column("成功率", justify="center")
        table.add_column("狀態", justify="center")

        total_tests = 0
        total_passed = 0

        for category, result in test_results:
            total_tests += result.total
            total_passed += result.passed

            status = "✅ 通過" if result.success_rate >= 80 else "❌ 失敗"
            success_rate = f"{result.success_rate:.1f}%"

            table.add_row(
                category,
                str(result.total),
                str(result.passed),
                str(result.failed),
                success_rate,
                status,
            )

        # 添加總計行
        overall_success_rate = (
            (total_passed / total_tests * 100) if total_tests > 0 else 0
        )
        overall_status = "✅ 通過" if overall_success_rate >= 80 else "❌ 失敗"

        table.add_section()
        table.add_row(
            "[bold]總計[/bold]",
            f"[bold]{total_tests}[/bold]",
            f"[bold]{total_passed}[/bold]",
            f"[bold]{total_tests - total_passed}[/bold]",
            f"[bold]{overall_success_rate:.1f}%[/bold]",
            f"[bold]{overall_status}[/bold]",
        )

        console.print(table)

        # 3. 效能測試
        console.print("\n🚀 [bold]效能測試[/bold]")

        async with APIValidator(base_url, timeout) as validator:
            perf_results = await validator.performance_test(
                endpoint="/health", concurrent_requests=10, duration=15
            )

            console.print(f"   總請求數: {perf_results['total_requests']}")
            console.print(f"   成功率: {perf_results['success_rate']:.1f}%")
            console.print(f"   平均回應時間: {perf_results['avg_response_time']:.3f}秒")
            console.print(f"   每秒請求數: {perf_results['requests_per_second']:.1f}")

        # 4. 最終評估
        console.print("\n" + "=" * 60)

        if overall_success_rate >= 90:
            console.print(
                "🎉 [bold green]整合測試通過！API 系統運作優秀。[/bold green]"
            )
            console.print("✨ 所有主要功能都正常運作，系統準備就緒。")
            return True
        elif overall_success_rate >= 80:
            console.print("👍 [bold yellow]整合測試大部分通過！[/bold yellow]")
            console.print("⚠️ 建議檢查失敗的測試項目並進行修復。")
            return True
        else:
            console.print("❌ [bold red]整合測試失敗！[/bold red]")
            console.print("🚨 系統存在嚴重問題，需要立即修復。")
            return False

    except Exception as e:
        console.print(f"\n❌ [bold red]整合測試執行失敗: {e}[/bold red]")
        return False


def test_documentation_generation():
    """測試文件生成功能。"""
    console.print("\n📚 [bold]測試文件生成功能[/bold]")

    try:
        from chinese_graphrag.api.app import create_app
        from chinese_graphrag.api.docs import APIDocumentationGenerator

        # 創建測試目錄
        test_output_dir = Path("test_docs")
        test_output_dir.mkdir(exist_ok=True)

        # 創建應用程式和文件生成器
        app = create_app()
        generator = APIDocumentationGenerator(app, str(test_output_dir))

        # 生成文件
        console.print("   生成 OpenAPI 規格...")
        openapi_spec = generator.generate_openapi_spec()
        assert "openapi" in openapi_spec
        console.print("   ✅ OpenAPI 規格生成成功")

        # 生成客戶端範例
        console.print("   生成客戶端範例...")
        examples = generator.generate_client_examples()
        assert "python" in examples
        assert "javascript" in examples
        assert "curl" in examples
        console.print("   ✅ 客戶端範例生成成功")

        # 生成 Postman 集合
        console.print("   生成 Postman 集合...")
        postman_collection = generator.generate_postman_collection()
        assert "info" in postman_collection
        console.print("   ✅ Postman 集合生成成功")

        # 清理測試檔案
        import shutil

        shutil.rmtree(test_output_dir, ignore_errors=True)

        console.print("🎉 [bold green]文件生成功能測試通過！[/bold green]")
        return True

    except Exception as e:
        console.print(f"❌ [bold red]文件生成功能測試失敗: {e}[/bold red]")
        return False


def test_cli_integration():
    """測試 CLI 整合。"""
    console.print("\n🖥️ [bold]測試 CLI 整合[/bold]")

    try:
        # 測試 CLI 命令導入
        from chinese_graphrag.cli.api_commands import api
        from chinese_graphrag.cli.main import cli

        # 檢查命令是否正確註冊
        assert api.name == "api"
        assert len(api.commands) > 0

        console.print("   ✅ API 命令模組導入成功")
        console.print("   ✅ CLI 整合成功")
        console.print("🎉 [bold green]CLI 整合測試通過！[/bold green]")
        return True

    except Exception as e:
        console.print(f"❌ [bold red]CLI 整合測試失敗: {e}[/bold red]")
        return False


async def main():
    """主測試函數。"""
    console.print("[bold blue]Chinese GraphRAG API 完整測試套件[/bold blue]")
    console.print("🧪 執行所有測試以驗證系統完整性")
    console.print()

    test_passed = 0
    test_total = 3

    # 1. 文件生成測試
    if test_documentation_generation():
        test_passed += 1

    # 2. CLI 整合測試
    if test_cli_integration():
        test_passed += 1

    # 3. API 整合測試（需要伺服器運行）
    console.print(
        "\n⚠️ [yellow]API 整合測試需要伺服器運行在 http://localhost:8000[/yellow]"
    )
    console.print("請先執行: chinese-graphrag api server")

    user_input = input("\n是否要執行 API 整合測試？(y/N): ").strip().lower()
    if user_input in ["y", "yes"]:
        if await run_integration_tests():
            test_passed += 1
    else:
        console.print("⏭️ 跳過 API 整合測試")
        test_total -= 1

    # 最終結果
    console.print("\n" + "=" * 80)
    console.print(f"📊 [bold]測試總結: {test_passed}/{test_total} 通過[/bold]")

    if test_passed == test_total:
        console.print(
            "🎉 [bold green]所有測試通過！Chinese GraphRAG API 系統完整且功能正常。[/bold green]"
        )
        return True
    else:
        console.print("❌ [bold red]部分測試失敗，請檢查並修復問題。[/bold red]")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n👋 測試已取消")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ 測試執行失敗: {e}")
        sys.exit(1)
