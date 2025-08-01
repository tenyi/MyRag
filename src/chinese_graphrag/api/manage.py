#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API 管理腳本

提供 API 文件生成、驗證和部署管理功能的命令列工具。
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from .app import create_app
from .docs import APIDocumentationGenerator
from .validation import APIValidator


def generate_docs(output_dir: str = "docs/api"):
    """生成 API 文件。
    
    Args:
        output_dir: 輸出目錄
    """
    print("📚 生成 API 文件...")
    
    # 創建 FastAPI 應用程式
    app = create_app()
    
    # 創建文件生成器
    generator = APIDocumentationGenerator(app, output_dir)
    
    # 生成完整文件
    generator.generate_complete_documentation()
    
    print(f"✅ API 文件生成完成！查看 {output_dir} 目錄")


async def validate_api(base_url: str = "http://localhost:8000", 
                      report_file: Optional[str] = None):
    """驗證 API 端點。
    
    Args:
        base_url: API 基礎 URL
        report_file: 驗證報告檔案路徑
    """
    print("🔍 驗證 API 端點...")
    
    async with APIValidator(base_url) as validator:
        # 執行驗證
        results = await validator.validate_all_endpoints()
        
        # 生成報告
        report_path = report_file or "docs/api/validation-report.md"
        validator.generate_validation_report(results, report_path)
        
        # 計算總體成功率
        total_tests = sum(r.total for r in results.values())
        total_passed = sum(r.passed for r in results.values())
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"✅ API 驗證完成！總體成功率: {success_rate:.1f}%")
        
        if success_rate < 80:
            print("⚠️ 發現問題，請查看驗證報告")
            return False
        
        return True


async def performance_test(base_url: str = "http://localhost:8000",
                          endpoint: str = "/health",
                          concurrent: int = 10,
                          duration: int = 30):
    """執行效能測試。
    
    Args:
        base_url: API 基礎 URL
        endpoint: 測試端點
        concurrent: 並發數
        duration: 測試持續時間
    """
    print("🚀 執行效能測試...")
    
    async with APIValidator(base_url) as validator:
        results = await validator.performance_test(
            endpoint=endpoint,
            concurrent_requests=concurrent,
            duration=duration
        )
        
        print("✅ 效能測試完成！")
        return results


def start_server(host: str = "0.0.0.0", 
                port: int = 8000, 
                reload: bool = False,
                workers: int = 1):
    """啟動 API 伺服器。
    
    Args:
        host: 綁定主機
        port: 綁定端口
        reload: 是否啟用重載
        workers: 工作進程數
    """
    print(f"🚀 啟動 Chinese GraphRAG API 伺服器...")
    print(f"   地址: http://{host}:{port}")
    print(f"   文件: http://{host}:{port}/api/v1/docs")
    
    try:
        import uvicorn
        
        uvicorn.run(
            "chinese_graphrag.api.app:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info"
        )
    except ImportError:
        print("❌ 需要安裝 uvicorn：pip install uvicorn")
        sys.exit(1)


async def deploy_check(base_url: str):
    """部署前檢查。
    
    Args:
        base_url: API 基礎 URL
    """
    print("🔍 執行部署前檢查...")
    
    # 1. 驗證 API 端點
    print("1. 驗證 API 端點...")
    validation_passed = await validate_api(base_url)
    
    if not validation_passed:
        print("❌ API 驗證失敗，部署檢查未通過")
        return False
    
    # 2. 執行基本效能測試
    print("2. 執行效能測試...")
    try:
        perf_results = await performance_test(
            base_url=base_url,
            endpoint="/health",
            concurrent=5,
            duration=10
        )
        
        if perf_results["success_rate"] < 95:
            print("⚠️ 效能測試成功率偏低")
        
    except Exception as e:
        print(f"⚠️ 效能測試失敗: {e}")
    
    # 3. 檢查必要檔案
    print("3. 檢查必要檔案...")
    required_files = [
        "src/chinese_graphrag/api/app.py",
        "src/chinese_graphrag/api/models.py",
        "src/chinese_graphrag/config/settings.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要檔案: {missing_files}")
        return False
    
    print("✅ 部署前檢查通過！")
    return True


def main():
    """主函數。"""
    parser = argparse.ArgumentParser(
        description="Chinese GraphRAG API 管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  %(prog)s docs                           # 生成 API 文件
  %(prog)s validate                       # 驗證 API 端點
  %(prog)s perf --concurrent 20           # 執行效能測試
  %(prog)s server --port 8080             # 啟動伺服器
  %(prog)s deploy-check --url http://...  # 部署前檢查
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # docs 命令
    docs_parser = subparsers.add_parser("docs", help="生成 API 文件")
    docs_parser.add_argument(
        "--output", "-o",
        default="docs/api",
        help="輸出目錄 (預設: docs/api)"
    )
    
    # validate 命令
    validate_parser = subparsers.add_parser("validate", help="驗證 API 端點")
    validate_parser.add_argument(
        "--url", "-u",
        default="http://localhost:8000",
        help="API 基礎 URL (預設: http://localhost:8000)"
    )
    validate_parser.add_argument(
        "--report", "-r",
        help="驗證報告檔案路徑"
    )
    
    # perf 命令
    perf_parser = subparsers.add_parser("perf", help="執行效能測試")
    perf_parser.add_argument(
        "--url", "-u",
        default="http://localhost:8000",
        help="API 基礎 URL (預設: http://localhost:8000)"
    )
    perf_parser.add_argument(
        "--endpoint", "-e",
        default="/health",
        help="測試端點 (預設: /health)"
    )
    perf_parser.add_argument(
        "--concurrent", "-c",
        type=int,
        default=10,
        help="並發數 (預設: 10)"
    )
    perf_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=30,
        help="測試持續時間（秒） (預設: 30)"
    )
    
    # server 命令
    server_parser = subparsers.add_parser("server", help="啟動 API 伺服器")
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="綁定主機 (預設: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="綁定端口 (預設: 8000)"
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="啟用自動重載（開發模式）"
    )
    server_parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="工作進程數 (預設: 1)"
    )
    
    # deploy-check 命令
    deploy_parser = subparsers.add_parser("deploy-check", help="部署前檢查")
    deploy_parser.add_argument(
        "--url", "-u",
        default="http://localhost:8000",
        help="API 基礎 URL (預設: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "docs":
            generate_docs(args.output)
            
        elif args.command == "validate":
            asyncio.run(validate_api(args.url, args.report))
            
        elif args.command == "perf":
            asyncio.run(performance_test(
                args.url, args.endpoint, args.concurrent, args.duration
            ))
            
        elif args.command == "server":
            start_server(args.host, args.port, args.reload, args.workers)
            
        elif args.command == "deploy-check":
            success = asyncio.run(deploy_check(args.url))
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\\n👋 操作已取消")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()