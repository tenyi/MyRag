#!/usr/bin/env python3
"""
系統測試執行腳本

執行完整的系統測試，包括單元測試、整合測試和效能測試
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import click
from loguru import logger


class SystemTestRunner:
    """系統測試執行器"""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定日誌
        logger.remove()
        logger.add(
            self.test_dir / "system_tests.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="10 MB"
        )
        logger.add(sys.stdout, level="INFO")
        
        self.results = {
            "timestamp": time.time(),
            "test_suites": {},
            "summary": {}
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """執行所有測試"""
        logger.info("🚀 開始執行系統測試")
        
        # 1. 執行單元測試
        self._run_unit_tests()
        
        # 2. 執行整合測試
        self._run_integration_tests()
        
        # 3. 執行中文功能測試
        self._run_chinese_tests()
        
        # 4. 執行效能測試
        self._run_performance_tests()
        
        # 5. 生成測試摘要
        self._generate_summary()
        
        return self.results
    
    def _run_unit_tests(self):
        """執行單元測試"""
        logger.info("🧪 執行單元測試")
        
        try:
            # 執行 pytest 單元測試
            cmd = [
                "uv", "run", "pytest", 
                "tests/",
                "-m", "not integration and not slow",
                "--tb=short",
                "-v",
                "--junit-xml=pytest-unit.xml",
                "--cov=src/chinese_graphrag",
                "--cov-report=xml"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            self.results["test_suites"]["unit_tests"] = {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "return_code": result.returncode,
                "stdout": result.stdout[-1000:] if result.stdout else "",  # 只保留最後1000字符
                "stderr": result.stderr[-1000:] if result.stderr else "",
                "duration": "約5分鐘"
            }
            
            if result.returncode == 0:
                logger.info("✅ 單元測試通過")
            else:
                logger.warning(f"⚠️ 單元測試部分失敗 (返回碼: {result.returncode})")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ 單元測試超時")
            self.results["test_suites"]["unit_tests"] = {
                "status": "TIMEOUT",
                "error": "測試執行超時"
            }
        except Exception as e:
            logger.error(f"❌ 單元測試執行失敗: {e}")
            self.results["test_suites"]["unit_tests"] = {
                "status": "ERROR",
                "error": str(e)
            }
    
    def _run_integration_tests(self):
        """執行整合測試"""
        logger.info("🔗 執行整合測試")
        
        try:
            # 執行整合測試
            cmd = [
                "uv", "run", "pytest",
                "tests/integration/",
                "-v",
                "--tb=short",
                "--junit-xml=pytest-integration.xml",
                "--timeout=300"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            self.results["test_suites"]["integration_tests"] = {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "return_code": result.returncode,
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
                "duration": "約10分鐘"
            }
            
            if result.returncode == 0:
                logger.info("✅ 整合測試通過")
            else:
                logger.warning(f"⚠️ 整合測試部分失敗 (返回碼: {result.returncode})")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ 整合測試超時")
            self.results["test_suites"]["integration_tests"] = {
                "status": "TIMEOUT",
                "error": "測試執行超時"
            }
        except Exception as e:
            logger.error(f"❌ 整合測試執行失敗: {e}")
            self.results["test_suites"]["integration_tests"] = {
                "status": "ERROR",
                "error": str(e)
            }
    
    def _run_chinese_tests(self):
        """執行中文功能測試"""
        logger.info("🇨🇳 執行中文功能測試")
        
        try:
            # 執行中文特定測試
            cmd = [
                "uv", "run", "pytest",
                "tests/",
                "-m", "chinese",
                "-v",
                "--tb=short",
                "--junit-xml=pytest-chinese.xml"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            self.results["test_suites"]["chinese_tests"] = {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "return_code": result.returncode,
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
                "duration": "約5分鐘"
            }
            
            if result.returncode == 0:
                logger.info("✅ 中文功能測試通過")
            else:
                logger.warning(f"⚠️ 中文功能測試部分失敗 (返回碼: {result.returncode})")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ 中文功能測試超時")
            self.results["test_suites"]["chinese_tests"] = {
                "status": "TIMEOUT",
                "error": "測試執行超時"
            }
        except Exception as e:
            logger.error(f"❌ 中文功能測試執行失敗: {e}")
            self.results["test_suites"]["chinese_tests"] = {
                "status": "ERROR",
                "error": str(e)
            }
    
    def _run_performance_tests(self):
        """執行效能測試"""
        logger.info("⚡ 執行效能測試")
        
        try:
            # 執行效能測試
            cmd = [
                "uv", "run", "pytest",
                "tests/integration/test_performance.py",
                "-v",
                "--tb=short",
                "--junit-xml=pytest-performance.xml"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            self.results["test_suites"]["performance_tests"] = {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "return_code": result.returncode,
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
                "duration": "約10分鐘"
            }
            
            if result.returncode == 0:
                logger.info("✅ 效能測試通過")
            else:
                logger.warning(f"⚠️ 效能測試部分失敗 (返回碼: {result.returncode})")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ 效能測試超時")
            self.results["test_suites"]["performance_tests"] = {
                "status": "TIMEOUT",
                "error": "測試執行超時"
            }
        except Exception as e:
            logger.error(f"❌ 效能測試執行失敗: {e}")
            self.results["test_suites"]["performance_tests"] = {
                "status": "ERROR",
                "error": str(e)
            }
    
    def _generate_summary(self):
        """生成測試摘要"""
        logger.info("📊 生成測試摘要")
        
        test_suites = self.results["test_suites"]
        total_suites = len(test_suites)
        passed_suites = sum(1 for suite in test_suites.values() if suite.get("status") == "PASS")
        failed_suites = sum(1 for suite in test_suites.values() if suite.get("status") == "FAIL")
        error_suites = sum(1 for suite in test_suites.values() if suite.get("status") in ["ERROR", "TIMEOUT"])
        
        self.results["summary"] = {
            "total_test_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": failed_suites,
            "error_suites": error_suites,
            "success_rate": (passed_suites / total_suites * 100) if total_suites > 0 else 0,
            "overall_status": "PASS" if failed_suites == 0 and error_suites == 0 else "FAIL"
        }
        
        summary = self.results["summary"]
        logger.info("📊 測試摘要:")
        logger.info(f"   總測試套件: {total_suites}")
        logger.info(f"   通過: {passed_suites}")
        logger.info(f"   失敗: {failed_suites}")
        logger.info(f"   錯誤: {error_suites}")
        logger.info(f"   成功率: {summary['success_rate']:.1f}%")
        logger.info(f"   整體狀態: {summary['overall_status']}")
    
    def save_results(self, output_path: Path):
        """儲存測試結果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 測試結果已儲存至: {output_path}")


def run_basic_system_check():
    """執行基本系統檢查"""
    logger.info("🔍 執行基本系統檢查")
    
    checks = {
        "python_version": sys.version_info >= (3, 11),
        "uv_available": False,
        "core_modules": False,
        "config_files": False
    }
    
    # 檢查 uv
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        checks["uv_available"] = result.returncode == 0
    except FileNotFoundError:
        pass
    
    # 檢查核心模組
    try:
        from src.chinese_graphrag.processors.chinese_text_processor import ChineseTextProcessor
        from src.chinese_graphrag.config.loader import ConfigLoader
        checks["core_modules"] = True
    except ImportError:
        pass
    
    # 檢查配置檔案
    config_files = [
        Path("config/settings.yaml"),
        Path("pyproject.toml"),
        Path("README.md")
    ]
    checks["config_files"] = all(f.exists() for f in config_files)
    
    # 輸出檢查結果
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"   {status_icon} {check}: {'通過' if status else '失敗'}")
    
    all_passed = all(checks.values())
    if all_passed:
        logger.info("✅ 基本系統檢查通過")
    else:
        logger.warning("⚠️ 基本系統檢查發現問題")
    
    return all_passed


@click.command()
@click.option("--test-dir", type=click.Path(), default="./system_test_results",
              help="測試結果目錄")
@click.option("--output", type=click.Path(), default="system_test_results.json",
              help="結果輸出檔案")
@click.option("--skip-slow", is_flag=True, help="跳過慢速測試")
@click.option("--basic-check-only", is_flag=True, help="只執行基本檢查")
@click.option("--verbose", "-v", is_flag=True, help="詳細輸出")
def main(test_dir: str, output: str, skip_slow: bool, basic_check_only: bool, verbose: bool):
    """執行系統測試"""
    
    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    # 執行基本系統檢查
    basic_check_passed = run_basic_system_check()
    
    if basic_check_only:
        if basic_check_passed:
            logger.success("🎉 基本系統檢查通過！")
            sys.exit(0)
        else:
            logger.error("❌ 基本系統檢查失敗！")
            sys.exit(1)
    
    if not basic_check_passed:
        logger.warning("⚠️ 基本系統檢查未完全通過，但繼續執行測試")
    
    # 建立測試目錄
    test_dir_path = Path(test_dir)
    test_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 執行系統測試
    runner = SystemTestRunner(test_dir_path)
    results = runner.run_all_tests()
    
    # 儲存結果
    output_path = test_dir_path / output
    runner.save_results(output_path)
    
    # 輸出最終結果
    summary = results["summary"]
    if summary["overall_status"] == "PASS":
        logger.success(f"🎉 系統測試通過！成功率: {summary['success_rate']:.1f}%")
        sys.exit(0)
    else:
        logger.error(f"❌ 系統測試失敗！成功率: {summary['success_rate']:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()