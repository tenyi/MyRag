#!/usr/bin/env python3
"""
測試自動化執行器

整合所有測試自動化功能的主要執行腳本。
包括測試資料管理、測試執行、品質閘門檢查和結果通知。
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.test_data_manager import TestDataManager
from scripts.quality_gate import QualityGateChecker
from scripts.test_notifier import TestNotifier


class TestAutomationRunner:
    """測試自動化執行器"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化測試自動化執行器
        
        Args:
            config_dir: 配置檔案目錄
        """
        self.config_dir = Path(config_dir)
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "test-results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化各個元件
        self.data_manager = TestDataManager()
        self.quality_checker = QualityGateChecker(
            str(self.config_dir / "quality_gate.yaml")
        )
        self.notifier = TestNotifier(
            str(self.config_dir / "quality_gate.yaml")
        )
    
    def setup_test_environment(self, reset_data: bool = False) -> bool:
        """設定測試環境"""
        print("🚀 設定測試環境...")
        
        try:
            # 初始化測試資料
            if reset_data:
                print("⚠️ 重設測試資料...")
                self.data_manager.reset_all_data(confirm=True)
            
            # 建立測試資料
            print("📄 建立範例文件...")
            documents = self.data_manager.create_sample_documents()
            
            print("⚙️ 建立測試配置...")
            configs = self.data_manager.create_test_configs()
            
            print("🧪 建立測試夾具...")
            fixtures = self.data_manager.create_test_fixtures()
            
            # 驗證資料完整性
            print("✅ 驗證資料完整性...")
            integrity_results = self.data_manager.validate_data_integrity()
            valid_count = sum(integrity_results.values())
            total_count = len(integrity_results)
            
            if valid_count != total_count:
                print(f"⚠️ 資料驗證失敗 ({valid_count}/{total_count})")
                for name, is_valid in integrity_results.items():
                    if not is_valid:
                        print(f"   ❌ {name}")
                return False
            
            print(f"✅ 測試環境設定完成 ({valid_count}/{total_count} 驗證通過)")
            return True
            
        except Exception as e:
            print(f"❌ 測試環境設定失敗: {e}")
            return False
    
    def run_code_quality_checks(self) -> bool:
        """執行程式碼品質檢查"""
        print("🔍 執行程式碼品質檢查...")
        
        checks = [
            ("Black 格式檢查", ["uv", "run", "black", "--check", "--diff", "src/", "tests/"]),
            ("isort 導入排序檢查", ["uv", "run", "isort", "--check-only", "--diff", "src/", "tests/"]),
            ("flake8 程式碼風格檢查", ["uv", "run", "flake8", "src/", "tests/"]),
            ("mypy 型別檢查", ["uv", "run", "mypy", "src/"])
        ]
        
        all_passed = True
        
        for check_name, command in checks:
            print(f"   執行 {check_name}...")
            try:
                result = subprocess.run(
                    command,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"   ✅ {check_name} 通過")
                else:
                    print(f"   ❌ {check_name} 失敗")
                    if result.stdout:
                        print(f"      輸出: {result.stdout[:200]}...")
                    if result.stderr:
                        print(f"      錯誤: {result.stderr[:200]}...")
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                print(f"   ⏰ {check_name} 超時")
                all_passed = False
            except Exception as e:
                print(f"   🔥 {check_name} 執行錯誤: {e}")
                all_passed = False
        
        return all_passed
    
    def run_tests(self, test_types: List[str]) -> bool:
        """執行測試"""
        print("🧪 執行測試...")
        
        test_commands = {
            "unit": [
                "uv", "run", "pytest", "tests/",
                "--cov=src/chinese_graphrag",
                "--cov-report=xml",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--junit-xml=pytest-unit.xml",
                "-m", "not integration and not slow",
                "--tb=short", "-v"
            ],
            "integration": [
                "uv", "run", "pytest", "tests/integration/",
                "--junit-xml=pytest-integration.xml",
                "-m", "integration",
                "--tb=short", "-v",
                "--timeout=300"
            ],
            "chinese": [
                "uv", "run", "pytest", "tests/",
                "--junit-xml=pytest-chinese.xml",
                "-m", "chinese",
                "--tb=short", "-v"
            ],
            "performance": [
                "uv", "run", "pytest", "tests/integration/test_performance.py",
                "--benchmark-json=benchmark.json",
                "--benchmark-only", "-v"
            ]
        }
        
        all_passed = True
        
        for test_type in test_types:
            if test_type not in test_commands:
                print(f"⚠️ 未知的測試類型: {test_type}")
                continue
            
            print(f"   執行 {test_type} 測試...")
            try:
                result = subprocess.run(
                    test_commands[test_type],
                    cwd=self.project_root,
                    timeout=600  # 10 分鐘超時
                )
                
                if result.returncode == 0:
                    print(f"   ✅ {test_type} 測試通過")
                else:
                    print(f"   ❌ {test_type} 測試失敗 (退出碼: {result.returncode})")
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                print(f"   ⏰ {test_type} 測試超時")
                all_passed = False
            except Exception as e:
                print(f"   🔥 {test_type} 測試執行錯誤: {e}")
                all_passed = False
        
        return all_passed
    
    def run_security_scan(self) -> bool:
        """執行安全性掃描"""
        print("🔒 執行安全性掃描...")
        
        try:
            # 安裝 bandit
            subprocess.run(
                ["pip", "install", "bandit[toml]"],
                cwd=self.project_root,
                capture_output=True,
                timeout=120
            )
            
            # 執行掃描
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # bandit 可能會返回非零退出碼但仍然成功生成報告
            if Path(self.project_root / "bandit-report.json").exists():
                print("   ✅ 安全性掃描完成")
                return True
            else:
                print("   ❌ 安全性掃描失敗")
                if result.stderr:
                    print(f"      錯誤: {result.stderr[:200]}...")
                return False
                
        except Exception as e:
            print(f"   🔥 安全性掃描執行錯誤: {e}")
            return False
    
    def generate_test_report(self) -> bool:
        """生成測試報告"""
        print("📊 生成測試報告...")
        
        try:
            result = subprocess.run([
                "uv", "run", "python", "scripts/generate_test_report.py",
                "--output-dir", "test-reports",
                "--format", "html,json,summary",
                "--include-coverage",
                "--include-performance"
            ], cwd=self.project_root, timeout=120)
            
            if result.returncode == 0:
                print("   ✅ 測試報告生成完成")
                return True
            else:
                print("   ❌ 測試報告生成失敗")
                return False
                
        except Exception as e:
            print(f"   🔥 測試報告生成錯誤: {e}")
            return False
    
    def run_quality_gate(self) -> bool:
        """執行品質閘門檢查"""
        print("🚪 執行品質閘門檢查...")
        
        try:
            passed = self.quality_checker.run_all_checks(
                test_results_dir=str(self.project_root),
                coverage_file="coverage.xml",
                benchmark_file="benchmark.json",
                security_file="bandit-report.json"
            )
            
            # 列印結果
            self.quality_checker.print_results()
            
            # 儲存結果
            self.quality_checker.save_results("quality-gate-results.json")
            
            return passed
            
        except Exception as e:
            print(f"   🔥 品質閘門檢查錯誤: {e}")
            return False
    
    def send_notifications(self) -> bool:
        """發送通知"""
        print("📢 發送通知...")
        
        try:
            # 載入品質閘門結果
            self.notifier.load_quality_gate_results("quality-gate-results.json")
            
            # 發送通知
            success = self.notifier.send_notifications()
            
            return success
            
        except Exception as e:
            print(f"   🔥 通知發送錯誤: {e}")
            return False
    
    def cleanup(self):
        """清理臨時檔案"""
        print("🧹 清理臨時檔案...")
        
        try:
            self.data_manager.cleanup_temp_data()
            print("   ✅ 清理完成")
        except Exception as e:
            print(f"   ⚠️ 清理錯誤: {e}")
    
    def run_full_pipeline(self, 
                         test_types: List[str],
                         skip_quality_checks: bool = False,
                         skip_security_scan: bool = False,
                         reset_data: bool = False) -> bool:
        """執行完整的測試自動化管道"""
        print("🚀 開始執行完整的測試自動化管道...")
        start_time = time.time()
        
        try:
            # 1. 設定測試環境
            if not self.setup_test_environment(reset_data):
                return False
            
            # 2. 程式碼品質檢查
            if not skip_quality_checks:
                if not self.run_code_quality_checks():
                    print("⚠️ 程式碼品質檢查失敗，但繼續執行測試...")
            
            # 3. 執行測試
            if not self.run_tests(test_types):
                print("❌ 測試執行失敗")
            
            # 4. 安全性掃描
            if not skip_security_scan:
                if not self.run_security_scan():
                    print("⚠️ 安全性掃描失敗，但繼續執行...")
            
            # 5. 生成測試報告
            if not self.generate_test_report():
                print("⚠️ 測試報告生成失敗，但繼續執行...")
            
            # 6. 品質閘門檢查
            quality_passed = self.run_quality_gate()
            
            # 7. 發送通知
            if not self.send_notifications():
                print("⚠️ 通知發送失敗")
            
            # 8. 清理
            self.cleanup()
            
            # 計算執行時間
            duration = time.time() - start_time
            print(f"\n⏱️ 總執行時間: {duration:.1f} 秒")
            
            if quality_passed:
                print("🎉 測試自動化管道執行成功！")
                return True
            else:
                print("❌ 測試自動化管道執行失敗！")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️ 用戶中斷執行")
            self.cleanup()
            return False
        except Exception as e:
            print(f"\n🔥 管道執行發生未預期錯誤: {e}")
            self.cleanup()
            return False


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description="測試自動化執行器")
    parser.add_argument(
        "--test-types",
        nargs="+",
        default=["unit", "integration", "chinese"],
        choices=["unit", "integration", "chinese", "performance"],
        help="要執行的測試類型"
    )
    parser.add_argument("--config-dir", default="config", help="配置檔案目錄")
    parser.add_argument("--skip-quality-checks", action="store_true", help="跳過程式碼品質檢查")
    parser.add_argument("--skip-security-scan", action="store_true", help="跳過安全性掃描")
    parser.add_argument("--reset-data", action="store_true", help="重設測試資料")
    parser.add_argument("--setup-only", action="store_true", help="僅設定測試環境")
    
    args = parser.parse_args()
    
    # 建立執行器
    runner = TestAutomationRunner(args.config_dir)
    
    if args.setup_only:
        # 僅設定測試環境
        success = runner.setup_test_environment(args.reset_data)
    else:
        # 執行完整管道
        success = runner.run_full_pipeline(
            test_types=args.test_types,
            skip_quality_checks=args.skip_quality_checks,
            skip_security_scan=args.skip_security_scan,
            reset_data=args.reset_data
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()