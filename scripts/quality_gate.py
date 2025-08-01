#!/usr/bin/env python3
"""
品質閘門檢查器

檢查測試結果、覆蓋率和其他品質指標，決定是否通過品質閘門。
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


class QualityGateChecker:
    """品質閘門檢查器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化品質閘門檢查器
        
        Args:
            config_file: 品質閘門配置檔案路徑
        """
        self.config = self._load_config(config_file)
        self.results = {
            "overall_status": "UNKNOWN",
            "checks": {},
            "metrics": {},
            "violations": []
        }
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """載入品質閘門配置"""
        default_config = {
            "thresholds": {
                "test_pass_rate": 90.0,
                "code_coverage": 80.0,
                "integration_pass_rate": 85.0,
                "chinese_test_pass_rate": 90.0,
                "max_test_duration": 300,  # 秒
                "max_security_issues": 0
            },
            "required_checks": [
                "test_pass_rate",
                "code_coverage",
                "integration_tests",
                "chinese_tests"
            ],
            "optional_checks": [
                "performance_tests",
                "security_scan"
            ],
            "fail_on_missing_optional": False
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                # 合併配置
                default_config.update(user_config)
        
        return default_config
    
    def check_test_results(self, test_results_dir: str) -> bool:
        """檢查測試結果"""
        test_dir = Path(test_results_dir)
        
        # 檢查單元測試
        unit_test_passed = self._check_junit_xml(
            test_dir, "pytest-unit*.xml", "unit_tests"
        )
        
        # 檢查整合測試
        integration_test_passed = self._check_junit_xml(
            test_dir, "pytest-integration*.xml", "integration_tests"
        )
        
        # 檢查中文測試
        chinese_test_passed = self._check_junit_xml(
            test_dir, "pytest-chinese*.xml", "chinese_tests"
        )
        
        # 計算總體通過率
        all_tests_data = self._collect_all_test_data(test_dir)
        overall_pass_rate = self._calculate_overall_pass_rate(all_tests_data)
        
        self.results["metrics"]["overall_pass_rate"] = overall_pass_rate
        self.results["checks"]["test_pass_rate"] = {
            "status": "PASS" if overall_pass_rate >= self.config["thresholds"]["test_pass_rate"] else "FAIL",
            "value": overall_pass_rate,
            "threshold": self.config["thresholds"]["test_pass_rate"],
            "message": f"整體測試通過率: {overall_pass_rate:.1f}%"
        }
        
        if overall_pass_rate < self.config["thresholds"]["test_pass_rate"]:
            self.results["violations"].append(
                f"測試通過率 {overall_pass_rate:.1f}% 低於閾值 {self.config['thresholds']['test_pass_rate']}%"
            )
        
        return (unit_test_passed and integration_test_passed and 
                chinese_test_passed and 
                overall_pass_rate >= self.config["thresholds"]["test_pass_rate"])
    
    def _check_junit_xml(self, test_dir: Path, pattern: str, check_name: str) -> bool:
        """檢查 JUnit XML 測試結果"""
        xml_files = list(test_dir.glob(f"**/{pattern}"))
        
        if not xml_files:
            self.results["checks"][check_name] = {
                "status": "MISSING",
                "message": f"找不到 {pattern} 測試結果檔案"
            }
            return False
        
        total_tests = 0
        failed_tests = 0
        total_time = 0.0
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                tests = int(root.get('tests', 0))
                failures = int(root.get('failures', 0))
                errors = int(root.get('errors', 0))
                time = float(root.get('time', 0))
                
                total_tests += tests
                failed_tests += failures + errors
                total_time += time
                
            except Exception as e:
                self.results["violations"].append(f"解析 {xml_file} 時發生錯誤: {e}")
                return False
        
        if total_tests == 0:
            self.results["checks"][check_name] = {
                "status": "FAIL",
                "message": f"{check_name}: 沒有找到任何測試"
            }
            return False
        
        pass_rate = (total_tests - failed_tests) / total_tests * 100
        
        # 根據測試類型設定不同的閾值
        threshold_key = f"{check_name}_pass_rate"
        if threshold_key not in self.config["thresholds"]:
            threshold_key = "test_pass_rate"
        
        threshold = self.config["thresholds"][threshold_key]
        status = "PASS" if pass_rate >= threshold else "FAIL"
        
        self.results["checks"][check_name] = {
            "status": status,
            "value": pass_rate,
            "threshold": threshold,
            "total_tests": total_tests,
            "failed_tests": failed_tests,
            "duration": total_time,
            "message": f"{check_name}: {pass_rate:.1f}% 通過率 ({total_tests - failed_tests}/{total_tests})"
        }
        
        if status == "FAIL":
            self.results["violations"].append(
                f"{check_name} 通過率 {pass_rate:.1f}% 低於閾值 {threshold}%"
            )
        
        return status == "PASS"
    
    def _collect_all_test_data(self, test_dir: Path) -> List[Dict[str, Any]]:
        """收集所有測試資料"""
        all_data = []
        xml_files = list(test_dir.glob("**/pytest-*.xml"))
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                all_data.append({
                    "file": str(xml_file),
                    "tests": int(root.get('tests', 0)),
                    "failures": int(root.get('failures', 0)),
                    "errors": int(root.get('errors', 0)),
                    "time": float(root.get('time', 0))
                })
            except Exception as e:
                print(f"警告: 無法解析 {xml_file}: {e}")
        
        return all_data
    
    def _calculate_overall_pass_rate(self, test_data: List[Dict[str, Any]]) -> float:
        """計算整體通過率"""
        total_tests = sum(data["tests"] for data in test_data)
        total_failures = sum(data["failures"] + data["errors"] for data in test_data)
        
        if total_tests == 0:
            return 0.0
        
        return (total_tests - total_failures) / total_tests * 100
    
    def check_code_coverage(self, coverage_file: str) -> bool:
        """檢查程式碼覆蓋率"""
        coverage_path = Path(coverage_file)
        
        if not coverage_path.exists():
            self.results["checks"]["code_coverage"] = {
                "status": "MISSING",
                "message": "找不到覆蓋率報告檔案"
            }
            return False
        
        try:
            # 解析 coverage.xml
            tree = ET.parse(coverage_path)
            root = tree.getroot()
            
            # 計算整體覆蓋率
            total_lines = 0
            covered_lines = 0
            
            for package in root.findall('.//package'):
                for class_elem in package.findall('classes/class'):
                    for line in class_elem.findall('lines/line'):
                        total_lines += 1
                        if int(line.get('hits', 0)) > 0:
                            covered_lines += 1
            
            if total_lines == 0:
                coverage_rate = 0.0
            else:
                coverage_rate = covered_lines / total_lines * 100
            
            threshold = self.config["thresholds"]["code_coverage"]
            status = "PASS" if coverage_rate >= threshold else "FAIL"
            
            self.results["metrics"]["code_coverage"] = coverage_rate
            self.results["checks"]["code_coverage"] = {
                "status": status,
                "value": coverage_rate,
                "threshold": threshold,
                "covered_lines": covered_lines,
                "total_lines": total_lines,
                "message": f"程式碼覆蓋率: {coverage_rate:.1f}%"
            }
            
            if status == "FAIL":
                self.results["violations"].append(
                    f"程式碼覆蓋率 {coverage_rate:.1f}% 低於閾值 {threshold}%"
                )
            
            return status == "PASS"
            
        except Exception as e:
            self.results["checks"]["code_coverage"] = {
                "status": "ERROR",
                "message": f"解析覆蓋率檔案時發生錯誤: {e}"
            }
            return False
    
    def check_performance_benchmarks(self, benchmark_file: str) -> bool:
        """檢查效能基準測試"""
        benchmark_path = Path(benchmark_file)
        
        if not benchmark_path.exists():
            self.results["checks"]["performance_tests"] = {
                "status": "MISSING",
                "message": "找不到效能測試結果檔案"
            }
            return not self.config.get("fail_on_missing_optional", False)
        
        try:
            with open(benchmark_path, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)
            
            # 檢查是否有效能回歸
            performance_issues = []
            
            for benchmark in benchmark_data.get("benchmarks", []):
                name = benchmark.get("name", "unknown")
                stats = benchmark.get("stats", {})
                mean_time = stats.get("mean", 0)
                
                # 這裡可以添加效能閾值檢查
                # 例如：某些操作不應該超過特定時間
                if "embedding" in name.lower() and mean_time > 1.0:
                    performance_issues.append(f"{name}: {mean_time:.3f}s > 1.0s")
                elif "query" in name.lower() and mean_time > 0.5:
                    performance_issues.append(f"{name}: {mean_time:.3f}s > 0.5s")
            
            status = "PASS" if not performance_issues else "FAIL"
            
            self.results["checks"]["performance_tests"] = {
                "status": status,
                "benchmarks_count": len(benchmark_data.get("benchmarks", [])),
                "issues": performance_issues,
                "message": f"效能測試: {len(benchmark_data.get('benchmarks', []))} 個基準測試"
            }
            
            if performance_issues:
                self.results["violations"].extend(performance_issues)
            
            return status == "PASS"
            
        except Exception as e:
            self.results["checks"]["performance_tests"] = {
                "status": "ERROR",
                "message": f"解析效能測試檔案時發生錯誤: {e}"
            }
            return False
    
    def check_security_scan(self, security_file: str) -> bool:
        """檢查安全性掃描結果"""
        security_path = Path(security_file)
        
        if not security_path.exists():
            self.results["checks"]["security_scan"] = {
                "status": "MISSING",
                "message": "找不到安全掃描結果檔案"
            }
            return not self.config.get("fail_on_missing_optional", False)
        
        try:
            with open(security_path, 'r', encoding='utf-8') as f:
                security_data = json.load(f)
            
            # 統計安全問題
            high_issues = 0
            medium_issues = 0
            low_issues = 0
            
            for result in security_data.get("results", []):
                severity = result.get("issue_severity", "").lower()
                if severity == "high":
                    high_issues += 1
                elif severity == "medium":
                    medium_issues += 1
                elif severity == "low":
                    low_issues += 1
            
            total_issues = high_issues + medium_issues + low_issues
            max_issues = self.config["thresholds"]["max_security_issues"]
            
            status = "PASS" if total_issues <= max_issues else "FAIL"
            
            self.results["checks"]["security_scan"] = {
                "status": status,
                "total_issues": total_issues,
                "high_issues": high_issues,
                "medium_issues": medium_issues,
                "low_issues": low_issues,
                "threshold": max_issues,
                "message": f"安全掃描: {total_issues} 個問題 (高: {high_issues}, 中: {medium_issues}, 低: {low_issues})"
            }
            
            if status == "FAIL":
                self.results["violations"].append(
                    f"安全問題數量 {total_issues} 超過閾值 {max_issues}"
                )
            
            return status == "PASS"
            
        except Exception as e:
            self.results["checks"]["security_scan"] = {
                "status": "ERROR",
                "message": f"解析安全掃描檔案時發生錯誤: {e}"
            }
            return False
    
    def run_all_checks(self, 
                      test_results_dir: str,
                      coverage_file: str = "coverage.xml",
                      benchmark_file: str = "benchmark.json",
                      security_file: str = "bandit-report.json") -> bool:
        """執行所有品質檢查"""
        
        print("🔍 開始執行品質閘門檢查...")
        
        all_passed = True
        
        # 必要檢查
        for check in self.config["required_checks"]:
            if check == "test_pass_rate" or check in ["unit_tests", "integration_tests", "chinese_tests"]:
                if not self.check_test_results(test_results_dir):
                    all_passed = False
            elif check == "code_coverage":
                if not self.check_code_coverage(coverage_file):
                    all_passed = False
        
        # 可選檢查
        for check in self.config["optional_checks"]:
            if check == "performance_tests":
                if not self.check_performance_benchmarks(benchmark_file):
                    if self.config.get("fail_on_missing_optional", False):
                        all_passed = False
            elif check == "security_scan":
                if not self.check_security_scan(security_file):
                    if self.config.get("fail_on_missing_optional", False):
                        all_passed = False
        
        # 設定整體狀態
        self.results["overall_status"] = "PASS" if all_passed else "FAIL"
        
        return all_passed
    
    def print_results(self):
        """列印檢查結果"""
        print(f"\n📊 品質閘門檢查結果: {self.results['overall_status']}")
        print("=" * 50)
        
        # 列印各項檢查結果
        for check_name, check_data in self.results["checks"].items():
            status = check_data["status"]
            message = check_data.get("message", "")
            
            status_icon = {
                "PASS": "✅",
                "FAIL": "❌", 
                "MISSING": "⚠️",
                "ERROR": "🔥"
            }.get(status, "❓")
            
            print(f"{status_icon} {check_name}: {message}")
            
            if "value" in check_data and "threshold" in check_data:
                print(f"   值: {check_data['value']:.1f}, 閾值: {check_data['threshold']}")
        
        # 列印違規項目
        if self.results["violations"]:
            print(f"\n❌ 發現 {len(self.results['violations'])} 個違規項目:")
            for violation in self.results["violations"]:
                print(f"   • {violation}")
        
        # 列印關鍵指標
        if self.results["metrics"]:
            print(f"\n📈 關鍵指標:")
            for metric, value in self.results["metrics"].items():
                print(f"   • {metric}: {value:.1f}%")
    
    def save_results(self, output_file: str):
        """儲存檢查結果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"📄 檢查結果已儲存到: {output_file}")


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description="品質閘門檢查器")
    parser.add_argument("--test-results-dir", default=".", help="測試結果目錄")
    parser.add_argument("--coverage-file", default="coverage.xml", help="覆蓋率檔案")
    parser.add_argument("--benchmark-file", default="benchmark.json", help="效能測試檔案")
    parser.add_argument("--security-file", default="bandit-report.json", help="安全掃描檔案")
    parser.add_argument("--config", help="品質閘門配置檔案")
    parser.add_argument("--output", default="quality-gate-results.json", help="結果輸出檔案")
    
    args = parser.parse_args()
    
    # 建立檢查器
    checker = QualityGateChecker(args.config)
    
    # 執行檢查
    passed = checker.run_all_checks(
        args.test_results_dir,
        args.coverage_file,
        args.benchmark_file,
        args.security_file
    )
    
    # 列印結果
    checker.print_results()
    
    # 儲存結果
    checker.save_results(args.output)
    
    # 設定退出碼
    if not passed:
        print(f"\n❌ 品質閘門檢查失敗！")
        sys.exit(1)
    else:
        print(f"\n✅ 品質閘門檢查通過！")
        sys.exit(0)


if __name__ == "__main__":
    main()