#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API 驗證和測試模組

提供 API 端點的自動化驗證和測試功能。
"""

import asyncio
import json
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, ValidationError


@dataclass
class TestResult:
    """測試結果資料類。"""

    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """驗證結果資料類。"""

    endpoint: str
    passed: int
    failed: int
    total: int
    success_rate: float
    results: List[TestResult]


class APIValidator:
    """API 驗證器。

    負責驗證 API 端點的功能性、效能和正確性。
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """初始化驗證器。

        Args:
            base_url: API 基礎 URL
            timeout: 請求超時時間（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.api_prefix = "/api/v1"
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self):
        """異步上下文管理器進入。"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器退出。"""
        await self.client.aclose()

    async def validate_endpoint(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        expected_status: int = 200,
    ) -> TestResult:
        """驗證單一端點。

        Args:
            endpoint: 端點路徑
            method: HTTP 方法
            payload: 請求載荷
            headers: 請求標頭
            expected_status: 預期狀態碼

        Returns:
            測試結果
        """
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            # 準備請求參數
            request_kwargs = {"method": method, "url": url, "headers": headers or {}}

            if payload:
                request_kwargs["json"] = payload

            # 發送請求
            response = await self.client.request(**request_kwargs)
            response_time = time.time() - start_time

            # 檢查狀態碼
            success = response.status_code == expected_status
            error_message = (
                None
                if success
                else f"預期狀態碼 {expected_status}，實際 {response.status_code}"
            )

            # 解析回應資料
            try:
                response_data = response.json()
            except:
                response_data = {"raw_content": response.text}

            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time=response_time,
                success=success,
                error_message=error_message,
                response_data=response_data,
            )

        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error_message=str(e),
            )

    async def validate_health_endpoints(self) -> ValidationResult:
        """驗證健康檢查端點。

        Returns:
            驗證結果
        """
        test_cases = [
            {"endpoint": "/health", "method": "GET", "expected_status": 200},
            {"endpoint": "/health/detailed", "method": "GET", "expected_status": 200},
            {"endpoint": "/health/ready", "method": "GET", "expected_status": 200},
            {"endpoint": "/health/live", "method": "GET", "expected_status": 200},
        ]

        results = []
        for test_case in test_cases:
            result = await self.validate_endpoint(**test_case)
            results.append(result)

        return self._calculate_validation_result("health", results)

    async def validate_index_endpoints(self) -> ValidationResult:
        """驗證索引端點。

        Returns:
            驗證結果
        """
        test_cases = [
            {
                "endpoint": f"{self.api_prefix}/index",
                "method": "POST",
                "payload": {
                    "input_path": "./test_documents",
                    "output_path": "./test_output",
                    "file_types": ["txt"],
                    "batch_size": 16,
                    "force_rebuild": False,
                },
                "expected_status": 202,  # 異步任務應返回 202
            }
        ]

        results = []
        for test_case in test_cases:
            result = await self.validate_endpoint(**test_case)
            results.append(result)

            # 如果索引請求成功，測試狀態查詢
            if result.success and result.response_data:
                task_id = result.response_data.get("data", {}).get("task_id")
                if task_id:
                    status_result = await self.validate_endpoint(
                        f"{self.api_prefix}/index/status/{task_id}",
                        "GET",
                        expected_status=200,
                    )
                    results.append(status_result)

        return self._calculate_validation_result("index", results)

    async def validate_query_endpoints(self) -> ValidationResult:
        """驗證查詢端點。

        Returns:
            驗證結果
        """
        test_cases = [
            {
                "endpoint": f"{self.api_prefix}/query",
                "method": "POST",
                "payload": {
                    "query": "測試查詢",
                    "query_type": "global_search",
                    # "max_tokens": 1000,
                    # "temperature": 0.7,
                },
                "expected_status": 200,
            },
            {
                "endpoint": f"{self.api_prefix}/query/batch",
                "method": "POST",
                "payload": {
                    "queries": [
                        {
                            "id": "test1",
                            "query": "測試查詢1",
                            "query_type": "global_search",
                        },
                        {
                            "id": "test2",
                            "query": "測試查詢2",
                            "query_type": "local_search",
                        },
                    ],
                    # "max_tokens": 1000,
                    # "temperature": 0.7,
                },
                "expected_status": 200,
            },
            {
                "endpoint": f"{self.api_prefix}/query/suggestions",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": f"{self.api_prefix}/query/history",
                "method": "GET",
                "expected_status": 200,
            },
        ]

        results = []
        for test_case in test_cases:
            result = await self.validate_endpoint(**test_case)
            results.append(result)

        return self._calculate_validation_result("query", results)

    async def validate_config_endpoints(self) -> ValidationResult:
        """驗證配置端點。

        Returns:
            驗證結果
        """
        test_cases = [
            {
                "endpoint": f"{self.api_prefix}/config",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": f"{self.api_prefix}/config/schema",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": f"{self.api_prefix}/config/validate",
                "method": "POST",
                "payload": {
                    "llm": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 2000},
                    "embedding": {"model": "BAAI/bge-m3", "batch_size": 32},
                },
                "expected_status": 200,
            },
        ]

        results = []
        for test_case in test_cases:
            result = await self.validate_endpoint(**test_case)
            results.append(result)

        return self._calculate_validation_result("config", results)

    async def validate_monitoring_endpoints(self) -> ValidationResult:
        """驗證監控端點。

        Returns:
            驗證結果
        """
        test_cases = [
            {
                "endpoint": f"{self.api_prefix}/monitoring/metrics",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": f"{self.api_prefix}/monitoring/alerts",
                "method": "GET",
                "expected_status": 200,
            },
        ]

        results = []
        for test_case in test_cases:
            result = await self.validate_endpoint(**test_case)
            results.append(result)

        return self._calculate_validation_result("monitoring", results)

    def _calculate_validation_result(
        self, category: str, results: List[TestResult]
    ) -> ValidationResult:
        """計算驗證結果統計。

        Args:
            category: 類別名稱
            results: 測試結果列表

        Returns:
            驗證結果統計
        """
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0

        return ValidationResult(
            endpoint=category,
            passed=passed,
            failed=failed,
            total=total,
            success_rate=success_rate,
            results=results,
        )

    async def validate_all_endpoints(self) -> Dict[str, ValidationResult]:
        """驗證所有端點。

        Returns:
            所有端點的驗證結果
        """
        print("🚀 開始驗證 API 端點...")

        # 並行執行所有驗證
        tasks = [
            ("health", self.validate_health_endpoints()),
            ("index", self.validate_index_endpoints()),
            ("query", self.validate_query_endpoints()),
            ("config", self.validate_config_endpoints()),
            ("monitoring", self.validate_monitoring_endpoints()),
        ]

        results = {}
        for category, task in tasks:
            try:
                print(f"⏳ 驗證 {category} 端點...")
                result = await task
                results[category] = result

                if result.success_rate >= 80:
                    print(f"✅ {category} 端點驗證通過 ({result.success_rate:.1f}%)")
                else:
                    print(f"❌ {category} 端點驗證失敗 ({result.success_rate:.1f}%)")

            except Exception as e:
                print(f"❌ {category} 端點驗證異常: {e}")
                results[category] = ValidationResult(
                    endpoint=category,
                    passed=0,
                    failed=1,
                    total=1,
                    success_rate=0.0,
                    results=[
                        TestResult(
                            endpoint=category,
                            method="UNKNOWN",
                            status_code=0,
                            response_time=0.0,
                            success=False,
                            error_message=str(e),
                        )
                    ],
                )

        return results

    async def performance_test(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        concurrent_requests: int = 10,
        duration: int = 30,
    ) -> Dict[str, Any]:
        """執行效能測試。

        Args:
            endpoint: 端點路徑
            method: HTTP 方法
            payload: 請求載荷
            concurrent_requests: 並發請求數
            duration: 測試持續時間（秒）

        Returns:
            效能測試結果
        """
        print(f"🔥 開始效能測試: {endpoint}")
        print(f"   並發數: {concurrent_requests}, 持續時間: {duration}秒")

        start_time = time.time()
        end_time = start_time + duration
        results = []

        async def single_request():
            """單次請求。"""
            while time.time() < end_time:
                result = await self.validate_endpoint(endpoint, method, payload)
                results.append(result)
                await asyncio.sleep(0.1)  # 避免過度請求

        # 執行並發請求
        tasks = [single_request() for _ in range(concurrent_requests)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # 計算統計數據
        if results:
            response_times = [r.response_time for r in results]
            success_count = sum(1 for r in results if r.success)

            stats = {
                "total_requests": len(results),
                "successful_requests": success_count,
                "failed_requests": len(results) - success_count,
                "success_rate": (success_count / len(results) * 100),
                "avg_response_time": sum(response_times) / len(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "requests_per_second": len(results) / duration,
                "duration": duration,
                "concurrent_requests": concurrent_requests,
            }
        else:
            stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "min_response_time": 0.0,
                "max_response_time": 0.0,
                "requests_per_second": 0.0,
                "duration": duration,
                "concurrent_requests": concurrent_requests,
            }

        print(f"📊 效能測試結果:")
        print(f"   總請求數: {stats['total_requests']}")
        print(f"   成功率: {stats['success_rate']:.1f}%")
        print(f"   平均回應時間: {stats['avg_response_time']:.3f}秒")
        print(f"   每秒請求數: {stats['requests_per_second']:.1f}")

        return stats

    def generate_validation_report(
        self, results: Dict[str, ValidationResult], output_file: Optional[str] = None
    ) -> str:
        """生成驗證報告。

        Args:
            results: 驗證結果
            output_file: 輸出檔案路徑

        Returns:
            報告內容
        """
        # 計算總體統計
        total_tests = sum(r.total for r in results.values())
        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        overall_success_rate = (
            (total_passed / total_tests * 100) if total_tests > 0 else 0
        )

        # 生成報告內容
        report_lines = [
            "# Chinese GraphRAG API 驗證報告",
            f"",
            f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**API 基礎 URL**: {self.base_url}",
            f"",
            f"## 總體統計",
            f"",
            f"- 總測試數: {total_tests}",
            f"- 通過數: {total_passed}",
            f"- 失敗數: {total_failed}",
            f"- 總體成功率: {overall_success_rate:.1f}%",
            f"",
            f"## 詳細結果",
            f"",
        ]

        for category, result in results.items():
            status_icon = "✅" if result.success_rate >= 80 else "❌"
            report_lines.extend(
                [
                    f"### {status_icon} {category.title()} 端點",
                    f"",
                    f"- 測試數: {result.total}",
                    f"- 通過數: {result.passed}",
                    f"- 失敗數: {result.failed}",
                    f"- 成功率: {result.success_rate:.1f}%",
                    f"",
                ]
            )

            # 添加失敗的測試詳情
            failed_tests = [r for r in result.results if not r.success]
            if failed_tests:
                report_lines.extend([f"#### 失敗的測試:", f""])

                for test in failed_tests:
                    report_lines.extend(
                        [
                            f"- **{test.method} {test.endpoint}**",
                            f"  - 狀態碼: {test.status_code}",
                            f"  - 錯誤訊息: {test.error_message}",
                            f"  - 回應時間: {test.response_time:.3f}秒",
                            f"",
                        ]
                    )

            report_lines.append("")

        # 添加建議
        report_lines.extend([f"## 建議", f"", f"### 成功率分析", f""])

        if overall_success_rate >= 90:
            report_lines.append(
                "🎉 **優秀**: API 整體表現良好，所有主要功能都正常運作。"
            )
        elif overall_success_rate >= 80:
            report_lines.append(
                "👍 **良好**: API 大部分功能正常，建議修復失敗的測試案例。"
            )
        elif overall_success_rate >= 60:
            report_lines.append(
                "⚠️ **需要改進**: API 存在一些問題，建議優先修復關鍵功能。"
            )
        else:
            report_lines.append(
                "🚨 **需要立即處理**: API 存在嚴重問題，建議暫停部署並修復所有錯誤。"
            )

        report_lines.extend(
            [
                f"",
                f"### 下一步行動",
                f"",
                f"1. 修復所有失敗的測試案例",
                f"2. 檢查 API 回應格式是否符合規格",
                f"3. 驗證錯誤處理機制",
                f"4. 執行效能測試",
                f"5. 更新 API 文件",
                f"",
                f"---",
                f"*本報告由 Chinese GraphRAG API 驗證器自動生成*",
            ]
        )

        report = "\\n".join(report_lines)

        # 儲存報告
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"📄 驗證報告已儲存到: {output_file}")

        return report


async def main():
    """主函數，執行完整的 API 驗證。"""
    print("🚀 Chinese GraphRAG API 驗證器")
    print("=" * 50)

    async with APIValidator() as validator:
        try:
            # 執行所有端點驗證
            results = await validator.validate_all_endpoints()

            # 生成驗證報告
            report = validator.generate_validation_report(
                results, "docs/api/validation-report.md"
            )

            # 執行效能測試（可選）
            print("\\n" + "=" * 50)
            print("🔥 執行效能測試...")

            perf_results = await validator.performance_test(
                "/health", concurrent_requests=5, duration=10
            )

            # 顯示總結
            print("\\n" + "=" * 50)
            print("📊 驗證總結:")

            total_tests = sum(r.total for r in results.values())
            total_passed = sum(r.passed for r in results.values())
            overall_success_rate = (
                (total_passed / total_tests * 100) if total_tests > 0 else 0
            )

            print(f"   總測試數: {total_tests}")
            print(f"   通過數: {total_passed}")
            print(f"   總體成功率: {overall_success_rate:.1f}%")

            if overall_success_rate >= 80:
                print("✅ API 驗證通過！")
            else:
                print("❌ API 驗證失敗，請檢查報告！")

        except Exception as e:
            print(f"❌ 驗證過程發生錯誤: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
