"""
測試資料一致性檢查系統
"""

import hashlib
import json
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from src.chinese_graphrag.exceptions.consistency import (
    ConsistencyChecker,
    ConsistencyIssue,
    ConsistencyLevel,
    ConsistencyReport,
    DataConsistencyManager,
    FileSystemChecker,
    IndexChecker,
    IssueType,
    MetadataChecker,
    Severity,
    VectorStoreChecker,
    get_consistency_manager,
)


class TestConsistencyIssue:
    """測試一致性問題"""

    def test_issue_creation(self):
        """測試問題建立"""
        now = datetime.now()
        issue = ConsistencyIssue(
            issue_id="issue_001",
            issue_type=IssueType.MISSING_FILE,
            severity=Severity.ERROR,
            description="檔案不存在",
            affected_resource="/test/missing.txt",
            details={"expected_path": "/test/missing.txt"},
            suggested_action="檢查檔案位置",
            timestamp=now,
            auto_fixable=True,
        )

        assert issue.issue_id == "issue_001"
        assert issue.issue_type == IssueType.MISSING_FILE
        assert issue.severity == Severity.ERROR
        assert issue.description == "檔案不存在"
        assert issue.auto_fixable is True

    def test_issue_serialization(self):
        """測試問題序列化"""
        now = datetime.now()
        issue = ConsistencyIssue(
            issue_id="serialize_001",
            issue_type=IssueType.HASH_MISMATCH,
            severity=Severity.WARNING,
            description="雜湊不匹配",
            affected_resource="/test/file.txt",
            details={"expected": "abc123", "actual": "def456"},
            suggested_action="重新計算雜湊",
            timestamp=now,
        )

        # 轉換為字典
        data = issue.to_dict()
        assert data["issue_id"] == "serialize_001"
        assert data["issue_type"] == "hash_mismatch"
        assert data["severity"] == "warning"
        assert data["timestamp"] == now.isoformat()
        assert data["details"]["expected"] == "abc123"


class TestConsistencyReport:
    """測試一致性報告"""

    def test_report_creation(self):
        """測試報告建立"""
        now = datetime.now()

        issues = [
            ConsistencyIssue(
                issue_id="test_001",
                issue_type=IssueType.MISSING_FILE,
                severity=Severity.ERROR,
                description="測試問題1",
                affected_resource="resource1",
                details={},
                suggested_action="修復1",
                timestamp=now,
            ),
            ConsistencyIssue(
                issue_id="test_002",
                issue_type=IssueType.HASH_MISMATCH,
                severity=Severity.WARNING,
                description="測試問題2",
                affected_resource="resource2",
                details={},
                suggested_action="修復2",
                timestamp=now,
            ),
        ]

        issues_by_type = {IssueType.MISSING_FILE: 1, IssueType.HASH_MISMATCH: 1}

        issues_by_severity = {Severity.ERROR: 1, Severity.WARNING: 1}

        report = ConsistencyReport(
            report_id="report_001",
            timestamp=now,
            consistency_level=ConsistencyLevel.BASIC,
            total_checked=10,
            issues_found=2,
            issues_by_type=issues_by_type,
            issues_by_severity=issues_by_severity,
            issues=issues,
            execution_time_seconds=1.5,
        )

        assert report.report_id == "report_001"
        assert report.total_checked == 10
        assert report.issues_found == 2
        assert len(report.issues) == 2
        assert report.execution_time_seconds == 1.5

    def test_report_serialization(self):
        """測試報告序列化"""
        now = datetime.now()

        issue = ConsistencyIssue(
            issue_id="serialize",
            issue_type=IssueType.ORPHANED_INDEX,
            severity=Severity.INFO,
            description="序列化測試",
            affected_resource="test_resource",
            details={},
            suggested_action="測試動作",
            timestamp=now,
        )

        report = ConsistencyReport(
            report_id="serialize_report",
            timestamp=now,
            consistency_level=ConsistencyLevel.THOROUGH,
            total_checked=5,
            issues_found=1,
            issues_by_type={IssueType.ORPHANED_INDEX: 1},
            issues_by_severity={Severity.INFO: 1},
            issues=[issue],
            execution_time_seconds=0.5,
        )

        # 轉換為字典
        data = report.to_dict()
        assert data["report_id"] == "serialize_report"
        assert data["consistency_level"] == "thorough"
        assert data["timestamp"] == now.isoformat()
        assert len(data["issues"]) == 1


class TestFileSystemChecker:
    """測試檔案系統檢查器"""

    def test_checker_creation(self):
        """測試檢查器建立"""
        checker = FileSystemChecker()
        assert checker.name == "檔案系統"
        assert "檔案系統" in checker.get_description()

    def test_check_existing_files(self, temp_dir):
        """測試檢查存在的檔案"""
        checker = FileSystemChecker()

        # 建立測試檔案
        test_file = temp_dir / "test_file.txt"
        test_content = "測試檔案內容"

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        # 計算檔案雜湊
        hasher = hashlib.sha256()
        hasher.update(test_content.encode("utf-8"))
        content_hash = hasher.hexdigest()

        # 準備檔案註冊表
        file_registry = {
            str(test_file): {
                "size_bytes": len(test_content.encode("utf-8")),
                "content_hash": content_hash,
            }
        }

        context = {"file_registry": file_registry, "data_directory": str(temp_dir)}

        # 執行檢查
        issues = checker.check(context)

        # 正常檔案不應該有問題
        assert len(issues) == 0

    def test_check_missing_files(self, temp_dir):
        """測試檢查缺失檔案"""
        checker = FileSystemChecker()

        # 準備不存在檔案的註冊表
        missing_file = str(temp_dir / "missing.txt")
        file_registry = {missing_file: {"size_bytes": 100, "content_hash": "abc123"}}

        context = {"file_registry": file_registry, "data_directory": str(temp_dir)}

        # 執行檢查
        issues = checker.check(context)

        # 應該發現缺失檔案問題
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.MISSING_FILE
        assert issues[0].severity == Severity.ERROR
        assert missing_file in issues[0].description

    def test_check_size_mismatch(self, temp_dir):
        """測試檢查檔案大小不匹配"""
        checker = FileSystemChecker()

        # 建立測試檔案
        test_file = temp_dir / "size_test.txt"
        test_content = "實際內容"

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        # 準備錯誤大小的註冊表
        file_registry = {
            str(test_file): {
                "size_bytes": 999,  # 錯誤的大小
                # 不包含 content_hash，這樣就不會檢查雜湊
            }
        }

        context = {"file_registry": file_registry}

        # 執行檢查
        issues = checker.check(context)

        # 應該發現大小不匹配問題
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.SIZE_MISMATCH
        assert issues[0].severity == Severity.WARNING
        assert issues[0].auto_fixable is True

    def test_check_hash_mismatch(self, temp_dir):
        """測試檢查雜湊不匹配"""
        checker = FileSystemChecker()

        # 建立測試檔案
        test_file = temp_dir / "hash_test.txt"
        test_content = "測試雜湊內容"

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        # 準備錯誤雜湊的註冊表
        file_registry = {
            str(test_file): {
                "size_bytes": len(test_content.encode("utf-8")),
                "content_hash": "wrong_hash_value",
            }
        }

        context = {"file_registry": file_registry}

        # 執行檢查
        issues = checker.check(context)

        # 應該發現雜湊不匹配問題
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.HASH_MISMATCH
        assert issues[0].severity == Severity.ERROR
        assert "雜湊不匹配" in issues[0].description


class TestIndexChecker:
    """測試索引檢查器"""

    def test_checker_creation(self):
        """測試檢查器建立"""
        checker = IndexChecker()
        assert checker.name == "索引"
        assert "索引" in checker.get_description()

    def test_check_missing_index(self):
        """測試檢查缺失索引"""
        checker = IndexChecker()

        # 準備有檔案但沒有索引的情況
        file_registry = {
            "/test/file1.txt": {"document_id": "doc_1"},
            "/test/file2.txt": {"document_id": "doc_2"},
        }

        index_registry = {
            "doc_1": {"source_file": "/test/file1.txt"}
            # 缺少 doc_2 的索引
        }

        context = {"file_registry": file_registry, "index_registry": index_registry}

        # 執行檢查
        issues = checker.check(context)

        # 應該發現缺失索引問題
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.MISSING_INDEX
        assert issues[0].severity == Severity.WARNING
        assert issues[0].auto_fixable is True

    def test_check_orphaned_index(self):
        """測試檢查孤立索引"""
        checker = IndexChecker()

        # 準備有索引但沒有對應檔案的情況
        file_registry = {"/test/file1.txt": {"document_id": "doc_1"}}

        index_registry = {
            "doc_1": {"source_file": "/test/file1.txt"},
            "doc_2": {"source_file": "/test/missing.txt"},  # 對應檔案不存在
        }

        context = {"file_registry": file_registry, "index_registry": index_registry}

        # 執行檢查
        issues = checker.check(context)

        # 應該發現孤立索引問題
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.ORPHANED_INDEX
        assert issues[0].severity == Severity.WARNING
        assert issues[0].auto_fixable is True


class TestVectorStoreChecker:
    """測試向量儲存檢查器"""

    def test_checker_creation(self):
        """測試檢查器建立"""
        checker = VectorStoreChecker()
        assert checker.name == "向量儲存"

    def test_check_no_vector_store(self):
        """測試沒有向量儲存的情況"""
        checker = VectorStoreChecker()

        context = {"vector_store": None}

        # 執行檢查
        issues = checker.check(context)

        # 沒有向量儲存時不應該有問題
        assert len(issues) == 0

    def test_check_missing_vectors(self):
        """測試檢查缺失向量"""
        checker = VectorStoreChecker()

        # 模擬向量儲存
        mock_vector_store = Mock()
        mock_vector_store.list_document_ids.return_value = ["doc_1"]  # 只有一個向量
        # 確保 get_vector 方法不會被調用或返回 None
        mock_vector_store.get_vector.return_value = None

        index_registry = {
            "doc_1": {"source_file": "/test/file1.txt"},
            "doc_2": {"source_file": "/test/file2.txt"},  # 缺少向量
        }

        context = {"vector_store": mock_vector_store, "index_registry": index_registry}

        # 執行檢查
        issues = checker.check(context)

        # 應該發現缺失向量問題
        missing_issues = [
            issue
            for issue in issues
            if issue.issue_type == IssueType.MISSING_INDEX
            and "doc_2" in issue.description
        ]
        assert len(missing_issues) == 1
        assert missing_issues[0].auto_fixable is True

    def test_check_orphaned_vectors(self):
        """測試檢查孤立向量"""
        checker = VectorStoreChecker()

        # 模擬向量儲存
        mock_vector_store = Mock()
        mock_vector_store.list_document_ids.return_value = ["doc_1", "doc_2"]
        # 確保 get_vector 方法不會被調用或返回 None
        mock_vector_store.get_vector.return_value = None

        index_registry = {
            "doc_1": {"source_file": "/test/file1.txt"}
            # 缺少 doc_2 的索引
        }

        context = {"vector_store": mock_vector_store, "index_registry": index_registry}

        # 執行檢查
        issues = checker.check(context)

        # 應該發現孤立向量問題
        orphaned_issues = [
            issue
            for issue in issues
            if issue.issue_type == IssueType.ORPHANED_INDEX
            and "doc_2" in issue.description
        ]
        assert len(orphaned_issues) == 1
        assert orphaned_issues[0].auto_fixable is True

    def test_check_vector_dimension(self):
        """測試檢查向量維度"""
        checker = VectorStoreChecker()

        # 模擬向量儲存
        mock_vector_store = Mock()
        mock_vector_store.list_document_ids.return_value = ["doc_1"]
        mock_vector_store.get_vector.return_value = [0.1] * 512  # 錯誤維度

        index_registry = {"doc_1": {"source_file": "/test/file1.txt"}}

        context = {
            "vector_store": mock_vector_store,
            "index_registry": index_registry,
            "expected_vector_dimension": 768,  # 期望維度
        }

        # 執行檢查
        issues = checker.check(context)

        # 應該發現向量維度問題
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.VECTOR_CORRUPTION
        assert issues[0].severity == Severity.ERROR
        assert "維度不正確" in issues[0].description


class TestMetadataChecker:
    """測試元資料檢查器"""

    def test_checker_creation(self):
        """測試檢查器建立"""
        checker = MetadataChecker()
        assert checker.name == "元資料"

    def test_check_missing_fields(self):
        """測試檢查缺失欄位"""
        checker = MetadataChecker()

        # 準備缺少必要欄位的索引註冊表
        index_registry = {
            "doc_1": {
                "source_file": "/test/file1.txt",
                "created_time": "2024-01-01T00:00:00",
                # 缺少 content_hash
            },
            "doc_2": {
                "source_file": "/test/file2.txt",
                "content_hash": "abc123",
                # 缺少 created_time
            },
        }

        context = {"index_registry": index_registry}

        # 執行檢查
        issues = checker.check(context)

        # 應該發現兩個缺失欄位問題
        assert len(issues) == 2
        assert all(issue.issue_type == IssueType.SCHEMA_VIOLATION for issue in issues)
        assert all(issue.auto_fixable is True for issue in issues)

    def test_check_invalid_timestamp(self):
        """測試檢查無效時間戳"""
        checker = MetadataChecker()

        # 準備有無效時間戳的索引註冊表
        index_registry = {
            "doc_1": {
                "source_file": "/test/file1.txt",
                "created_time": "invalid_timestamp",
                "content_hash": "abc123",
            },
            "doc_2": {
                "source_file": "/test/file2.txt",
                "created_time": "2024-01-01T00:00:00Z",  # 有效時間戳
                "content_hash": "def456",
            },
        }

        context = {"index_registry": index_registry}

        # 執行檢查
        issues = checker.check(context)

        # 應該發現一個時間戳問題
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.METADATA_CORRUPTION
        assert issues[0].severity == Severity.WARNING
        assert issues[0].auto_fixable is True
        assert "時間戳格式不正確" in issues[0].description


class TestDataConsistencyManager:
    """測試資料一致性管理器"""

    def test_manager_creation(self):
        """測試管理器建立"""
        manager = DataConsistencyManager()

        assert len(manager.checkers) == 4  # 預設註冊4個檢查器
        assert len(manager.reports) == 0

    def test_register_checker(self):
        """測試註冊檢查器"""
        manager = DataConsistencyManager()

        class CustomChecker(ConsistencyChecker):
            def __init__(self):
                super().__init__("自訂檢查器")

            def check(self, context):
                return []

        initial_count = len(manager.checkers)
        custom_checker = CustomChecker()
        manager.register_checker(custom_checker)

        assert len(manager.checkers) == initial_count + 1
        assert custom_checker in manager.checkers

    def test_run_consistency_check(self, temp_dir):
        """測試執行一致性檢查"""
        manager = DataConsistencyManager()

        # 準備測試檔案
        test_file = temp_dir / "consistency_test.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("一致性檢查測試")

        # 準備上下文
        context = {
            "file_registry": {
                str(test_file): {
                    "size_bytes": 100,  # 錯誤的大小
                    "content_hash": "wrong_hash",
                }
            },
            "index_registry": {},
            "vector_store": None,
        }

        # 執行檢查
        report = manager.run_consistency_check(context, ConsistencyLevel.BASIC)

        assert report is not None
        assert report.consistency_level == ConsistencyLevel.BASIC
        assert report.total_checked >= 1
        assert report.issues_found >= 1  # 應該發現大小和雜湊問題
        assert len(report.issues) == report.issues_found

    def test_check_with_filters(self):
        """測試帶過濾器的檢查"""
        manager = DataConsistencyManager()

        # 只執行檔案系統檢查器
        context = {"file_registry": {}, "index_registry": {}}

        report = manager.run_consistency_check(context, checkers=["檔案系統"])

        assert report.total_checked == 1  # 只執行了一個檢查器

    def test_checker_error_handling(self):
        """測試檢查器錯誤處理"""
        manager = DataConsistencyManager()

        # 建立會拋出異常的檢查器
        class ErrorChecker(ConsistencyChecker):
            def __init__(self):
                super().__init__("錯誤檢查器")

            def check(self, context):
                raise Exception("檢查器故障")

        error_checker = ErrorChecker()
        manager.register_checker(error_checker)

        # 執行檢查
        report = manager.run_consistency_check({})

        # 應該包含檢查器錯誤問題
        error_issues = [
            issue
            for issue in report.issues
            if issue.issue_type == IssueType.METADATA_CORRUPTION
            and "檢查器執行失敗" in issue.description
        ]
        assert len(error_issues) >= 1

    def test_auto_fix_issues(self):
        """測試自動修復問題"""
        manager = DataConsistencyManager()

        # 建立可修復的問題
        fixable_issue = ConsistencyIssue(
            issue_id="fixable_001",
            issue_type=IssueType.ORPHANED_INDEX,
            severity=Severity.WARNING,
            description="可修復問題",
            affected_resource="test_resource",
            details={},
            suggested_action="自動修復",
            timestamp=datetime.now(),
            auto_fixable=True,
        )

        # 建立不可修復的問題
        unfixable_issue = ConsistencyIssue(
            issue_id="unfixable_001",
            issue_type=IssueType.VECTOR_CORRUPTION,
            severity=Severity.ERROR,
            description="不可修復問題",
            affected_resource="test_resource",
            details={},
            suggested_action="手動修復",
            timestamp=datetime.now(),
            auto_fixable=False,
        )

        # 建立測試報告
        report = ConsistencyReport(
            report_id="test_report",
            timestamp=datetime.now(),
            consistency_level=ConsistencyLevel.BASIC,
            total_checked=1,
            issues_found=2,
            issues_by_type={},
            issues_by_severity={},
            issues=[fixable_issue, unfixable_issue],
            execution_time_seconds=1.0,
        )

        # 執行自動修復
        results = manager.auto_fix_issues(report)

        assert results["total_fixable"] == 1
        assert results["fixed"] == 1
        assert results["failed"] == 0

    def test_report_history_management(self):
        """測試報告歷史管理"""
        manager = DataConsistencyManager()

        # 建立多個報告
        for i in range(55):  # 超過限制數量
            report = ConsistencyReport(
                report_id=f"report_{i}",
                timestamp=datetime.now(),
                consistency_level=ConsistencyLevel.BASIC,
                total_checked=1,
                issues_found=0,
                issues_by_type={},
                issues_by_severity={},
                issues=[],
                execution_time_seconds=0.1,
            )
            manager.reports.append(report)

        # 執行新檢查觸發歷史限制
        manager.run_consistency_check({})

        # 驗證報告數量被限制
        assert len(manager.reports) == 50

    def test_export_report(self, temp_dir):
        """測試匯出報告"""
        manager = DataConsistencyManager()

        # 建立測試報告
        issue = ConsistencyIssue(
            issue_id="export_test",
            issue_type=IssueType.MISSING_FILE,
            severity=Severity.ERROR,
            description="匯出測試問題",
            affected_resource="/test/export.txt",
            details={},
            suggested_action="測試動作",
            timestamp=datetime.now(),
        )

        report = ConsistencyReport(
            report_id="export_report",
            timestamp=datetime.now(),
            consistency_level=ConsistencyLevel.BASIC,
            total_checked=1,
            issues_found=1,
            issues_by_type={IssueType.MISSING_FILE: 1},
            issues_by_severity={Severity.ERROR: 1},
            issues=[issue],
            execution_time_seconds=0.5,
        )

        # 匯出報告
        export_path = temp_dir / "exported_report.json"
        manager.export_report(report, export_path)

        # 驗證檔案被建立
        assert export_path.exists()

        # 驗證內容
        with open(export_path, "r", encoding="utf-8") as f:
            exported_data = json.load(f)

        assert exported_data["report_id"] == "export_report"
        assert exported_data["issues_found"] == 1
        assert len(exported_data["issues"]) == 1


class TestGlobalConsistencyManager:
    """測試全域一致性管理器"""

    def test_get_consistency_manager(self):
        """測試獲取全域一致性管理器"""
        manager1 = get_consistency_manager()
        manager2 = get_consistency_manager()

        # 驗證單例模式
        assert manager1 is manager2
        assert isinstance(manager1, DataConsistencyManager)


@pytest.mark.integration
class TestConsistencyIntegration:
    """測試一致性檢查整合"""

    def test_full_consistency_workflow(self, temp_dir):
        """測試完整一致性檢查工作流程"""
        manager = DataConsistencyManager()

        # 第一階段：建立測試資料
        test_files = []
        file_registry = {}
        index_registry = {}

        for i in range(3):
            # 建立實際檔案
            file_path = temp_dir / f"test_{i}.txt"
            content = f"測試檔案 {i} 的內容"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            test_files.append(file_path)

            # 計算正確的元資料
            hasher = hashlib.sha256()
            hasher.update(content.encode("utf-8"))
            content_hash = hasher.hexdigest()

            # 建立檔案註冊表
            file_registry[str(file_path)] = {
                "size_bytes": len(content.encode("utf-8")),
                "content_hash": content_hash,
                "document_id": f"doc_{i}",
            }

            # 建立索引註冊表
            index_registry[f"doc_{i}"] = {
                "source_file": str(file_path),
                "created_time": datetime.now().isoformat(),
                "content_hash": content_hash,
            }

        # 建立模擬向量儲存
        mock_vector_store = Mock()
        mock_vector_store.list_document_ids.return_value = ["doc_0", "doc_1", "doc_2"]
        mock_vector_store.get_vector.return_value = [0.1] * 768

        # 第二階段：執行完整檢查
        context = {
            "file_registry": file_registry,
            "index_registry": index_registry,
            "vector_store": mock_vector_store,
            "expected_vector_dimension": 768,
            "data_directory": str(temp_dir),
        }

        report = manager.run_consistency_check(context, ConsistencyLevel.COMPREHENSIVE)

        # 驗證檢查結果（所有資料一致，不應該有問題）
        assert report.issues_found == 0
        assert report.total_checked == 4  # 4個檢查器

        # 第三階段：引入不一致問題
        # 刪除一個檔案但保留其註冊資訊
        test_files[0].unlink()

        # 修改另一個檔案但不更新註冊資訊
        with open(test_files[1], "a", encoding="utf-8") as f:
            f.write("\n修改內容")

        # 添加孤立索引
        index_registry["orphaned_doc"] = {
            "source_file": "/non/existent/file.txt",
            "created_time": datetime.now().isoformat(),
            "content_hash": "orphaned_hash",
        }

        # 第四階段：再次執行檢查
        report = manager.run_consistency_check(context, ConsistencyLevel.COMPREHENSIVE)

        # 驗證發現問題
        assert report.issues_found > 0

        # 驗證問題類型
        issue_types = [issue.issue_type for issue in report.issues]
        assert IssueType.MISSING_FILE in issue_types  # 刪除的檔案
        assert IssueType.HASH_MISMATCH in issue_types  # 修改的檔案
        assert IssueType.ORPHANED_INDEX in issue_types  # 孤立索引

        # 第五階段：嘗試自動修復
        auto_fix_results = manager.auto_fix_issues(report)

        # 驗證修復結果
        assert auto_fix_results["total_fixable"] > 0

    def test_concurrent_consistency_checks(self, temp_dir):
        """測試併發一致性檢查"""
        manager = DataConsistencyManager()

        # 準備測試上下文
        context = {"file_registry": {}, "index_registry": {}, "vector_store": None}

        # 使用多執行緒執行併發檢查
        reports = []

        def run_check(thread_id):
            report = manager.run_consistency_check(context, ConsistencyLevel.BASIC)
            reports.append((thread_id, report))

        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_check, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有執行緒完成
        for thread in threads:
            thread.join()

        # 驗證所有檢查都完成
        assert len(reports) == 3

        # 驗證報告都有效
        for thread_id, report in reports:
            assert report is not None
            assert report.report_id is not None
            assert report.total_checked >= 0


if __name__ == "__main__":
    pytest.main([__file__])
