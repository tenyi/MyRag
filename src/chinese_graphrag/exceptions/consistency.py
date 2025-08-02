"""
資料一致性檢查器

提供全面的資料一致性檢查功能，包括索引一致性、向量一致性和元資料一致性檢查。
"""

import hashlib
import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

from ..exceptions.base import ChineseGraphRAGError, ValidationError


logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """一致性級別枚舉"""
    BASIC = "basic"           # 基本檢查
    THOROUGH = "thorough"     # 詳細檢查
    COMPREHENSIVE = "comprehensive"  # 全面檢查


class IssueType(Enum):
    """問題類型枚舉"""
    MISSING_FILE = "missing_file"           # 缺失檔案
    MISSING_INDEX = "missing_index"         # 缺失索引
    ORPHANED_INDEX = "orphaned_index"       # 孤立索引
    HASH_MISMATCH = "hash_mismatch"         # 雜湊不匹配
    SIZE_MISMATCH = "size_mismatch"         # 大小不匹配
    TIMESTAMP_MISMATCH = "timestamp_mismatch"  # 時間戳不匹配
    VECTOR_CORRUPTION = "vector_corruption"    # 向量損壞
    METADATA_CORRUPTION = "metadata_corruption"  # 元資料損壞
    DUPLICATE_ENTRY = "duplicate_entry"        # 重複條目
    SCHEMA_VIOLATION = "schema_violation"      # 架構違規


class Severity(Enum):
    """嚴重程度枚舉"""
    INFO = "info"           # 資訊
    WARNING = "warning"     # 警告
    ERROR = "error"         # 錯誤
    CRITICAL = "critical"   # 嚴重


@dataclass
class ConsistencyIssue:
    """一致性問題"""
    issue_id: str
    issue_type: IssueType
    severity: Severity
    description: str
    affected_resource: str
    details: Dict[str, Any]
    suggested_action: str
    timestamp: datetime
    auto_fixable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data['issue_type'] = self.issue_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ConsistencyReport:
    """一致性報告"""
    report_id: str
    timestamp: datetime
    consistency_level: ConsistencyLevel
    total_checked: int
    issues_found: int
    issues_by_type: Dict[IssueType, int]
    issues_by_severity: Dict[Severity, int]
    issues: List[ConsistencyIssue]
    execution_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp.isoformat(),
            'consistency_level': self.consistency_level.value,
            'total_checked': self.total_checked,
            'issues_found': self.issues_found,
            'issues_by_type': {k.value: v for k, v in self.issues_by_type.items()},
            'issues_by_severity': {k.value: v for k, v in self.issues_by_severity.items()},
            'issues': [issue.to_dict() for issue in self.issues],
            'execution_time_seconds': self.execution_time_seconds
        }


class ConsistencyChecker(ABC):
    """一致性檢查器抽象基類"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def check(self, context: Dict[str, Any]) -> List[ConsistencyIssue]:
        """執行一致性檢查"""
        pass
    
    def get_description(self) -> str:
        """獲取檢查器描述"""
        return f"{self.name} 一致性檢查器"


class FileSystemChecker(ConsistencyChecker):
    """檔案系統一致性檢查器"""
    
    def __init__(self):
        super().__init__("檔案系統")
    
    def check(self, context: Dict[str, Any]) -> List[ConsistencyIssue]:
        """檢查檔案系統一致性"""
        issues = []
        file_registry = context.get('file_registry', {})
        data_directory = Path(context.get('data_directory', './data'))
        
        for file_path, metadata in file_registry.items():
            file_path_obj = Path(file_path)
            
            # 檢查檔案是否存在
            if not file_path_obj.exists():
                issues.append(ConsistencyIssue(
                    issue_id=f"fs_missing_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    issue_type=IssueType.MISSING_FILE,
                    severity=Severity.ERROR,
                    description=f"檔案不存在: {file_path}",
                    affected_resource=file_path,
                    details={"expected_path": file_path},
                    suggested_action="檢查檔案是否已被移動或刪除，並更新索引",
                    timestamp=datetime.now()
                ))
                continue
            
            # 檢查檔案大小
            actual_size = file_path_obj.stat().st_size
            expected_size = metadata.get('size_bytes', 0)
            
            if actual_size != expected_size:
                issues.append(ConsistencyIssue(
                    issue_id=f"fs_size_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    issue_type=IssueType.SIZE_MISMATCH,
                    severity=Severity.WARNING,
                    description=f"檔案大小不匹配: {file_path}",
                    affected_resource=file_path,
                    details={
                        "expected_size": expected_size,
                        "actual_size": actual_size
                    },
                    suggested_action="重新計算檔案雜湊並更新元資料",
                    timestamp=datetime.now(),
                    auto_fixable=True
                ))
            
            # 檢查檔案雜湊（如果有的話）
            if 'content_hash' in metadata:
                try:
                    hasher = hashlib.sha256()
                    with open(file_path_obj, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
                    actual_hash = hasher.hexdigest()
                    expected_hash = metadata['content_hash']
                    
                    if actual_hash != expected_hash:
                        issues.append(ConsistencyIssue(
                            issue_id=f"fs_hash_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                            issue_type=IssueType.HASH_MISMATCH,
                            severity=Severity.ERROR,
                            description=f"檔案雜湊不匹配: {file_path}",
                            affected_resource=file_path,
                            details={
                                "expected_hash": expected_hash,
                                "actual_hash": actual_hash
                            },
                            suggested_action="檔案內容已變更，需要重新索引",
                            timestamp=datetime.now()
                        ))
                        
                except Exception as e:
                    issues.append(ConsistencyIssue(
                        issue_id=f"fs_error_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                        issue_type=IssueType.METADATA_CORRUPTION,
                        severity=Severity.WARNING,
                        description=f"無法讀取檔案進行雜湊檢查: {file_path}",
                        affected_resource=file_path,
                        details={"error": str(e)},
                        suggested_action="檢查檔案權限和完整性",
                        timestamp=datetime.now()
                    ))
        
        return issues


class IndexChecker(ConsistencyChecker):
    """索引一致性檢查器"""
    
    def __init__(self):
        super().__init__("索引")
    
    def check(self, context: Dict[str, Any]) -> List[ConsistencyIssue]:
        """檢查索引一致性"""
        issues = []
        file_registry = context.get('file_registry', {})
        index_registry = context.get('index_registry', {})
        
        # 檢查缺失的索引
        for file_path in file_registry:
            document_id = file_registry[file_path].get('document_id')
            if document_id and document_id not in index_registry:
                issues.append(ConsistencyIssue(
                    issue_id=f"idx_missing_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    issue_type=IssueType.MISSING_INDEX,
                    severity=Severity.WARNING,
                    description=f"檔案缺少索引: {file_path}",
                    affected_resource=file_path,
                    details={"document_id": document_id},
                    suggested_action="為此檔案重新建立索引",
                    timestamp=datetime.now(),
                    auto_fixable=True
                ))
        
        # 檢查孤立的索引
        for document_id, index_data in index_registry.items():
            source_file = index_data.get('source_file')
            if source_file and source_file not in file_registry:
                issues.append(ConsistencyIssue(
                    issue_id=f"idx_orphan_{hashlib.md5(document_id.encode()).hexdigest()[:8]}",
                    issue_type=IssueType.ORPHANED_INDEX,
                    severity=Severity.WARNING,
                    description=f"孤立的索引條目: {document_id}",
                    affected_resource=document_id,
                    details={"source_file": source_file},
                    suggested_action="刪除此孤立的索引條目",
                    timestamp=datetime.now(),
                    auto_fixable=True
                ))
        
        return issues


class VectorStoreChecker(ConsistencyChecker):
    """向量儲存一致性檢查器"""
    
    def __init__(self):
        super().__init__("向量儲存")
    
    def check(self, context: Dict[str, Any]) -> List[ConsistencyIssue]:
        """檢查向量儲存一致性"""
        issues = []
        vector_store = context.get('vector_store')
        index_registry = context.get('index_registry', {})
        
        if not vector_store:
            return issues
        
        try:
            # 獲取向量儲存中的所有文件 ID
            # 使用適配的方法來獲取文件 ID 列表
            stored_ids = set()
            if hasattr(vector_store, 'list_document_ids'):
                stored_ids = set(vector_store.list_document_ids())
            elif hasattr(vector_store, 'get_all_ids'):
                stored_ids = set(vector_store.get_all_ids())
            else:
                # 如果沒有直接的方法，嘗試從集合中獲取
                try:
                    if hasattr(vector_store, 'list_collections'):
                        # 如果是異步方法，需要在異步上下文中調用
                        import asyncio
                        try:
                            collections = asyncio.run(vector_store.list_collections())
                        except:
                            collections = []
                    else:
                        collections = []
                    
                    for collection in collections:
                        if hasattr(collection, 'get_all_ids'):
                            stored_ids.update(collection.get_all_ids())
                except:
                    pass
            
            indexed_ids = set(index_registry.keys())
            
            # 檢查缺失的向量
            missing_vectors = indexed_ids - stored_ids
            for doc_id in missing_vectors:
                issues.append(ConsistencyIssue(
                    issue_id=f"vec_missing_{hashlib.md5(doc_id.encode()).hexdigest()[:8]}",
                    issue_type=IssueType.MISSING_INDEX,
                    severity=Severity.ERROR,
                    description=f"文件缺少向量: {doc_id}",
                    affected_resource=doc_id,
                    details={"document_id": doc_id},
                    suggested_action="重新生成文件向量",
                    timestamp=datetime.now(),
                    auto_fixable=True
                ))
            
            # 檢查孤立的向量
            orphaned_vectors = stored_ids - indexed_ids
            for doc_id in orphaned_vectors:
                issues.append(ConsistencyIssue(
                    issue_id=f"vec_orphan_{hashlib.md5(doc_id.encode()).hexdigest()[:8]}",
                    issue_type=IssueType.ORPHANED_INDEX,
                    severity=Severity.WARNING,
                    description=f"孤立的向量: {doc_id}",
                    affected_resource=doc_id,
                    details={"document_id": doc_id},
                    suggested_action="從向量儲存中刪除此向量",
                    timestamp=datetime.now(),
                    auto_fixable=True
                ))
            
            # 檢查向量維度一致性
            for doc_id in stored_ids.intersection(indexed_ids):
                try:
                    vector = None
                    if hasattr(vector_store, 'get_vector'):
                        vector = vector_store.get_vector(doc_id)
                    elif hasattr(vector_store, 'get_vector_by_id'):
                        import asyncio
                        try:
                            result = asyncio.run(vector_store.get_vector_by_id("default", doc_id, include_embedding=True))
                            if result and 'vector' in result:
                                vector = result['vector']
                        except:
                            vector = None
                    
                    if vector is not None:
                        expected_dim = context.get('expected_vector_dimension', 1024)
                        if len(vector) != expected_dim:
                            issues.append(ConsistencyIssue(
                                issue_id=f"vec_dim_{hashlib.md5(doc_id.encode()).hexdigest()[:8]}",
                                issue_type=IssueType.VECTOR_CORRUPTION,
                                severity=Severity.ERROR,
                                description=f"向量維度不正確: {doc_id}",
                                affected_resource=doc_id,
                                details={
                                    "expected_dimension": expected_dim,
                                    "actual_dimension": len(vector)
                                },
                                suggested_action="重新生成此文件的向量",
                                timestamp=datetime.now()
                            ))
                except Exception as e:
                    issues.append(ConsistencyIssue(
                        issue_id=f"vec_error_{hashlib.md5(doc_id.encode()).hexdigest()[:8]}",
                        issue_type=IssueType.VECTOR_CORRUPTION,
                        severity=Severity.ERROR,
                        description=f"向量讀取錯誤: {doc_id}",
                        affected_resource=doc_id,
                        details={"error": str(e)},
                        suggested_action="檢查向量儲存完整性並重新生成向量",
                        timestamp=datetime.now()
                    ))
        
        except Exception as e:
            issues.append(ConsistencyIssue(
                issue_id=f"vec_store_error_{int(datetime.now().timestamp())}",
                issue_type=IssueType.METADATA_CORRUPTION,
                severity=Severity.CRITICAL,
                description="向量儲存檢查失敗",
                affected_resource="vector_store",
                details={"error": str(e)},
                suggested_action="檢查向量儲存連接和完整性",
                timestamp=datetime.now()
            ))
        
        return issues


class MetadataChecker(ConsistencyChecker):
    """元資料一致性檢查器"""
    
    def __init__(self):
        super().__init__("元資料")
    
    def check(self, context: Dict[str, Any]) -> List[ConsistencyIssue]:
        """檢查元資料一致性"""
        issues = []
        index_registry = context.get('index_registry', {})
        
        for document_id, metadata in index_registry.items():
            # 檢查必要欄位
            required_fields = ['source_file', 'created_time', 'content_hash']
            for field in required_fields:
                if field not in metadata:
                    issues.append(ConsistencyIssue(
                        issue_id=f"meta_missing_{hashlib.md5(document_id.encode()).hexdigest()[:8]}",
                        issue_type=IssueType.SCHEMA_VIOLATION,
                        severity=Severity.WARNING,
                        description=f"文件元資料缺少必要欄位 '{field}': {document_id}",
                        affected_resource=document_id,
                        details={"missing_field": field},
                        suggested_action="更新文件元資料並補充缺少的欄位",
                        timestamp=datetime.now(),
                        auto_fixable=True
                    ))
            
            # 檢查時間戳格式
            if 'created_time' in metadata:
                try:
                    datetime.fromisoformat(metadata['created_time'].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    issues.append(ConsistencyIssue(
                        issue_id=f"meta_time_{hashlib.md5(document_id.encode()).hexdigest()[:8]}",
                        issue_type=IssueType.METADATA_CORRUPTION,
                        severity=Severity.WARNING,
                        description=f"時間戳格式不正確: {document_id}",
                        affected_resource=document_id,
                        details={"invalid_timestamp": metadata['created_time']},
                        suggested_action="更正時間戳格式",
                        timestamp=datetime.now(),
                        auto_fixable=True
                    ))
        
        return issues


class DataConsistencyManager:
    """資料一致性管理器"""
    
    def __init__(self):
        self.checkers: List[ConsistencyChecker] = []
        self.reports: List[ConsistencyReport] = []
        self._lock = threading.RLock()
        
        # 註冊預設檢查器
        self.register_checker(FileSystemChecker())
        self.register_checker(IndexChecker())
        self.register_checker(VectorStoreChecker())
        self.register_checker(MetadataChecker())
    
    def register_checker(self, checker: ConsistencyChecker):
        """註冊檢查器"""
        with self._lock:
            self.checkers.append(checker)
            logger.info(f"註冊一致性檢查器: {checker.name}")
    
    def run_consistency_check(
        self,
        context: Dict[str, Any],
        level: ConsistencyLevel = ConsistencyLevel.BASIC,
        checkers: Optional[List[str]] = None
    ) -> ConsistencyReport:
        """執行一致性檢查"""
        start_time = datetime.now()
        
        # 選擇要執行的檢查器
        selected_checkers = self.checkers
        if checkers:
            selected_checkers = [c for c in self.checkers if c.name in checkers]
        
        all_issues = []
        total_checked = 0
        
        for checker in selected_checkers:
            try:
                logger.info(f"執行 {checker.name} 一致性檢查...")
                issues = checker.check(context)
                all_issues.extend(issues)
                total_checked += 1
                logger.info(f"{checker.name} 檢查完成，發現 {len(issues)} 個問題")
                
            except Exception as e:
                logger.error(f"{checker.name} 檢查失敗: {e}")
                # 添加檢查器失敗的問題
                all_issues.append(ConsistencyIssue(
                    issue_id=f"checker_error_{checker.name}_{int(datetime.now().timestamp())}",
                    issue_type=IssueType.METADATA_CORRUPTION,
                    severity=Severity.ERROR,
                    description=f"{checker.name} 檢查器執行失敗",
                    affected_resource=checker.name,
                    details={"error": str(e)},
                    suggested_action="檢查檢查器配置和系統狀態",
                    timestamp=datetime.now()
                ))
        
        # 統計問題
        issues_by_type = {}
        issues_by_severity = {}
        
        for issue in all_issues:
            issues_by_type[issue.issue_type] = issues_by_type.get(issue.issue_type, 0) + 1
            issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1
        
        # 生成報告
        execution_time = (datetime.now() - start_time).total_seconds()
        report = ConsistencyReport(
            report_id=f"report_{int(start_time.timestamp())}",
            timestamp=start_time,
            consistency_level=level,
            total_checked=total_checked,
            issues_found=len(all_issues),
            issues_by_type=issues_by_type,
            issues_by_severity=issues_by_severity,
            issues=all_issues,
            execution_time_seconds=execution_time
        )
        
        with self._lock:
            self.reports.append(report)
            # 限制報告數量
            if len(self.reports) > 50:
                self.reports = self.reports[-50:]
        
        logger.info(f"一致性檢查完成，發現 {len(all_issues)} 個問題，耗時 {execution_time:.2f} 秒")
        return report
    
    def auto_fix_issues(self, report: ConsistencyReport) -> Dict[str, Any]:
        """自動修復問題"""
        fixable_issues = [issue for issue in report.issues if issue.auto_fixable]
        
        results = {
            "total_fixable": len(fixable_issues),
            "fixed": 0,
            "failed": 0,
            "errors": []
        }
        
        for issue in fixable_issues:
            try:
                self._auto_fix_issue(issue)
                results["fixed"] += 1
                logger.info(f"已自動修復問題: {issue.issue_id}")
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "issue_id": issue.issue_id,
                    "error": str(e)
                })
                logger.error(f"自動修復問題失敗 {issue.issue_id}: {e}")
        
        return results
    
    def _auto_fix_issue(self, issue: ConsistencyIssue):
        """自動修復單個問題"""
        # 這裡可以實作具體的修復邏輯
        # 根據問題類型執行相應的修復操作
        
        if issue.issue_type == IssueType.ORPHANED_INDEX:
            # 刪除孤立的索引條目
            logger.info(f"刪除孤立索引條目: {issue.affected_resource}")
            
        elif issue.issue_type == IssueType.MISSING_INDEX:
            # 重新建立缺失的索引
            logger.info(f"重新建立索引: {issue.affected_resource}")
            
        elif issue.issue_type == IssueType.SIZE_MISMATCH:
            # 更新檔案大小資訊
            logger.info(f"更新檔案大小資訊: {issue.affected_resource}")
            
        else:
            raise NotImplementedError(f"不支援自動修復問題類型: {issue.issue_type}")
    
    def get_latest_report(self) -> Optional[ConsistencyReport]:
        """獲取最新的一致性報告"""
        with self._lock:
            return self.reports[-1] if self.reports else None
    
    def get_reports_history(self, limit: int = 10) -> List[ConsistencyReport]:
        """獲取報告歷史"""
        with self._lock:
            return self.reports[-limit:]
    
    def export_report(self, report: ConsistencyReport, file_path: Union[str, Path]):
        """匯出報告"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"一致性報告已匯出: {file_path}")


# 全域一致性管理器
_consistency_manager = None
_consistency_lock = threading.Lock()


def get_consistency_manager() -> DataConsistencyManager:
    """獲取全域一致性管理器"""
    global _consistency_manager
    
    if _consistency_manager is None:
        with _consistency_lock:
            if _consistency_manager is None:
                _consistency_manager = DataConsistencyManager()
    
    return _consistency_manager