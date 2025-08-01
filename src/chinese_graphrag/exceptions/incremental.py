"""
增量索引管理器

提供增量索引功能，支援文件變更檢測、增量更新和索引同步機制。
"""

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

from ..exceptions.base import ChineseGraphRAGError, IndexError


logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """變更類型枚舉"""
    ADDED = "added"         # 新增
    MODIFIED = "modified"   # 修改
    DELETED = "deleted"     # 刪除
    MOVED = "moved"         # 移動


class IndexStatus(Enum):
    """索引狀態枚舉"""
    PENDING = "pending"     # 待處理
    PROCESSING = "processing"   # 處理中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"       # 失敗
    SKIPPED = "skipped"     # 跳過


@dataclass
class FileMetadata:
    """檔案元資料"""
    file_path: str
    size_bytes: int
    modified_time: datetime
    content_hash: str
    encoding: str
    document_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data['modified_time'] = self.modified_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        """從字典創建"""
        data = data.copy()
        data['modified_time'] = datetime.fromisoformat(data['modified_time'])
        return cls(**data)


@dataclass
class ChangeRecord:
    """變更記錄"""
    change_id: str
    file_path: str
    change_type: ChangeType
    timestamp: datetime
    old_metadata: Optional[FileMetadata] = None
    new_metadata: Optional[FileMetadata] = None
    processed: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'change_id': self.change_id,
            'file_path': self.file_path,
            'change_type': self.change_type.value,
            'timestamp': self.timestamp.isoformat(),
            'old_metadata': self.old_metadata.to_dict() if self.old_metadata else None,
            'new_metadata': self.new_metadata.to_dict() if self.new_metadata else None,
            'processed': self.processed,
            'error_message': self.error_message
        }


class FileWatcher:
    """檔案監控器"""
    
    def __init__(self, watch_paths: List[Union[str, Path]]):
        self.watch_paths = [Path(p) for p in watch_paths]
        self.file_registry: Dict[str, FileMetadata] = {}
        self.change_listeners: List[callable] = []
        self._lock = threading.RLock()
    
    def add_change_listener(self, listener: callable):
        """添加變更監聽器"""
        with self._lock:
            self.change_listeners.append(listener)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """計算檔案雜湊"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"計算檔案雜湊失敗 {file_path}: {e}")
            return ""
    
    def _detect_encoding(self, file_path: Path) -> str:
        """檢測檔案編碼"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8') or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _create_file_metadata(self, file_path: Path) -> FileMetadata:
        """創建檔案元資料"""
        stat = file_path.stat()
        return FileMetadata(
            file_path=str(file_path),
            size_bytes=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            content_hash=self._calculate_file_hash(file_path),
            encoding=self._detect_encoding(file_path)
        )
    
    def scan_changes(self) -> List[ChangeRecord]:
        """掃描變更"""
        changes = []
        current_files = {}
        
        # 掃描當前檔案
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
                
            if watch_path.is_file():
                current_files[str(watch_path)] = self._create_file_metadata(watch_path)
            else:
                for file_path in watch_path.rglob("*"):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        current_files[str(file_path)] = self._create_file_metadata(file_path)
        
        with self._lock:
            # 檢測新增和修改的檔案
            for file_path, new_metadata in current_files.items():
                old_metadata = self.file_registry.get(file_path)
                
                if old_metadata is None:
                    # 新增的檔案
                    change = ChangeRecord(
                        change_id=f"add_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                        file_path=file_path,
                        change_type=ChangeType.ADDED,
                        timestamp=datetime.now(),
                        new_metadata=new_metadata
                    )
                    changes.append(change)
                    
                elif (old_metadata.content_hash != new_metadata.content_hash or 
                      old_metadata.modified_time != new_metadata.modified_time):
                    # 修改的檔案
                    change = ChangeRecord(
                        change_id=f"mod_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                        file_path=file_path,
                        change_type=ChangeType.MODIFIED,
                        timestamp=datetime.now(),
                        old_metadata=old_metadata,
                        new_metadata=new_metadata
                    )
                    changes.append(change)
            
            # 檢測刪除的檔案
            for file_path, old_metadata in self.file_registry.items():
                if file_path not in current_files:
                    change = ChangeRecord(
                        change_id=f"del_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                        file_path=file_path,
                        change_type=ChangeType.DELETED,
                        timestamp=datetime.now(),
                        old_metadata=old_metadata
                    )
                    changes.append(change)
            
            # 更新檔案註冊表
            self.file_registry = current_files
        
        # 通知監聽器
        for change in changes:
            for listener in self.change_listeners:
                try:
                    listener(change)
                except Exception as e:
                    logger.error(f"變更監聽器執行失敗: {e}")
        
        return changes


class IncrementalIndexStorage:
    """增量索引儲存"""
    
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._lock = threading.RLock()
    
    def _init_database(self):
        """初始化資料庫"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_path TEXT PRIMARY KEY,
                    size_bytes INTEGER,
                    modified_time TEXT,
                    content_hash TEXT,
                    encoding TEXT,
                    document_id TEXT,
                    last_indexed TEXT,
                    index_status TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS change_records (
                    change_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    change_type TEXT,
                    timestamp TEXT,
                    old_metadata TEXT,
                    new_metadata TEXT,
                    processed BOOLEAN,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS index_batches (
                    batch_id TEXT PRIMARY KEY,
                    created_time TEXT,
                    status TEXT,
                    total_files INTEGER,
                    processed_files INTEGER,
                    error_count INTEGER,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    def save_file_metadata(self, metadata: FileMetadata):
        """儲存檔案元資料"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO file_metadata
                    (file_path, size_bytes, modified_time, content_hash, encoding, document_id, last_indexed, index_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.file_path,
                    metadata.size_bytes,
                    metadata.modified_time.isoformat(),
                    metadata.content_hash,
                    metadata.encoding,
                    metadata.document_id,
                    datetime.now().isoformat(),
                    IndexStatus.PENDING.value
                ))
                conn.commit()
    
    def get_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """獲取檔案元資料"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_path, size_bytes, modified_time, content_hash, encoding, document_id
                    FROM file_metadata WHERE file_path = ?
                """, (file_path,))
                
                row = cursor.fetchone()
                if row:
                    return FileMetadata(
                        file_path=row[0],
                        size_bytes=row[1],
                        modified_time=datetime.fromisoformat(row[2]),
                        content_hash=row[3],
                        encoding=row[4],
                        document_id=row[5]
                    )
                return None
    
    def save_change_record(self, change: ChangeRecord):
        """儲存變更記錄"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO change_records
                    (change_id, file_path, change_type, timestamp, old_metadata, new_metadata, processed, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    change.change_id,
                    change.file_path,
                    change.change_type.value,
                    change.timestamp.isoformat(),
                    json.dumps(change.old_metadata.to_dict()) if change.old_metadata else None,
                    json.dumps(change.new_metadata.to_dict()) if change.new_metadata else None,
                    change.processed,
                    change.error_message
                ))
                conn.commit()
    
    def get_pending_changes(self, limit: int = 100) -> List[ChangeRecord]:
        """獲取待處理的變更"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT change_id, file_path, change_type, timestamp, old_metadata, new_metadata, processed, error_message
                    FROM change_records WHERE processed = 0
                    ORDER BY timestamp ASC LIMIT ?
                """, (limit,))
                
                changes = []
                for row in cursor.fetchall():
                    old_metadata = None
                    new_metadata = None
                    
                    if row[4]:
                        old_metadata = FileMetadata.from_dict(json.loads(row[4]))
                    if row[5]:
                        new_metadata = FileMetadata.from_dict(json.loads(row[5]))
                    
                    change = ChangeRecord(
                        change_id=row[0],
                        file_path=row[1],
                        change_type=ChangeType(row[2]),
                        timestamp=datetime.fromisoformat(row[3]),
                        old_metadata=old_metadata,
                        new_metadata=new_metadata,
                        processed=bool(row[6]),
                        error_message=row[7]
                    )
                    changes.append(change)
                
                return changes
    
    def mark_change_processed(self, change_id: str, error_message: Optional[str] = None):
        """標記變更為已處理"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE change_records 
                    SET processed = 1, error_message = ?
                    WHERE change_id = ?
                """, (error_message, change_id))
                conn.commit()


class IncrementalIndexManager:
    """增量索引管理器"""
    
    def __init__(
        self,
        storage: IncrementalIndexStorage,
        watch_paths: List[Union[str, Path]],
        batch_size: int = 50
    ):
        self.storage = storage
        self.file_watcher = FileWatcher(watch_paths)
        self.batch_size = batch_size
        self.processing = False
        self._lock = threading.RLock()
        
        # 註冊變更監聽器
        self.file_watcher.add_change_listener(self._on_file_change)
    
    def _on_file_change(self, change: ChangeRecord):
        """檔案變更回調"""
        logger.info(f"檢測到檔案變更: {change.file_path} ({change.change_type.value})")
        self.storage.save_change_record(change)
    
    def scan_for_changes(self) -> List[ChangeRecord]:
        """掃描檔案變更"""
        logger.info("開始掃描檔案變更...")
        changes = self.file_watcher.scan_changes()
        logger.info(f"發現 {len(changes)} 個檔案變更")
        return changes
    
    def process_pending_changes(self, max_batches: int = 10) -> Dict[str, Any]:
        """處理待處理的變更"""
        if self.processing:
            logger.warning("增量索引正在處理中，跳過本次執行")
            return {"status": "skipped", "reason": "already_processing"}
        
        with self._lock:
            self.processing = True
            
            try:
                results = {
                    "total_processed": 0,
                    "successful": 0,
                    "failed": 0,
                    "batches": []
                }
                
                for batch_num in range(max_batches):
                    changes = self.storage.get_pending_changes(self.batch_size)
                    if not changes:
                        break
                    
                    batch_result = self._process_change_batch(changes, batch_num + 1)
                    results["batches"].append(batch_result)
                    results["total_processed"] += batch_result["processed"]
                    results["successful"] += batch_result["successful"]
                    results["failed"] += batch_result["failed"]
                
                logger.info(f"增量索引處理完成: {results}")
                return results
                
            finally:
                self.processing = False
    
    def _process_change_batch(self, changes: List[ChangeRecord], batch_num: int) -> Dict[str, Any]:
        """處理變更批次"""
        logger.info(f"處理變更批次 {batch_num}，包含 {len(changes)} 個變更")
        
        batch_result = {
            "batch_number": batch_num,
            "processed": len(changes),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for change in changes:
            try:
                self._process_single_change(change)
                batch_result["successful"] += 1
                self.storage.mark_change_processed(change.change_id)
                
            except Exception as e:
                error_msg = f"處理變更失敗: {e}"
                logger.error(f"處理變更 {change.change_id} 失敗: {e}")
                batch_result["failed"] += 1
                batch_result["errors"].append({
                    "change_id": change.change_id,
                    "file_path": change.file_path,
                    "error": str(e)
                })
                self.storage.mark_change_processed(change.change_id, error_msg)
        
        return batch_result
    
    def _process_single_change(self, change: ChangeRecord):
        """處理單個變更"""
        if change.change_type == ChangeType.ADDED:
            self._handle_file_added(change)
        elif change.change_type == ChangeType.MODIFIED:
            self._handle_file_modified(change)
        elif change.change_type == ChangeType.DELETED:
            self._handle_file_deleted(change)
        else:
            logger.warning(f"未知的變更類型: {change.change_type}")
    
    def _handle_file_added(self, change: ChangeRecord):
        """處理檔案新增"""
        logger.debug(f"處理新增檔案: {change.file_path}")
        
        if change.new_metadata:
            # 儲存檔案元資料
            self.storage.save_file_metadata(change.new_metadata)
            
            # 這裡可以觸發實際的索引操作
            # 例如：調用文件處理器和索引引擎
            logger.info(f"檔案已加入索引佇列: {change.file_path}")
    
    def _handle_file_modified(self, change: ChangeRecord):
        """處理檔案修改"""
        logger.debug(f"處理修改檔案: {change.file_path}")
        
        if change.new_metadata:
            # 更新檔案元資料
            self.storage.save_file_metadata(change.new_metadata)
            
            # 標記需要重新索引
            logger.info(f"檔案標記為需要重新索引: {change.file_path}")
    
    def _handle_file_deleted(self, change: ChangeRecord):
        """處理檔案刪除"""
        logger.debug(f"處理刪除檔案: {change.file_path}")
        
        # 這裡可以觸發索引清理操作
        # 例如：從向量資料庫中移除相關數據
        logger.info(f"檔案標記為需要從索引中移除: {change.file_path}")
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """獲取索引統計資訊"""
        with sqlite3.connect(self.storage.db_path) as conn:
            # 檔案統計
            cursor = conn.execute("SELECT COUNT(*) FROM file_metadata")
            total_files = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM file_metadata WHERE index_status = ?", (IndexStatus.COMPLETED.value,))
            indexed_files = cursor.fetchone()[0]
            
            # 變更統計
            cursor = conn.execute("SELECT COUNT(*) FROM change_records WHERE processed = 0")
            pending_changes = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM change_records WHERE processed = 1 AND error_message IS NULL")
            successful_changes = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM change_records WHERE processed = 1 AND error_message IS NOT NULL")
            failed_changes = cursor.fetchone()[0]
            
            return {
                "files": {
                    "total": total_files,
                    "indexed": indexed_files,
                    "pending": total_files - indexed_files
                },
                "changes": {
                    "pending": pending_changes,
                    "successful": successful_changes,
                    "failed": failed_changes
                },
                "processing": self.processing
            }
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """清理舊記錄"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.storage.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM change_records 
                WHERE processed = 1 AND timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"清理了 {deleted_count} 條舊的變更記錄")
            return deleted_count