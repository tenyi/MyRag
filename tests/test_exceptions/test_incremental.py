"""
測試增量索引管理器
"""

import json
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest

from src.chinese_graphrag.exceptions.incremental import (
    ChangeRecord,
    ChangeType,
    FileMetadata,
    FileWatcher,
    IncrementalIndexManager,
    IncrementalIndexStorage,
    IndexStatus,
)


class TestFileMetadata:
    """測試檔案元資料"""

    def test_metadata_creation(self):
        """測試元資料建立"""
        now = datetime.now()
        metadata = FileMetadata(
            file_path="/test/file.txt",
            size_bytes=1024,
            modified_time=now,
            content_hash="abc123",
            encoding="utf-8",
            document_id="doc_001",
        )

        assert metadata.file_path == "/test/file.txt"
        assert metadata.size_bytes == 1024
        assert metadata.modified_time == now
        assert metadata.content_hash == "abc123"
        assert metadata.encoding == "utf-8"
        assert metadata.document_id == "doc_001"

    def test_metadata_serialization(self):
        """測試元資料序列化"""
        now = datetime.now()
        metadata = FileMetadata(
            file_path="/test/serialize.txt",
            size_bytes=2048,
            modified_time=now,
            content_hash="def456",
            encoding="utf-8",
        )

        # 轉換為字典
        data = metadata.to_dict()
        assert data["file_path"] == "/test/serialize.txt"
        assert data["size_bytes"] == 2048
        assert data["modified_time"] == now.isoformat()

        # 從字典恢復
        restored = FileMetadata.from_dict(data)
        assert restored.file_path == metadata.file_path
        assert restored.size_bytes == metadata.size_bytes
        assert restored.modified_time == metadata.modified_time


class TestChangeRecord:
    """測試變更記錄"""

    def test_change_record_creation(self):
        """測試變更記錄建立"""
        now = datetime.now()
        old_metadata = FileMetadata(
            file_path="/test/old.txt",
            size_bytes=512,
            modified_time=now - timedelta(hours=1),
            content_hash="old123",
            encoding="utf-8",
        )

        new_metadata = FileMetadata(
            file_path="/test/new.txt",
            size_bytes=1024,
            modified_time=now,
            content_hash="new456",
            encoding="utf-8",
        )

        change = ChangeRecord(
            change_id="change_001",
            file_path="/test/modified.txt",
            change_type=ChangeType.MODIFIED,
            timestamp=now,
            old_metadata=old_metadata,
            new_metadata=new_metadata,
            processed=False,
        )

        assert change.change_id == "change_001"
        assert change.change_type == ChangeType.MODIFIED
        assert change.processed is False
        assert change.old_metadata is old_metadata
        assert change.new_metadata is new_metadata

    def test_change_record_serialization(self):
        """測試變更記錄序列化"""
        now = datetime.now()
        metadata = FileMetadata(
            file_path="/test/file.txt",
            size_bytes=1024,
            modified_time=now,
            content_hash="abc123",
            encoding="utf-8",
        )

        change = ChangeRecord(
            change_id="change_002",
            file_path="/test/file.txt",
            change_type=ChangeType.ADDED,
            timestamp=now,
            new_metadata=metadata,
        )

        # 轉換為字典
        data = change.to_dict()
        assert data["change_id"] == "change_002"
        assert data["change_type"] == "added"
        assert data["timestamp"] == now.isoformat()
        assert data["new_metadata"] is not None


class TestFileWatcher:
    """測試檔案監控器"""

    def test_watcher_creation(self, temp_dir):
        """測試監控器建立"""
        watcher = FileWatcher([temp_dir])

        assert len(watcher.watch_paths) == 1
        assert watcher.watch_paths[0] == temp_dir
        assert len(watcher.file_registry) == 0
        assert len(watcher.change_listeners) == 0

    def test_add_change_listener(self, temp_dir):
        """測試添加變更監聽器"""
        watcher = FileWatcher([temp_dir])

        def test_listener(change):
            pass

        watcher.add_change_listener(test_listener)

        assert len(watcher.change_listeners) == 1
        assert watcher.change_listeners[0] is test_listener

    def test_calculate_file_hash(self, temp_dir):
        """測試計算檔案雜湊"""
        watcher = FileWatcher([temp_dir])

        # 建立測試檔案
        test_file = temp_dir / "test_hash.txt"
        test_content = "測試檔案內容用於計算雜湊值"

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        # 計算雜湊
        hash_value = watcher._calculate_file_hash(test_file)

        assert hash_value is not None
        assert len(hash_value) == 64  # SHA256 雜湊長度

        # 相同內容應該產生相同雜湊
        hash_value2 = watcher._calculate_file_hash(test_file)
        assert hash_value == hash_value2

    def test_detect_encoding(self, temp_dir):
        """測試檢測檔案編碼"""
        watcher = FileWatcher([temp_dir])

        # 建立 UTF-8 檔案
        utf8_file = temp_dir / "utf8.txt"
        with open(utf8_file, "w", encoding="utf-8") as f:
            f.write("UTF-8 中文測試")

        encoding = watcher._detect_encoding(utf8_file)
        assert encoding in ["utf-8", "UTF-8"]

    def test_create_file_metadata(self, temp_dir):
        """測試建立檔案元資料"""
        watcher = FileWatcher([temp_dir])

        # 建立測試檔案
        test_file = temp_dir / "metadata_test.txt"
        test_content = "測試元資料建立"

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        # 建立元資料
        metadata = watcher._create_file_metadata(test_file)

        assert metadata.file_path == str(test_file)
        assert metadata.size_bytes > 0
        assert metadata.modified_time is not None
        assert metadata.content_hash is not None
        assert metadata.encoding is not None

    def test_scan_changes_new_files(self, temp_dir):
        """測試掃描新檔案變更"""
        watcher = FileWatcher([temp_dir])

        # 第一次掃描（空目錄）
        changes = watcher.scan_changes()
        assert len(changes) == 0

        # 建立新檔案
        new_file = temp_dir / "new_file.txt"
        with open(new_file, "w", encoding="utf-8") as f:
            f.write("新建立的檔案")

        # 第二次掃描
        changes = watcher.scan_changes()
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.ADDED
        assert changes[0].file_path == str(new_file)
        assert changes[0].new_metadata is not None

    def test_scan_changes_modified_files(self, temp_dir):
        """測試掃描修改檔案變更"""
        watcher = FileWatcher([temp_dir])

        # 建立初始檔案
        test_file = temp_dir / "modify_test.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("初始內容")

        # 第一次掃描
        watcher.scan_changes()

        # 等待一下確保時間戳不同
        time.sleep(0.1)

        # 修改檔案
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("修改後的內容")

        # 第二次掃描
        changes = watcher.scan_changes()
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.MODIFIED
        assert changes[0].old_metadata is not None
        assert changes[0].new_metadata is not None
        assert (
            changes[0].old_metadata.content_hash != changes[0].new_metadata.content_hash
        )

    def test_scan_changes_deleted_files(self, temp_dir):
        """測試掃描刪除檔案變更"""
        watcher = FileWatcher([temp_dir])

        # 建立檔案
        test_file = temp_dir / "delete_test.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("將被刪除的檔案")

        # 第一次掃描
        watcher.scan_changes()

        # 刪除檔案
        test_file.unlink()

        # 第二次掃描
        changes = watcher.scan_changes()
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.DELETED
        assert changes[0].old_metadata is not None
        assert changes[0].new_metadata is None

    def test_change_listener_notification(self, temp_dir):
        """測試變更監聽器通知"""
        watcher = FileWatcher([temp_dir])

        # 記錄監聽器接收到的變更
        received_changes = []

        def change_listener(change):
            received_changes.append(change)

        watcher.add_change_listener(change_listener)

        # 建立新檔案
        new_file = temp_dir / "listener_test.txt"
        with open(new_file, "w", encoding="utf-8") as f:
            f.write("監聽器測試")

        # 掃描變更
        changes = watcher.scan_changes()

        # 驗證監聽器被調用
        assert len(received_changes) == 1
        assert received_changes[0].change_type == ChangeType.ADDED


class TestIncrementalIndexStorage:
    """測試增量索引儲存"""

    def test_storage_initialization(self, temp_dir):
        """測試儲存初始化"""
        db_path = temp_dir / "index.db"
        storage = IncrementalIndexStorage(db_path)

        assert storage.db_path == db_path
        assert db_path.exists()

        # 驗證資料庫表被建立
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            assert "file_metadata" in tables
            assert "change_records" in tables
            assert "index_batches" in tables

    def test_save_and_get_file_metadata(self, temp_dir):
        """測試儲存和獲取檔案元資料"""
        storage = IncrementalIndexStorage(temp_dir / "test.db")

        # 建立測試元資料
        metadata = FileMetadata(
            file_path="/test/file.txt",
            size_bytes=1024,
            modified_time=datetime.now(),
            content_hash="abc123",
            encoding="utf-8",
            document_id="doc_001",
        )

        # 儲存元資料
        storage.save_file_metadata(metadata)

        # 獲取元資料
        retrieved = storage.get_file_metadata("/test/file.txt")

        assert retrieved is not None
        assert retrieved.file_path == metadata.file_path
        assert retrieved.size_bytes == metadata.size_bytes
        assert retrieved.content_hash == metadata.content_hash
        assert retrieved.document_id == metadata.document_id

    def test_save_and_get_change_record(self, temp_dir):
        """測試儲存和獲取變更記錄"""
        storage = IncrementalIndexStorage(temp_dir / "test.db")

        # 建立測試變更
        metadata = FileMetadata(
            file_path="/test/change.txt",
            size_bytes=512,
            modified_time=datetime.now(),
            content_hash="change123",
            encoding="utf-8",
        )

        change = ChangeRecord(
            change_id="change_001",
            file_path="/test/change.txt",
            change_type=ChangeType.ADDED,
            timestamp=datetime.now(),
            new_metadata=metadata,
        )

        # 儲存變更記錄
        storage.save_change_record(change)

        # 獲取待處理變更
        pending_changes = storage.get_pending_changes()

        assert len(pending_changes) == 1
        assert pending_changes[0].change_id == "change_001"
        assert pending_changes[0].change_type == ChangeType.ADDED
        assert pending_changes[0].processed is False

    def test_mark_change_processed(self, temp_dir):
        """測試標記變更為已處理"""
        storage = IncrementalIndexStorage(temp_dir / "test.db")

        # 建立並儲存變更記錄
        change = ChangeRecord(
            change_id="process_test",
            file_path="/test/process.txt",
            change_type=ChangeType.MODIFIED,
            timestamp=datetime.now(),
        )

        storage.save_change_record(change)

        # 驗證為待處理
        pending = storage.get_pending_changes()
        assert len(pending) == 1

        # 標記為已處理
        storage.mark_change_processed("process_test")

        # 驗證不再出現在待處理中
        pending = storage.get_pending_changes()
        assert len(pending) == 0

    def test_pending_changes_limit(self, temp_dir):
        """測試待處理變更限制"""
        storage = IncrementalIndexStorage(temp_dir / "test.db")

        # 建立多個變更記錄
        for i in range(150):  # 超過預設限制
            change = ChangeRecord(
                change_id=f"change_{i:03d}",
                file_path=f"/test/file_{i}.txt",
                change_type=ChangeType.ADDED,
                timestamp=datetime.now(),
            )
            storage.save_change_record(change)

        # 獲取待處理變更（預設限制 100）
        pending = storage.get_pending_changes()
        assert len(pending) == 100

        # 獲取待處理變更（自定義限制）
        pending_50 = storage.get_pending_changes(50)
        assert len(pending_50) == 50


class TestIncrementalIndexManager:
    """測試增量索引管理器"""

    def test_manager_creation(self, temp_dir):
        """測試管理器建立"""
        storage = IncrementalIndexStorage(temp_dir / "manager.db")
        watch_paths = [temp_dir / "watch"]

        manager = IncrementalIndexManager(storage, watch_paths)

        assert manager.storage is storage
        assert len(manager.file_watcher.watch_paths) == 1
        assert manager.batch_size == 50  # 預設值
        assert manager.processing is False

    def test_scan_for_changes(self, temp_dir):
        """測試掃描檔案變更"""
        storage = IncrementalIndexStorage(temp_dir / "scan.db")
        watch_dir = temp_dir / "watch"
        watch_dir.mkdir()

        manager = IncrementalIndexManager(storage, [watch_dir])

        # 建立測試檔案
        test_file = watch_dir / "scan_test.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("掃描測試檔案")

        # 掃描變更
        changes = manager.scan_for_changes()

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.ADDED

        # 驗證變更被儲存到資料庫
        pending = storage.get_pending_changes()
        assert len(pending) == 1

    def test_process_pending_changes(self, temp_dir):
        """測試處理待處理變更"""
        storage = IncrementalIndexStorage(temp_dir / "process.db")
        manager = IncrementalIndexManager(storage, [])

        # 手動添加一些變更記錄
        for i in range(5):
            change = ChangeRecord(
                change_id=f"pending_{i}",
                file_path=f"/test/pending_{i}.txt",
                change_type=ChangeType.ADDED,
                timestamp=datetime.now(),
            )
            storage.save_change_record(change)

        # 處理待處理變更
        results = manager.process_pending_changes(max_batches=1)

        assert results["total_processed"] == 5
        assert results["successful"] == 5
        assert results["failed"] == 0
        assert len(results["batches"]) == 1

        # 驗證變更被標記為已處理
        pending = storage.get_pending_changes()
        assert len(pending) == 0

    def test_processing_concurrency_protection(self, temp_dir):
        """測試處理併發保護"""
        storage = IncrementalIndexStorage(temp_dir / "concurrent.db")
        manager = IncrementalIndexManager(storage, [])

        # 模擬正在處理
        manager.processing = True

        # 嘗試再次處理應該被跳過
        results = manager.process_pending_changes()

        assert results["status"] == "skipped"
        assert results["reason"] == "already_processing"

    def test_file_change_handling(self, temp_dir):
        """測試檔案變更處理"""
        storage = IncrementalIndexStorage(temp_dir / "handle.db")
        watch_dir = temp_dir / "watch"
        watch_dir.mkdir()

        manager = IncrementalIndexManager(storage, [watch_dir])

        # 模擬檔案新增
        add_metadata = FileMetadata(
            file_path=str(watch_dir / "added.txt"),
            size_bytes=100,
            modified_time=datetime.now(),
            content_hash="add123",
            encoding="utf-8",
        )

        add_change = ChangeRecord(
            change_id="add_test",
            file_path=str(watch_dir / "added.txt"),
            change_type=ChangeType.ADDED,
            timestamp=datetime.now(),
            new_metadata=add_metadata,
        )

        # 處理新增變更
        manager._process_single_change(add_change)

        # 驗證檔案元資料被儲存
        stored_metadata = storage.get_file_metadata(str(watch_dir / "added.txt"))
        assert stored_metadata is not None
        assert stored_metadata.content_hash == "add123"

    def test_get_index_statistics(self, temp_dir):
        """測試獲取索引統計"""
        storage = IncrementalIndexStorage(temp_dir / "stats.db")
        manager = IncrementalIndexManager(storage, [])

        # 添加一些測試資料
        metadata = FileMetadata(
            file_path="/test/stats.txt",
            size_bytes=1024,
            modified_time=datetime.now(),
            content_hash="stats123",
            encoding="utf-8",
        )
        storage.save_file_metadata(metadata)

        change = ChangeRecord(
            change_id="stats_change",
            file_path="/test/stats.txt",
            change_type=ChangeType.ADDED,
            timestamp=datetime.now(),
        )
        storage.save_change_record(change)

        # 獲取統計資訊
        stats = manager.get_index_statistics()

        assert "files" in stats
        assert "changes" in stats
        assert stats["files"]["total"] >= 1
        assert stats["changes"]["pending"] >= 1
        assert stats["processing"] is False

    def test_cleanup_old_records(self, temp_dir):
        """測試清理舊記錄"""
        storage = IncrementalIndexStorage(temp_dir / "cleanup.db")
        manager = IncrementalIndexManager(storage, [])

        # 添加一些舊的已處理記錄
        old_timestamp = datetime.now() - timedelta(days=35)

        for i in range(3):
            change = ChangeRecord(
                change_id=f"old_{i}",
                file_path=f"/test/old_{i}.txt",
                change_type=ChangeType.ADDED,
                timestamp=old_timestamp,
                processed=True,
            )
            storage.save_change_record(change)

        # 添加一些新記錄
        for i in range(2):
            change = ChangeRecord(
                change_id=f"new_{i}",
                file_path=f"/test/new_{i}.txt",
                change_type=ChangeType.ADDED,
                timestamp=datetime.now(),
                processed=True,
            )
            storage.save_change_record(change)

        # 清理 30 天前的記錄
        deleted_count = manager.cleanup_old_records(days=30)

        assert deleted_count >= 3  # 應該刪除舊記錄


@pytest.mark.integration
class TestIncrementalIndexingIntegration:
    """測試增量索引整合"""

    def test_full_incremental_workflow(self, temp_dir):
        """測試完整增量索引工作流程"""
        # 設定增量索引管理器
        storage = IncrementalIndexStorage(temp_dir / "workflow.db")
        watch_dir = temp_dir / "documents"
        watch_dir.mkdir()

        manager = IncrementalIndexManager(storage, [watch_dir], batch_size=10)

        # 第一階段：建立初始檔案
        initial_files = []
        for i in range(5):
            file_path = watch_dir / f"doc_{i}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"這是文件 {i} 的內容。包含一些中文文本用於測試。")
            initial_files.append(file_path)

        # 掃描初始變更
        changes = manager.scan_for_changes()
        assert len(changes) == 5
        assert all(c.change_type == ChangeType.ADDED for c in changes)

        # 處理初始變更
        results = manager.process_pending_changes()
        assert results["total_processed"] == 5
        assert results["successful"] == 5

        # 第二階段：修改一些檔案
        modified_files = initial_files[:2]
        for i, file_path in enumerate(modified_files):
            time.sleep(0.1)  # 確保時間戳不同
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(f"\n修改 {i}：添加了新內容。")

        # 掃描修改變更
        changes = manager.scan_for_changes()
        assert len(changes) == 2
        assert all(c.change_type == ChangeType.MODIFIED for c in changes)

        # 處理修改變更
        results = manager.process_pending_changes()
        assert results["total_processed"] == 2
        assert results["successful"] == 2

        # 第三階段：刪除一個檔案
        deleted_file = initial_files[-1]
        deleted_file.unlink()

        # 掃描刪除變更
        changes = manager.scan_for_changes()
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.DELETED

        # 處理刪除變更
        results = manager.process_pending_changes()
        assert results["total_processed"] == 1
        assert results["successful"] == 1

        # 第四階段：驗證最終狀態
        stats = manager.get_index_statistics()
        assert stats["files"]["total"] >= 4  # 5 個初始 - 1 個刪除

    def test_concurrent_file_operations(self, temp_dir):
        """測試併發檔案操作"""
        storage = IncrementalIndexStorage(temp_dir / "concurrent.db")
        watch_dir = temp_dir / "concurrent"
        watch_dir.mkdir()

        manager = IncrementalIndexManager(storage, [watch_dir])

        # 使用多執行緒同時建立檔案
        import threading

        def create_files(thread_id, count):
            for i in range(count):
                file_path = watch_dir / f"thread_{thread_id}_file_{i}.txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"執行緒 {thread_id} 建立的檔案 {i}")

        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=create_files, args=(thread_id, 5))
            threads.append(thread)
            thread.start()

        # 等待所有執行緒完成
        for thread in threads:
            thread.join()

        # 掃描所有變更
        changes = manager.scan_for_changes()
        assert len(changes) == 15  # 3 執行緒 × 5 檔案

        # 處理所有變更
        results = manager.process_pending_changes()
        assert results["total_processed"] == 15
        assert results["successful"] == 15

    def test_error_handling_in_processing(self, temp_dir):
        """測試處理過程中的錯誤處理"""
        storage = IncrementalIndexStorage(temp_dir / "error.db")
        manager = IncrementalIndexManager(storage, [])

        # 建立一個會導致處理錯誤的變更記錄
        error_change = ChangeRecord(
            change_id="error_test",
            file_path="/non/existent/path.txt",
            change_type=ChangeType.ADDED,
            timestamp=datetime.now(),
        )

        storage.save_change_record(error_change)

        # 模擬處理錯誤
        with patch.object(
            manager, "_handle_file_added", side_effect=Exception("處理錯誤")
        ):
            results = manager.process_pending_changes()

        assert results["total_processed"] == 1
        assert results["failed"] == 1
        assert len(results["batches"][0]["errors"]) == 1

        # 驗證錯誤記錄被標記為已處理（帶錯誤訊息）
        pending = storage.get_pending_changes()
        assert len(pending) == 0


if __name__ == "__main__":
    pytest.main([__file__])
