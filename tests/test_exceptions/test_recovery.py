"""
測試檢查點管理和恢復機制
"""

import json
import pickle
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from src.chinese_graphrag.exceptions.recovery import (
    CheckpointManager,
    CheckpointMetadata,
    FileCheckpointStorage,
    RecoveryManager,
    StateManager,
    SystemState,
    get_recovery_manager,
)


class TestCheckpointMetadata:
    """測試檢查點元資料"""

    def test_metadata_creation(self):
        """測試元資料建立"""
        metadata = CheckpointMetadata(
            checkpoint_id="cp_001",
            timestamp=datetime.now(),
            operation="test_operation",
            version="1.0",
            checksum="abc123",
            size_bytes=1024,
            description="測試檢查點",
            tags=["test", "checkpoint"],
        )

        assert metadata.checkpoint_id == "cp_001"
        assert metadata.operation == "test_operation"
        assert metadata.size_bytes == 1024
        assert "test" in metadata.tags

    def test_metadata_serialization(self):
        """測試元資料序列化"""
        timestamp = datetime.now()
        metadata = CheckpointMetadata(
            checkpoint_id="cp_002",
            timestamp=timestamp,
            operation="serialize_test",
            version="1.0",
            checksum="def456",
            size_bytes=2048,
            description="序列化測試",
            tags=["serialize"],
        )

        # 轉換為字典
        data = metadata.to_dict()
        assert data["checkpoint_id"] == "cp_002"
        assert data["timestamp"] == timestamp.isoformat()

        # 從字典恢復
        restored = CheckpointMetadata.from_dict(data)
        assert restored.checkpoint_id == metadata.checkpoint_id
        assert restored.timestamp == metadata.timestamp


class TestSystemState:
    """測試系統狀態"""

    def test_state_creation(self):
        """測試狀態建立"""
        state = SystemState(
            state_id="state_001",
            timestamp=datetime.now(),
            component_states={
                "database": {"status": "connected", "connections": 5},
                "cache": {"status": "active", "memory_usage": "256MB"},
            },
            metadata={"capture_method": "manual", "version": "1.0"},
        )

        assert state.state_id == "state_001"
        assert "database" in state.component_states
        assert state.metadata["capture_method"] == "manual"

    def test_state_serialization(self):
        """測試狀態序列化"""
        timestamp = datetime.now()
        state = SystemState(
            state_id="state_002",
            timestamp=timestamp,
            component_states={"test_component": {"value": 42}},
            metadata={"test": True},
        )

        # 轉換為字典
        data = state.to_dict()
        assert data["state_id"] == "state_002"
        assert data["timestamp"] == timestamp.isoformat()

        # 從字典恢復
        restored = SystemState.from_dict(data)
        assert restored.state_id == state.state_id
        assert restored.component_states == state.component_states


class TestFileCheckpointStorage:
    """測試檔案檢查點儲存"""

    def test_storage_initialization(self, temp_dir):
        """測試儲存初始化"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")

        assert storage.storage_path.exists()
        assert storage.metadata_file.exists()
        assert isinstance(storage.metadata, dict)

    def test_save_and_load_checkpoint(self, temp_dir):
        """測試儲存和載入檢查點"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")

        # 準備測試資料
        test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": "data"}}

        metadata = CheckpointMetadata(
            checkpoint_id="test_cp",
            timestamp=datetime.now(),
            operation="test_save",
            version="1.0",
            checksum="",  # 將在儲存時計算
            size_bytes=0,  # 將在儲存時計算
            description="測試儲存",
            tags=["test"],
        )

        # 儲存檢查點
        success = storage.save_checkpoint("test_cp", test_data, metadata)
        assert success is True

        # 載入檢查點
        result = storage.load_checkpoint("test_cp")
        assert result is not None

        loaded_data, loaded_metadata = result
        assert loaded_data == test_data
        assert loaded_metadata.checkpoint_id == "test_cp"
        assert loaded_metadata.operation == "test_save"

    def test_list_checkpoints(self, temp_dir):
        """測試列出檢查點"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")

        # 建立多個檢查點
        for i in range(3):
            metadata = CheckpointMetadata(
                checkpoint_id=f"cp_{i}",
                timestamp=datetime.now(),
                operation=f"operation_{i}",
                version="1.0",
                checksum="",
                size_bytes=0,
                description=f"檢查點 {i}",
                tags=[f"tag_{i}"],
            )
            storage.save_checkpoint(f"cp_{i}", {"data": i}, metadata)

        # 列出檢查點
        checkpoints = storage.list_checkpoints()
        assert len(checkpoints) == 3

        checkpoint_ids = [cp.checkpoint_id for cp in checkpoints]
        assert "cp_0" in checkpoint_ids
        assert "cp_1" in checkpoint_ids
        assert "cp_2" in checkpoint_ids

    def test_delete_checkpoint(self, temp_dir):
        """測試刪除檢查點"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")

        # 建立檢查點
        metadata = CheckpointMetadata(
            checkpoint_id="delete_me",
            timestamp=datetime.now(),
            operation="delete_test",
            version="1.0",
            checksum="",
            size_bytes=0,
            description="將被刪除的檢查點",
            tags=["delete"],
        )

        storage.save_checkpoint("delete_me", {"data": "delete"}, metadata)

        # 確認存在
        result = storage.load_checkpoint("delete_me")
        assert result is not None

        # 刪除檢查點
        success = storage.delete_checkpoint("delete_me")
        assert success is True

        # 確認已刪除
        result = storage.load_checkpoint("delete_me")
        assert result is None

    def test_checksum_validation(self, temp_dir):
        """測試校驗和驗證"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")

        test_data = {"test": "checksum"}
        metadata = CheckpointMetadata(
            checkpoint_id="checksum_test",
            timestamp=datetime.now(),
            operation="checksum",
            version="1.0",
            checksum="",
            size_bytes=0,
            description="校驗和測試",
            tags=["checksum"],
        )

        # 儲存檢查點
        storage.save_checkpoint("checksum_test", test_data, metadata)

        # 人為破壞檢查點檔案
        checkpoint_file = storage.storage_path / "checksum_test.pkl"
        with open(checkpoint_file, "ab") as f:
            f.write(b"corrupt_data")

        # 嘗試載入應該失敗
        result = storage.load_checkpoint("checksum_test")
        assert result is None


class TestCheckpointManager:
    """測試檢查點管理器"""

    def test_manager_creation(self, temp_dir):
        """測試管理器建立"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        manager = CheckpointManager(storage)

        assert manager.storage is storage

    def test_create_checkpoint(self, temp_dir):
        """測試建立檢查點"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        manager = CheckpointManager(storage)

        test_data = {"operation": "test", "data": [1, 2, 3]}

        checkpoint_id = manager.create_checkpoint(
            data=test_data,
            operation="test_operation",
            description="測試檢查點建立",
            tags=["test", "create"],
        )

        assert checkpoint_id is not None
        assert checkpoint_id.startswith("cp_")

        # 驗證檢查點可以載入
        restored_data = manager.restore_checkpoint(checkpoint_id)
        assert restored_data == test_data

    def test_list_checkpoints_with_filters(self, temp_dir):
        """測試帶過濾器的檢查點列表"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        manager = CheckpointManager(storage)

        # 建立不同操作的檢查點
        manager.create_checkpoint(
            data={"data": 1}, operation="indexing", tags=["index", "test"]
        )

        manager.create_checkpoint(
            data={"data": 2}, operation="query", tags=["query", "test"]
        )

        manager.create_checkpoint(
            data={"data": 3}, operation="indexing", tags=["index", "production"]
        )

        # 按操作過濾
        indexing_checkpoints = manager.list_checkpoints(operation_filter="indexing")
        assert len(indexing_checkpoints) == 2

        # 按標籤過濾
        test_checkpoints = manager.list_checkpoints(tag_filter="test")
        assert len(test_checkpoints) == 2

        # 按時間過濾
        recent_checkpoints = manager.list_checkpoints(max_age_hours=1)
        assert len(recent_checkpoints) == 3

    def test_cleanup_old_checkpoints(self, temp_dir):
        """測試清理過期檢查點"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        manager = CheckpointManager(storage)

        # 建立多個檢查點
        checkpoint_ids = []
        for i in range(5):
            checkpoint_id = manager.create_checkpoint(
                data={"index": i}, operation=f"operation_{i}"
            )
            checkpoint_ids.append(checkpoint_id)

        # 模擬一些檢查點是舊的
        old_checkpoints = checkpoint_ids[:2]
        for checkpoint_id in old_checkpoints:
            # 修改元資料使其看起來很舊
            metadata = storage.metadata[checkpoint_id]
            metadata.timestamp = datetime.now() - timedelta(days=10)
            storage._save_metadata()

        # 清理過期檢查點，但保留最少 3 個
        deleted_count = manager.cleanup_old_checkpoints(max_age_days=7, keep_minimum=3)

        assert deleted_count == 2

        # 驗證剩餘檢查點
        remaining_checkpoints = manager.list_checkpoints()
        assert len(remaining_checkpoints) == 3


class TestStateManager:
    """測試狀態管理器"""

    def test_state_manager_creation(self):
        """測試狀態管理器建立"""
        manager = StateManager()

        assert manager.current_state is None
        assert len(manager.state_history) == 0
        assert len(manager.component_handlers) == 0

    def test_register_component(self):
        """測試註冊元件"""
        manager = StateManager()

        def database_state():
            return {"connections": 5, "status": "healthy"}

        def cache_state():
            return {"memory_usage": "256MB", "hit_rate": 0.85}

        manager.register_component("database", database_state)
        manager.register_component("cache", cache_state)

        assert len(manager.component_handlers) == 2
        assert "database" in manager.component_handlers
        assert "cache" in manager.component_handlers

    def test_capture_current_state(self):
        """測試捕獲當前狀態"""
        manager = StateManager()

        # 註冊元件處理器
        manager.register_component(
            "test_component", lambda: {"value": 42, "status": "active"}
        )

        # 捕獲狀態
        state = manager.capture_current_state()

        assert state is not None
        assert state.state_id is not None
        assert "test_component" in state.component_states
        assert state.component_states["test_component"]["value"] == 42

        # 驗證狀態被記錄
        assert manager.current_state is state
        assert len(manager.state_history) == 1

    def test_component_error_handling(self):
        """測試元件錯誤處理"""
        manager = StateManager()

        def failing_component():
            raise Exception("元件故障")

        def working_component():
            return {"status": "ok"}

        manager.register_component("failing", failing_component)
        manager.register_component("working", working_component)

        # 捕獲狀態應該處理錯誤
        state = manager.capture_current_state()

        assert "failing" in state.component_states
        assert "error" in state.component_states["failing"]
        assert state.component_states["working"]["status"] == "ok"

    def test_state_history_limit(self):
        """測試狀態歷史限制"""
        manager = StateManager()

        manager.register_component("counter", lambda: {"count": 1})

        # 建立超過限制的狀態
        for i in range(105):  # 超過預設限制 100
            manager.capture_current_state()

        # 驗證歷史記錄被限制
        assert len(manager.state_history) == 100

    def test_compare_states(self):
        """測試狀態比較"""
        manager = StateManager()

        counter = 0

        def counter_component():
            return {"count": counter}

        manager.register_component("counter", counter_component)

        # 捕獲第一個狀態
        state1 = manager.capture_current_state()

        # 修改元件狀態
        counter = 5

        # 捕獲第二個狀態
        state2 = manager.capture_current_state()

        # 比較狀態
        differences = manager.compare_states(state1, state2)

        assert "component_changes" in differences
        assert "counter" in differences["component_changes"]
        assert differences["component_changes"]["counter"]["before"]["count"] == 0
        assert differences["component_changes"]["counter"]["after"]["count"] == 5


class TestRecoveryManager:
    """測試恢復管理器"""

    def test_recovery_manager_creation(self, temp_dir):
        """測試恢復管理器建立"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        checkpoint_manager = CheckpointManager(storage)
        state_manager = StateManager()

        recovery_manager = RecoveryManager(checkpoint_manager, state_manager)

        assert recovery_manager.checkpoint_manager is checkpoint_manager
        assert recovery_manager.state_manager is state_manager
        assert len(recovery_manager.recovery_strategies) == 0

    def test_register_recovery_strategy(self, temp_dir):
        """測試註冊恢復策略"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        checkpoint_manager = CheckpointManager(storage)
        state_manager = StateManager()
        recovery_manager = RecoveryManager(checkpoint_manager, state_manager)

        def network_recovery(error, context):
            return "網路已恢復"

        recovery_manager.register_recovery_strategy("NetworkError", network_recovery)

        assert "NetworkError" in recovery_manager.recovery_strategies
        assert recovery_manager.recovery_strategies["NetworkError"] is network_recovery

    def test_auto_recover_success(self, temp_dir):
        """測試自動恢復成功"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        checkpoint_manager = CheckpointManager(storage)
        state_manager = StateManager()
        recovery_manager = RecoveryManager(checkpoint_manager, state_manager)

        # 註冊恢復策略
        def test_recovery(error, context):
            return f"已恢復: {error}"

        recovery_manager.register_recovery_strategy("TestError", test_recovery)

        # 建立測試錯誤
        class TestError(Exception):
            pass

        test_error = TestError("測試錯誤")

        # 執行自動恢復
        success = recovery_manager.auto_recover(test_error)

        assert success is True

    def test_auto_recover_no_strategy(self, temp_dir):
        """測試沒有恢復策略的情況"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        checkpoint_manager = CheckpointManager(storage)
        state_manager = StateManager()
        recovery_manager = RecoveryManager(checkpoint_manager, state_manager)

        # 沒有註冊任何策略
        test_error = Exception("未知錯誤")

        # 自動恢復應該失敗
        success = recovery_manager.auto_recover(test_error)

        assert success is False

    def test_create_recovery_checkpoint(self, temp_dir):
        """測試建立恢復檢查點"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        checkpoint_manager = CheckpointManager(storage)
        state_manager = StateManager()
        recovery_manager = RecoveryManager(checkpoint_manager, state_manager)

        test_data = {"recovery": "data", "timestamp": "2024-01-01"}

        checkpoint_id = recovery_manager.create_recovery_checkpoint(
            "test_recovery", test_data
        )

        assert checkpoint_id is not None

        # 驗證檢查點包含恢復標籤
        metadata = checkpoint_manager.get_checkpoint_info(checkpoint_id)
        assert "recovery" in metadata.tags
        assert "test_recovery" in metadata.tags

    def test_perform_recovery(self, temp_dir):
        """測試執行恢復操作"""
        storage = FileCheckpointStorage(temp_dir / "checkpoints")
        checkpoint_manager = CheckpointManager(storage)
        state_manager = StateManager()
        recovery_manager = RecoveryManager(checkpoint_manager, state_manager)

        # 建立恢復檢查點
        recovery_data = {"state": "saved", "components": ["db", "cache"]}
        checkpoint_id = recovery_manager.create_recovery_checkpoint(
            "full_recovery", recovery_data
        )

        # 定義恢復步驟
        executed_steps = []

        def recovery_step_1(data):
            executed_steps.append("step_1")
            assert data["state"] == "saved"

        def recovery_step_2(data):
            executed_steps.append("step_2")
            assert "components" in data

        recovery_steps = [recovery_step_1, recovery_step_2]

        # 執行恢復
        success = recovery_manager.perform_recovery(checkpoint_id, recovery_steps)

        assert success is True
        assert len(executed_steps) == 2
        assert "step_1" in executed_steps
        assert "step_2" in executed_steps


class TestGlobalRecoveryManager:
    """測試全域恢復管理器"""

    def test_get_recovery_manager(self):
        """測試獲取全域恢復管理器"""
        manager1 = get_recovery_manager()
        manager2 = get_recovery_manager()

        # 驗證單例模式
        assert manager1 is manager2
        assert isinstance(manager1, RecoveryManager)


@pytest.mark.integration
class TestRecoveryIntegration:
    """測試恢復機制整合"""

    def test_full_recovery_scenario(self, temp_dir):
        """測試完整恢復場景"""
        # 設定恢復管理器
        storage = FileCheckpointStorage(temp_dir / "recovery")
        checkpoint_manager = CheckpointManager(storage)
        state_manager = StateManager()
        recovery_manager = RecoveryManager(checkpoint_manager, state_manager)

        # 註冊系統元件
        system_state = {"database": "connected", "cache": "active"}

        def get_db_state():
            return {"status": system_state["database"], "connections": 10}

        def get_cache_state():
            return {"status": system_state["cache"], "memory": "512MB"}

        state_manager.register_component("database", get_db_state)
        state_manager.register_component("cache", get_cache_state)

        # 捕獲正常狀態
        normal_state = state_manager.capture_current_state()

        # 建立恢復檢查點
        checkpoint_id = recovery_manager.create_recovery_checkpoint(
            "system_backup",
            {
                "system_state": normal_state.to_dict(),
                "config": {"version": "1.0", "environment": "test"},
            },
        )

        # 模擬系統故障
        system_state["database"] = "disconnected"
        system_state["cache"] = "failed"

        # 捕獲故障狀態
        failed_state = state_manager.capture_current_state()

        # 註冊恢復策略
        def system_recovery(error, context):
            # 恢復系統狀態
            system_state["database"] = "connected"
            system_state["cache"] = "active"
            return "系統已恢復"

        recovery_manager.register_recovery_strategy("SystemError", system_recovery)

        # 執行恢復
        from src.chinese_graphrag.exceptions.base import SystemError

        system_error = SystemError("系統故障")

        success = recovery_manager.auto_recover(system_error)
        assert success is True

        # 驗證系統已恢復
        recovered_state = state_manager.capture_current_state()
        assert recovered_state.component_states["database"]["status"] == "connected"
        assert recovered_state.component_states["cache"]["status"] == "active"


if __name__ == "__main__":
    pytest.main([__file__])
