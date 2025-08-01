"""
恢復和檢查點管理

提供系統狀態恢復、檢查點管理和資料一致性檢查功能。
"""

import json
import logging
import pickle
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
import hashlib

from .base import ChineseGraphRAGError, SystemError


logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CheckpointMetadata:
    """檢查點元資料"""
    checkpoint_id: str
    timestamp: datetime
    operation: str
    version: str
    checksum: str
    size_bytes: int
    description: str
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """從字典創建"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class SystemState:
    """系統狀態"""
    state_id: str
    timestamp: datetime
    component_states: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'state_id': self.state_id,
            'timestamp': self.timestamp.isoformat(),
            'component_states': self.component_states,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """從字典創建"""
        return cls(
            state_id=data['state_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            component_states=data['component_states'],
            metadata=data['metadata']
        )


class CheckpointStorage(ABC):
    """檢查點儲存抽象基類"""
    
    @abstractmethod
    def save_checkpoint(
        self,
        checkpoint_id: str,
        data: Any,
        metadata: CheckpointMetadata
    ) -> bool:
        """儲存檢查點"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Optional[tuple[Any, CheckpointMetadata]]:
        """載入檢查點"""
        pass
    
    @abstractmethod
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """列出所有檢查點"""
        pass
    
    @abstractmethod
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """刪除檢查點"""
        pass


class FileCheckpointStorage(CheckpointStorage):
    """檔案系統檢查點儲存"""
    
    def __init__(self, storage_path: Union[str, Path]):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """載入元資料"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = {
                        k: CheckpointMetadata.from_dict(v) 
                        for k, v in data.items()
                    }
            except Exception as e:
                logger.error(f"載入檢查點元資料失敗: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """儲存元資料"""
        try:
            data = {k: v.to_dict() for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"儲存檢查點元資料失敗: {e}")
    
    def _calculate_checksum(self, data: bytes) -> str:
        """計算校驗和"""
        return hashlib.sha256(data).hexdigest()
    
    def save_checkpoint(
        self,
        checkpoint_id: str,
        data: Any,
        metadata: CheckpointMetadata
    ) -> bool:
        """儲存檢查點"""
        try:
            checkpoint_file = self.storage_path / f"{checkpoint_id}.pkl"
            
            # 序列化資料
            serialized_data = pickle.dumps(data)
            
            # 計算校驗和
            checksum = self._calculate_checksum(serialized_data)
            metadata.checksum = checksum
            metadata.size_bytes = len(serialized_data)
            
            # 儲存資料
            with open(checkpoint_file, 'wb') as f:
                f.write(serialized_data)
            
            # 更新元資料
            self.metadata[checkpoint_id] = metadata
            self._save_metadata()
            
            logger.info(f"檢查點已儲存: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"儲存檢查點失敗 {checkpoint_id}: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[tuple[Any, CheckpointMetadata]]:
        """載入檢查點"""
        try:
            if checkpoint_id not in self.metadata:
                logger.warning(f"檢查點不存在: {checkpoint_id}")
                return None
            
            checkpoint_file = self.storage_path / f"{checkpoint_id}.pkl"
            if not checkpoint_file.exists():
                logger.error(f"檢查點檔案不存在: {checkpoint_file}")
                return None
            
            # 載入資料
            with open(checkpoint_file, 'rb') as f:
                serialized_data = f.read()
            
            # 驗證校驗和
            metadata = self.metadata[checkpoint_id]
            expected_checksum = metadata.checksum
            actual_checksum = self._calculate_checksum(serialized_data)
            
            if expected_checksum != actual_checksum:
                logger.error(f"檢查點校驗和不匹配: {checkpoint_id}")
                return None
            
            # 反序列化資料
            data = pickle.loads(serialized_data)
            
            logger.info(f"檢查點已載入: {checkpoint_id}")
            return data, metadata
            
        except Exception as e:
            logger.error(f"載入檢查點失敗 {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """列出所有檢查點"""
        return list(self.metadata.values())
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """刪除檢查點"""
        try:
            if checkpoint_id not in self.metadata:
                logger.warning(f"檢查點不存在: {checkpoint_id}")
                return False
            
            checkpoint_file = self.storage_path / f"{checkpoint_id}.pkl"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            del self.metadata[checkpoint_id]
            self._save_metadata()
            
            logger.info(f"檢查點已刪除: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"刪除檢查點失敗 {checkpoint_id}: {e}")
            return False


class CheckpointManager:
    """檢查點管理器"""
    
    def __init__(self, storage: CheckpointStorage):
        self.storage = storage
        self._lock = threading.RLock()
    
    def create_checkpoint(
        self,
        data: Any,
        operation: str,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """建立檢查點"""
        with self._lock:
            checkpoint_id = f"cp_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
            
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now(),
                operation=operation,
                version="1.0",
                checksum="",  # 將在儲存時計算
                size_bytes=0,  # 將在儲存時計算
                description=description,
                tags=tags or []
            )
            
            if self.storage.save_checkpoint(checkpoint_id, data, metadata):
                logger.info(f"檢查點已建立: {checkpoint_id} - {operation}")
                return checkpoint_id
            else:
                raise SystemError(f"無法建立檢查點: {operation}")
    
    def restore_checkpoint(self, checkpoint_id: str) -> Any:
        """恢復檢查點"""
        with self._lock:
            result = self.storage.load_checkpoint(checkpoint_id)
            if result is None:
                raise SystemError(f"無法載入檢查點: {checkpoint_id}")
            
            data, metadata = result
            logger.info(f"檢查點已恢復: {checkpoint_id} - {metadata.operation}")
            return data
    
    def list_checkpoints(
        self,
        operation_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        max_age_hours: Optional[int] = None
    ) -> List[CheckpointMetadata]:
        """列出檢查點"""
        checkpoints = self.storage.list_checkpoints()
        
        # 應用過濾器
        if operation_filter:
            checkpoints = [cp for cp in checkpoints if operation_filter in cp.operation]
        
        if tag_filter:
            checkpoints = [cp for cp in checkpoints if tag_filter in cp.tags]
        
        if max_age_hours:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            checkpoints = [cp for cp in checkpoints if cp.timestamp >= cutoff_time]
        
        # 按時間排序
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        return checkpoints
    
    def cleanup_old_checkpoints(
        self,
        max_age_days: int = 7,
        keep_minimum: int = 3
    ) -> int:
        """清理過期檢查點"""
        with self._lock:
            all_checkpoints = self.storage.list_checkpoints()
            all_checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
            
            if len(all_checkpoints) <= keep_minimum:
                return 0
            
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            # 保留最新的檢查點和未過期的檢查點
            to_keep = set()
            to_delete = []
            
            for i, checkpoint in enumerate(all_checkpoints):
                if i < keep_minimum or checkpoint.timestamp >= cutoff_time:
                    to_keep.add(checkpoint.checkpoint_id)
                else:
                    to_delete.append(checkpoint.checkpoint_id)
            
            deleted_count = 0
            for checkpoint_id in to_delete:
                if self.storage.delete_checkpoint(checkpoint_id):
                    deleted_count += 1
            
            logger.info(f"清理了 {deleted_count} 個過期檢查點")
            return deleted_count
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """獲取檢查點資訊"""
        checkpoints = self.storage.list_checkpoints()
        for checkpoint in checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None


class StateManager:
    """系統狀態管理器"""
    
    def __init__(self):
        self.current_state: Optional[SystemState] = None
        self.state_history: List[SystemState] = []
        self.component_handlers: Dict[str, Callable] = {}
        self._lock = threading.RLock()
    
    def register_component(self, name: str, state_handler: Callable[[], Any]):
        """註冊元件狀態處理器"""
        with self._lock:
            self.component_handlers[name] = state_handler
            logger.info(f"註冊元件狀態處理器: {name}")
    
    def capture_current_state(self) -> SystemState:
        """捕獲當前系統狀態"""
        with self._lock:
            component_states = {}
            
            for name, handler in self.component_handlers.items():
                try:
                    component_states[name] = handler()
                except Exception as e:
                    logger.error(f"捕獲元件 {name} 狀態失敗: {e}")
                    component_states[name] = {"error": str(e)}
            
            state = SystemState(
                state_id=f"state_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(),
                component_states=component_states,
                metadata={
                    "captured_components": list(component_states.keys()),
                    "capture_method": "manual"
                }
            )
            
            self.current_state = state
            self.state_history.append(state)
            
            # 限制歷史記錄數量
            if len(self.state_history) > 100:
                self.state_history = self.state_history[-100:]
            
            logger.info(f"系統狀態已捕獲: {state.state_id}")
            return state
    
    def get_current_state(self) -> Optional[SystemState]:
        """獲取當前狀態"""
        return self.current_state
    
    def get_state_history(self, limit: int = 10) -> List[SystemState]:
        """獲取狀態歷史"""
        with self._lock:
            return self.state_history[-limit:]
    
    def compare_states(
        self,
        state1: SystemState,
        state2: SystemState
    ) -> Dict[str, Any]:
        """比較兩個系統狀態"""
        differences = {
            "timestamp_diff": (state2.timestamp - state1.timestamp).total_seconds(),
            "component_changes": {},
            "added_components": [],
            "removed_components": [],
        }
        
        components1 = set(state1.component_states.keys())
        components2 = set(state2.component_states.keys())
        
        differences["added_components"] = list(components2 - components1)
        differences["removed_components"] = list(components1 - components2)
        
        for component in components1.intersection(components2):
            state1_data = state1.component_states[component]
            state2_data = state2.component_states[component]
            
            if state1_data != state2_data:
                differences["component_changes"][component] = {
                    "before": state1_data,
                    "after": state2_data
                }
        
        return differences


class RecoveryManager:
    """恢復管理器"""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        state_manager: StateManager
    ):
        self.checkpoint_manager = checkpoint_manager
        self.state_manager = state_manager
        self.recovery_strategies: Dict[str, Callable] = {}
        self._lock = threading.RLock()
    
    def register_recovery_strategy(
        self,
        error_type: str,
        strategy: Callable[[Exception, Dict[str, Any]], Any]
    ):
        """註冊恢復策略"""
        with self._lock:
            self.recovery_strategies[error_type] = strategy
            logger.info(f"註冊恢復策略: {error_type}")
    
    def auto_recover(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """自動恢復"""
        with self._lock:
            error_type = type(error).__name__
            
            # 捕獲當前狀態
            current_state = self.state_manager.capture_current_state()
            
            # 查找合適的恢復策略
            strategy = self.recovery_strategies.get(error_type)
            if not strategy:
                # 嘗試通用策略
                strategy = self.recovery_strategies.get("generic")
            
            if not strategy:
                logger.warning(f"沒有找到 {error_type} 的恢復策略")
                return False
            
            try:
                logger.info(f"開始自動恢復: {error_type}")
                result = strategy(error, context or {})
                logger.info(f"自動恢復成功: {error_type}")
                return True
                
            except Exception as recovery_error:
                logger.error(f"自動恢復失敗: {recovery_error}")
                return False
    
    def create_recovery_checkpoint(
        self,
        operation: str,
        data: Any
    ) -> str:
        """建立恢復檢查點"""
        return self.checkpoint_manager.create_checkpoint(
            data=data,
            operation=f"recovery_{operation}",
            description=f"恢復用檢查點: {operation}",
            tags=["recovery", operation]
        )
    
    def perform_recovery(
        self,
        checkpoint_id: str,
        recovery_steps: Optional[List[Callable]] = None
    ) -> bool:
        """執行恢復操作"""
        with self._lock:
            try:
                # 恢復檢查點資料
                data = self.checkpoint_manager.restore_checkpoint(checkpoint_id)
                
                # 執行自訂恢復步驟
                if recovery_steps:
                    for step in recovery_steps:
                        step(data)
                
                logger.info(f"恢復操作完成: {checkpoint_id}")
                return True
                
            except Exception as e:
                logger.error(f"恢復操作失敗: {e}")
                return False


# 全域實例
_recovery_manager = None
_recovery_lock = threading.Lock()


def get_recovery_manager() -> RecoveryManager:
    """獲取全域恢復管理器"""
    global _recovery_manager
    
    if _recovery_manager is None:
        with _recovery_lock:
            if _recovery_manager is None:
                # 使用預設配置
                storage_path = Path("./data/checkpoints")
                storage = FileCheckpointStorage(storage_path)
                checkpoint_manager = CheckpointManager(storage)
                state_manager = StateManager()
                
                _recovery_manager = RecoveryManager(checkpoint_manager, state_manager)
    
    return _recovery_manager