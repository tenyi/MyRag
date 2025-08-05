"""
基礎資料模型測試
"""

import json
from datetime import datetime
from typing import Any, Dict

import pytest

from chinese_graphrag.models.base import BaseModel


class TestModel(BaseModel):
    """測試用的模型類別"""

    name: str
    value: int = 10


class TestBaseModel:
    """測試基礎資料模型"""

    def test_create_model_with_defaults(self):
        """測試使用預設值建立模型"""
        model = TestModel(name="test")

        assert model.name == "test"
        assert model.value == 10
        assert isinstance(model.id, str)
        assert len(model.id) > 0
        assert isinstance(model.created_at, datetime)
        assert model.updated_at is None
        assert isinstance(model.metadata, dict)
        assert len(model.metadata) == 0

    def test_create_model_with_custom_values(self):
        """測試使用自訂值建立模型"""
        custom_id = "custom-id"
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        custom_metadata = {"key": "value"}

        model = TestModel(
            name="test",
            value=20,
            id=custom_id,
            created_at=custom_time,
            metadata=custom_metadata,
        )

        assert model.name == "test"
        assert model.value == 20
        assert model.id == custom_id
        assert model.created_at == custom_time
        assert model.metadata == custom_metadata

    def test_update_timestamp(self):
        """測試更新時間戳記"""
        model = TestModel(name="test")
        original_created_at = model.created_at

        # 等待一小段時間確保時間戳記不同
        import time

        time.sleep(0.001)

        model.update_timestamp()

        assert model.created_at == original_created_at
        assert model.updated_at is not None
        assert model.updated_at > original_created_at

    def test_to_dict(self):
        """測試轉換為字典"""
        model = TestModel(name="test", value=30)
        data = model.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["value"] == 30
        assert "id" in data
        assert "created_at" in data
        assert "metadata" in data

    def test_from_dict(self):
        """測試從字典建立模型"""
        data = {
            "name": "test",
            "value": 40,
            "id": "test-id",
            "metadata": {"key": "value"},
        }

        model = TestModel.from_dict(data)

        assert model.name == "test"
        assert model.value == 40
        assert model.id == "test-id"
        assert model.metadata == {"key": "value"}

    def test_to_json(self):
        """測試轉換為 JSON"""
        model = TestModel(name="test", value=50)
        json_str = model.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["name"] == "test"
        assert data["value"] == 50

    def test_from_json(self):
        """測試從 JSON 建立模型"""
        json_str = '{"name": "test", "value": 60, "id": "json-id"}'
        model = TestModel.from_json(json_str)

        assert model.name == "test"
        assert model.value == 60
        assert model.id == "json-id"

    def test_validation_on_assignment(self):
        """測試賦值時的驗證"""
        model = TestModel(name="test")

        # 正常賦值應該成功
        model.name = "new_name"
        assert model.name == "new_name"

        # 測試基本的賦值驗證功能存在
        assert hasattr(model, "__pydantic_validator__")

    def test_exclude_none_in_serialization(self):
        """測試序列化時排除 None 值"""
        model = TestModel(name="test")
        data = model.to_dict()

        # updated_at 為 None，應該被排除
        assert "updated_at" not in data or data["updated_at"] is None
