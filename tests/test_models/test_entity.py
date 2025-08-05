"""
實體資料模型測試
"""

import numpy as np
import pytest
from pydantic import ValidationError

from chinese_graphrag.models.entity import Entity


class TestEntity:
    """測試實體資料模型"""

    def test_create_entity_with_required_fields(self):
        """測試使用必要欄位建立實體"""
        entity = Entity(name="測試實體", type="PERSON", description="這是一個測試實體")

        assert entity.name == "測試實體"
        assert entity.type == "PERSON"
        assert entity.description == "這是一個測試實體"
        assert entity.text_units == []
        assert entity.embedding is None
        assert entity.community_id is None
        assert entity.rank == 0.0
        assert entity.frequency == 1
        assert entity.confidence == 1.0

    def test_create_entity_with_all_fields(self):
        """測試使用所有欄位建立實體"""
        embedding = np.array([0.1, 0.2, 0.3])
        text_units = ["unit-1", "unit-2"]

        entity = Entity(
            name="完整實體",
            type="organization",
            description="完整的實體描述",
            text_units=text_units,
            embedding=embedding,
            community_id="community-1",
            rank=0.8,
            frequency=5,
            confidence=0.9,
        )

        assert entity.name == "完整實體"
        assert entity.type == "ORGANIZATION"  # 應該轉為大寫
        assert entity.description == "完整的實體描述"
        assert set(entity.text_units) == set(text_units)
        assert np.array_equal(entity.embedding, embedding)
        assert entity.community_id == "community-1"
        assert entity.rank == 0.8
        assert entity.frequency == 5
        assert entity.confidence == 0.9

    def test_name_validation(self):
        """測試實體名稱驗證"""
        # 空名稱應該失敗
        with pytest.raises(ValidationError):
            Entity(name="", type="PERSON", description="描述")

        # 只有空白的名稱應該失敗
        with pytest.raises(ValidationError):
            Entity(name="   ", type="PERSON", description="描述")

        # 太長的名稱應該失敗
        with pytest.raises(ValidationError):
            Entity(name="x" * 201, type="PERSON", description="描述")

        # 正常名稱應該成功，並且會被 strip
        entity = Entity(name="  正常實體  ", type="PERSON", description="描述")
        assert entity.name == "正常實體"

    def test_type_validation(self):
        """測試實體類型驗證"""
        # 空類型應該失敗
        with pytest.raises(ValidationError):
            Entity(name="實體", type="", description="描述")

        # 只有空白的類型應該失敗
        with pytest.raises(ValidationError):
            Entity(name="實體", type="   ", description="描述")

        # 太長的類型應該失敗
        with pytest.raises(ValidationError):
            Entity(name="實體", type="x" * 51, description="描述")

        # 正常類型應該成功，並且會被 strip 和轉為大寫
        entity = Entity(name="實體", type="  person  ", description="描述")
        assert entity.type == "PERSON"

    def test_description_validation(self):
        """測試實體描述驗證"""
        # 空描述應該失敗
        with pytest.raises(ValidationError):
            Entity(name="實體", type="PERSON", description="")

        # 只有空白的描述應該失敗
        with pytest.raises(ValidationError):
            Entity(name="實體", type="PERSON", description="   ")

        # 正常描述應該成功，並且會被 strip
        entity = Entity(name="實體", type="PERSON", description="  正常描述  ")
        assert entity.description == "正常描述"

    def test_text_units_validation(self):
        """測試文本單元列表驗證"""
        # 包含空字串和重複項目的列表應該被清理
        entity = Entity(
            name="實體",
            type="PERSON",
            description="描述",
            text_units=["unit-1", "", "unit-2", "unit-1", "   ", "unit-3"],
        )

        # 應該移除空字串和重複項目
        assert set(entity.text_units) == {"unit-1", "unit-2", "unit-3"}

    def test_rank_validation(self):
        """測試排名驗證"""
        # 正常範圍應該成功
        entity = Entity(name="實體", type="PERSON", description="描述", rank=0.5)
        assert entity.rank == 0.5

        # 邊界值應該成功
        entity = Entity(name="實體", type="PERSON", description="描述", rank=0.0)
        assert entity.rank == 0.0

        entity = Entity(name="實體", type="PERSON", description="描述", rank=1.0)
        assert entity.rank == 1.0

        # 超出範圍應該失敗
        with pytest.raises(ValidationError):
            Entity(name="實體", type="PERSON", description="描述", rank=-0.1)

        with pytest.raises(ValidationError):
            Entity(name="實體", type="PERSON", description="描述", rank=1.1)

    def test_frequency_validation(self):
        """測試頻率驗證"""
        # 正數應該成功
        entity = Entity(name="實體", type="PERSON", description="描述", frequency=5)
        assert entity.frequency == 5

        # 1 應該成功（最小值）
        entity = Entity(name="實體", type="PERSON", description="描述", frequency=1)
        assert entity.frequency == 1

        # 小於 1 應該失敗
        with pytest.raises(ValidationError):
            Entity(name="實體", type="PERSON", description="描述", frequency=0)

    def test_confidence_validation(self):
        """測試置信度驗證"""
        # 正常範圍應該成功
        entity = Entity(name="實體", type="PERSON", description="描述", confidence=0.8)
        assert entity.confidence == 0.8

        # 邊界值應該成功
        entity = Entity(name="實體", type="PERSON", description="描述", confidence=0.0)
        assert entity.confidence == 0.0

        entity = Entity(name="實體", type="PERSON", description="描述", confidence=1.0)
        assert entity.confidence == 1.0

        # 超出範圍應該失敗
        with pytest.raises(ValidationError):
            Entity(name="實體", type="PERSON", description="描述", confidence=-0.1)

        with pytest.raises(ValidationError):
            Entity(name="實體", type="PERSON", description="描述", confidence=1.1)

    def test_embedding_validation(self):
        """測試 embedding 驗證"""
        # 正確的 numpy array 應該成功
        embedding = np.array([0.1, 0.2, 0.3])
        entity = Entity(
            name="實體", type="PERSON", description="描述", embedding=embedding
        )
        assert np.array_equal(entity.embedding, embedding)

        # 非 numpy array 應該失敗
        with pytest.raises(ValidationError):
            Entity(
                name="實體",
                type="PERSON",
                description="描述",
                embedding=[0.1, 0.2, 0.3],
            )

        # 多維 array 應該失敗
        with pytest.raises(ValidationError):
            Entity(
                name="實體",
                type="PERSON",
                description="描述",
                embedding=np.array([[0.1, 0.2], [0.3, 0.4]]),
            )

        # 空 array 應該失敗
        with pytest.raises(ValidationError):
            Entity(
                name="實體", type="PERSON", description="描述", embedding=np.array([])
            )

    def test_properties(self):
        """測試屬性方法"""
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        entity = Entity(
            name="實體",
            type="PERSON",
            description="描述",
            text_units=["unit-1", "unit-2", "unit-3"],
            embedding=embedding,
            community_id="community-1",
        )

        # 測試各種屬性
        assert entity.has_embedding
        assert entity.embedding_dimension == 5
        assert entity.text_unit_count == 3
        assert entity.has_community

        # 測試沒有 embedding 和 community 的情況
        entity_simple = Entity(name="簡單實體", type="PERSON", description="描述")
        assert not entity_simple.has_embedding
        assert entity_simple.embedding_dimension is None
        assert entity_simple.text_unit_count == 0
        assert not entity_simple.has_community

    def test_add_text_unit(self):
        """測試新增文本單元"""
        entity = Entity(name="實體", type="PERSON", description="描述")

        # 新增文本單元
        entity.add_text_unit("unit-1")
        assert "unit-1" in entity.text_units

        # 重複新增應該不會增加
        entity.add_text_unit("unit-1")
        assert entity.text_units.count("unit-1") == 1

        # 新增空字串應該被忽略
        original_count = len(entity.text_units)
        entity.add_text_unit("")
        entity.add_text_unit("   ")
        assert len(entity.text_units) == original_count

    def test_remove_text_unit(self):
        """測試移除文本單元"""
        entity = Entity(
            name="實體",
            type="PERSON",
            description="描述",
            text_units=["unit-1", "unit-2", "unit-3"],
        )

        # 移除存在的文本單元
        result = entity.remove_text_unit("unit-2")
        assert result is True
        assert "unit-2" not in entity.text_units
        assert len(entity.text_units) == 2

        # 移除不存在的文本單元
        result = entity.remove_text_unit("unit-4")
        assert result is False
        assert len(entity.text_units) == 2

    def test_update_rank(self):
        """測試更新排名"""
        entity = Entity(name="實體", type="PERSON", description="描述")

        # 正常更新
        entity.update_rank(0.8)
        assert entity.rank == 0.8

        # 邊界值
        entity.update_rank(0.0)
        assert entity.rank == 0.0

        entity.update_rank(1.0)
        assert entity.rank == 1.0

        # 無效值應該拋出異常
        with pytest.raises(ValueError):
            entity.update_rank(-0.1)

        with pytest.raises(ValueError):
            entity.update_rank(1.1)

    def test_serialization(self):
        """測試序列化"""
        entity = Entity(
            name="測試實體",
            type="PERSON",
            description="測試描述",
            text_units=["unit-1", "unit-2"],
            community_id="community-1",
            rank=0.8,
            frequency=3,
            confidence=0.9,
        )

        # 測試轉換為字典
        data = entity.to_dict()
        assert data["name"] == "測試實體"
        assert data["type"] == "PERSON"
        assert data["description"] == "測試描述"
        assert set(data["text_units"]) == {"unit-1", "unit-2"}
        assert data["community_id"] == "community-1"
        assert data["rank"] == 0.8
        assert data["frequency"] == 3
        assert data["confidence"] == 0.9

        # 測試從字典重建
        new_entity = Entity.from_dict(data)
        assert new_entity.name == entity.name
        assert new_entity.type == entity.type
        assert new_entity.description == entity.description
        assert set(new_entity.text_units) == set(entity.text_units)
        assert new_entity.community_id == entity.community_id
        assert new_entity.rank == entity.rank
        assert new_entity.frequency == entity.frequency
        assert new_entity.confidence == entity.confidence
