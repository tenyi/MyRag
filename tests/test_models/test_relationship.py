"""
關係資料模型測試
"""

import pytest
from pydantic import ValidationError

from chinese_graphrag.models.relationship import Relationship


class TestRelationship:
    """測試關係資料模型"""

    def test_create_relationship_with_required_fields(self):
        """測試使用必要欄位建立關係"""
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="實體1認識實體2",
        )

        assert relationship.source_entity_id == "entity-1"
        assert relationship.target_entity_id == "entity-2"
        assert relationship.relationship_type == "KNOWS"
        assert relationship.description == "實體1認識實體2"
        assert relationship.weight == 1.0
        assert relationship.text_units == []
        assert relationship.confidence == 1.0
        assert relationship.frequency == 1
        assert relationship.bidirectional is False

    def test_create_relationship_with_all_fields(self):
        """測試使用所有欄位建立關係"""
        text_units = ["unit-1", "unit-2"]

        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="works_for",
            description="實體1為實體2工作",
            weight=0.8,
            text_units=text_units,
            confidence=0.9,
            frequency=3,
            bidirectional=True,
        )

        assert relationship.source_entity_id == "entity-1"
        assert relationship.target_entity_id == "entity-2"
        assert relationship.relationship_type == "WORKS_FOR"  # 應該轉為大寫
        assert relationship.description == "實體1為實體2工作"
        assert relationship.weight == 0.8
        assert set(relationship.text_units) == set(text_units)
        assert relationship.confidence == 0.9
        assert relationship.frequency == 3
        assert relationship.bidirectional is True

    def test_source_entity_id_validation(self):
        """測試來源實體 ID 驗證"""
        # 空 ID 應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="",
                target_entity_id="entity-2",
                relationship_type="KNOWS",
                description="描述",
            )

        # 只有空白的 ID 應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="   ",
                target_entity_id="entity-2",
                relationship_type="KNOWS",
                description="描述",
            )

        # 正常 ID 應該成功，並且會被 strip
        relationship = Relationship(
            source_entity_id="  entity-1  ",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
        )
        assert relationship.source_entity_id == "entity-1"

    def test_target_entity_id_validation(self):
        """測試目標實體 ID 驗證"""
        # 空 ID 應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="",
                relationship_type="KNOWS",
                description="描述",
            )

        # 只有空白的 ID 應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="   ",
                relationship_type="KNOWS",
                description="描述",
            )

        # 與來源實體相同應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-1",
                relationship_type="KNOWS",
                description="描述",
            )

        # 正常 ID 應該成功，並且會被 strip
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="  entity-2  ",
            relationship_type="KNOWS",
            description="描述",
        )
        assert relationship.target_entity_id == "entity-2"

    def test_relationship_type_validation(self):
        """測試關係類型驗證"""
        # 空類型應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="",
                description="描述",
            )

        # 只有空白的類型應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="   ",
                description="描述",
            )

        # 太長的類型應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="x" * 101,
                description="描述",
            )

        # 正常類型應該成功，並且會被 strip 和轉為大寫
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="  knows  ",
            description="描述",
        )
        assert relationship.relationship_type == "KNOWS"

    def test_description_validation(self):
        """測試關係描述驗證"""
        # 空描述應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="KNOWS",
                description="",
            )

        # 只有空白的描述應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="KNOWS",
                description="   ",
            )

        # 正常描述應該成功，並且會被 strip
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="  正常描述  ",
        )
        assert relationship.description == "正常描述"

    def test_weight_validation(self):
        """測試權重驗證"""
        # 正常範圍應該成功
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            weight=0.5,
        )
        assert relationship.weight == 0.5

        # 邊界值應該成功
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            weight=0.0,
        )
        assert relationship.weight == 0.0

        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            weight=1.0,
        )
        assert relationship.weight == 1.0

        # 超出範圍應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="KNOWS",
                description="描述",
                weight=-0.1,
            )

        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="KNOWS",
                description="描述",
                weight=1.1,
            )

    def test_text_units_validation(self):
        """測試文本單元列表驗證"""
        # 包含空字串和重複項目的列表應該被清理
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            text_units=["unit-1", "", "unit-2", "unit-1", "   ", "unit-3"],
        )

        # 應該移除空字串和重複項目
        assert set(relationship.text_units) == {"unit-1", "unit-2", "unit-3"}

    def test_confidence_validation(self):
        """測試置信度驗證"""
        # 正常範圍應該成功
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            confidence=0.8,
        )
        assert relationship.confidence == 0.8

        # 邊界值應該成功
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            confidence=0.0,
        )
        assert relationship.confidence == 0.0

        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            confidence=1.0,
        )
        assert relationship.confidence == 1.0

        # 超出範圍應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="KNOWS",
                description="描述",
                confidence=-0.1,
            )

        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="KNOWS",
                description="描述",
                confidence=1.1,
            )

    def test_frequency_validation(self):
        """測試頻率驗證"""
        # 正數應該成功
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            frequency=5,
        )
        assert relationship.frequency == 5

        # 1 應該成功（最小值）
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            frequency=1,
        )
        assert relationship.frequency == 1

        # 小於 1 應該失敗
        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="KNOWS",
                description="描述",
                frequency=0,
            )

    def test_properties(self):
        """測試屬性方法"""
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            text_units=["unit-1", "unit-2", "unit-3"],
        )

        # 測試各種屬性
        assert relationship.text_unit_count == 3
        assert relationship.entity_pair == ("entity-1", "entity-2")
        assert not relationship.is_self_relationship

        # 測試自我關係檢查（由於驗證會阻止建立自我關係，我們直接測試屬性邏輯）
        # 正常情況下不是自我關係
        assert not relationship.is_self_relationship

    def test_add_text_unit(self):
        """測試新增文本單元"""
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
        )

        # 新增文本單元
        relationship.add_text_unit("unit-1")
        assert "unit-1" in relationship.text_units

        # 重複新增應該不會增加
        relationship.add_text_unit("unit-1")
        assert relationship.text_units.count("unit-1") == 1

        # 新增空字串應該被忽略
        original_count = len(relationship.text_units)
        relationship.add_text_unit("")
        relationship.add_text_unit("   ")
        assert len(relationship.text_units) == original_count

    def test_remove_text_unit(self):
        """測試移除文本單元"""
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
            text_units=["unit-1", "unit-2", "unit-3"],
        )

        # 移除存在的文本單元
        result = relationship.remove_text_unit("unit-2")
        assert result is True
        assert "unit-2" not in relationship.text_units
        assert len(relationship.text_units) == 2

        # 移除不存在的文本單元
        result = relationship.remove_text_unit("unit-4")
        assert result is False
        assert len(relationship.text_units) == 2

    def test_update_weight(self):
        """測試更新權重"""
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="描述",
        )

        # 正常更新
        relationship.update_weight(0.8)
        assert relationship.weight == 0.8

        # 邊界值
        relationship.update_weight(0.0)
        assert relationship.weight == 0.0

        relationship.update_weight(1.0)
        assert relationship.weight == 1.0

        # 無效值應該拋出異常
        with pytest.raises(ValueError):
            relationship.update_weight(-0.1)

        with pytest.raises(ValueError):
            relationship.update_weight(1.1)

    def test_get_reverse_relationship(self):
        """測試取得反向關係"""
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="實體1認識實體2",
            weight=0.8,
            text_units=["unit-1", "unit-2"],
            confidence=0.9,
            frequency=3,
            bidirectional=True,
        )

        # 取得反向關係
        reverse = relationship.get_reverse_relationship()

        assert reverse.source_entity_id == "entity-2"
        assert reverse.target_entity_id == "entity-1"
        assert reverse.relationship_type == "KNOWS"
        assert reverse.description == "反向關係: 實體1認識實體2"
        assert reverse.weight == 0.8
        assert set(reverse.text_units) == {"unit-1", "unit-2"}
        assert reverse.confidence == 0.9
        assert reverse.frequency == 3
        assert reverse.bidirectional is True

        # 非雙向關係應該拋出異常
        relationship.bidirectional = False
        with pytest.raises(ValueError):
            relationship.get_reverse_relationship()

    def test_serialization(self):
        """測試序列化"""
        relationship = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="KNOWS",
            description="測試關係",
            weight=0.8,
            text_units=["unit-1", "unit-2"],
            confidence=0.9,
            frequency=3,
            bidirectional=True,
        )

        # 測試轉換為字典
        data = relationship.to_dict()
        assert data["source_entity_id"] == "entity-1"
        assert data["target_entity_id"] == "entity-2"
        assert data["relationship_type"] == "KNOWS"
        assert data["description"] == "測試關係"
        assert data["weight"] == 0.8
        assert set(data["text_units"]) == {"unit-1", "unit-2"}
        assert data["confidence"] == 0.9
        assert data["frequency"] == 3
        assert data["bidirectional"] is True

        # 測試從字典重建
        new_relationship = Relationship.from_dict(data)
        assert new_relationship.source_entity_id == relationship.source_entity_id
        assert new_relationship.target_entity_id == relationship.target_entity_id
        assert new_relationship.relationship_type == relationship.relationship_type
        assert new_relationship.description == relationship.description
        assert new_relationship.weight == relationship.weight
        assert set(new_relationship.text_units) == set(relationship.text_units)
        assert new_relationship.confidence == relationship.confidence
        assert new_relationship.frequency == relationship.frequency
        assert new_relationship.bidirectional == relationship.bidirectional
