"""
社群資料模型測試
"""

import numpy as np
import pytest
from pydantic import ValidationError

from chinese_graphrag.models.community import Community


class TestCommunity:
    """測試社群資料模型"""
    
    def test_create_community_with_required_fields(self):
        """測試使用必要欄位建立社群"""
        community = Community(
            title="測試社群",
            level=0,
            summary="這是測試社群的摘要",
            full_content="這是測試社群的完整內容描述"
        )
        
        assert community.title == "測試社群"
        assert community.level == 0
        assert community.summary == "這是測試社群的摘要"
        assert community.full_content == "這是測試社群的完整內容描述"
        assert community.entities == []
        assert community.relationships == []
        assert community.embedding is None
        assert community.rank == 0.0
        assert community.size is None
        assert community.density is None
        assert community.parent_community_id is None
        assert community.child_communities == []
    
    def test_create_community_with_all_fields(self):
        """測試使用所有欄位建立社群"""
        embedding = np.array([0.1, 0.2, 0.3])
        entities = ["entity-1", "entity-2"]
        relationships = ["rel-1", "rel-2"]
        child_communities = ["child-1", "child-2"]
        
        community = Community(
            title="完整社群",
            level=1,
            entities=entities,
            relationships=relationships,
            summary="完整社群摘要",
            full_content="完整社群的詳細內容",
            embedding=embedding,
            rank=0.8,
            size=10,
            density=0.7,
            parent_community_id="parent-1",
            child_communities=child_communities
        )
        
        assert community.title == "完整社群"
        assert community.level == 1
        assert set(community.entities) == set(entities)
        assert set(community.relationships) == set(relationships)
        assert community.summary == "完整社群摘要"
        assert community.full_content == "完整社群的詳細內容"
        assert np.array_equal(community.embedding, embedding)
        assert community.rank == 0.8
        assert community.size == 10
        assert community.density == 0.7
        assert community.parent_community_id == "parent-1"
        assert set(community.child_communities) == set(child_communities)
    
    def test_title_validation(self):
        """測試標題驗證"""
        # 空標題應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="",
                level=0,
                summary="摘要",
                full_content="內容"
            )
        
        # 只有空白的標題應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="   ",
                level=0,
                summary="摘要",
                full_content="內容"
            )
        
        # 太長的標題應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="x" * 201,
                level=0,
                summary="摘要",
                full_content="內容"
            )
        
        # 正常標題應該成功，並且會被 strip
        community = Community(
            title="  正常標題  ",
            level=0,
            summary="摘要",
            full_content="內容"
        )
        assert community.title == "正常標題"
    
    def test_level_validation(self):
        """測試層級驗證"""
        # 正數應該成功
        community = Community(
            title="社群",
            level=5,
            summary="摘要",
            full_content="內容"
        )
        assert community.level == 5
        
        # 零應該成功
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容"
        )
        assert community.level == 0
        
        # 負數應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=-1,
                summary="摘要",
                full_content="內容"
            )
    
    def test_summary_validation(self):
        """測試摘要驗證"""
        # 空摘要應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="",
                full_content="內容"
            )
        
        # 只有空白的摘要應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="   ",
                full_content="內容"
            )
        
        # 正常摘要應該成功，並且會被 strip
        community = Community(
            title="社群",
            level=0,
            summary="  正常摘要  ",
            full_content="內容"
        )
        assert community.summary == "正常摘要"
    
    def test_full_content_validation(self):
        """測試完整內容驗證"""
        # 空內容應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content=""
            )
        
        # 只有空白的內容應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content="   "
            )
        
        # 正常內容應該成功，並且會被 strip
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="  正常內容  "
        )
        assert community.full_content == "正常內容"
    
    def test_entities_validation(self):
        """測試實體列表驗證"""
        # 包含空字串和重複項目的列表應該被清理
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            entities=["entity-1", "", "entity-2", "entity-1", "   ", "entity-3"]
        )
        
        # 應該移除空字串和重複項目
        assert set(community.entities) == {"entity-1", "entity-2", "entity-3"}
    
    def test_relationships_validation(self):
        """測試關係列表驗證"""
        # 包含空字串和重複項目的列表應該被清理
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            relationships=["rel-1", "", "rel-2", "rel-1", "   ", "rel-3"]
        )
        
        # 應該移除空字串和重複項目
        assert set(community.relationships) == {"rel-1", "rel-2", "rel-3"}
    
    def test_child_communities_validation(self):
        """測試子社群列表驗證"""
        # 包含空字串和重複項目的列表應該被清理
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            child_communities=["child-1", "", "child-2", "child-1", "   ", "child-3"]
        )
        
        # 應該移除空字串和重複項目
        assert set(community.child_communities) == {"child-1", "child-2", "child-3"}
    
    def test_embedding_validation(self):
        """測試 embedding 驗證"""
        # 正確的 numpy array 應該成功
        embedding = np.array([0.1, 0.2, 0.3])
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            embedding=embedding
        )
        assert np.array_equal(community.embedding, embedding)
        
        # 非 numpy array 應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content="內容",
                embedding=[0.1, 0.2, 0.3]
            )
        
        # 多維 array 應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content="內容",
                embedding=np.array([[0.1, 0.2], [0.3, 0.4]])
            )
        
        # 空 array 應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content="內容",
                embedding=np.array([])
            )
    
    def test_rank_validation(self):
        """測試排名驗證"""
        # 正常範圍應該成功
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            rank=0.5
        )
        assert community.rank == 0.5
        
        # 邊界值應該成功
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            rank=0.0
        )
        assert community.rank == 0.0
        
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            rank=1.0
        )
        assert community.rank == 1.0
        
        # 超出範圍應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content="內容",
                rank=-0.1
            )
        
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content="內容",
                rank=1.1
            )
    
    def test_size_validation(self):
        """測試大小驗證"""
        # 正數應該成功
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            size=10
        )
        assert community.size == 10
        
        # 零應該成功
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            size=0
        )
        assert community.size == 0
        
        # 負數應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content="內容",
                size=-1
            )
    
    def test_density_validation(self):
        """測試密度驗證"""
        # 正常範圍應該成功
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            density=0.5
        )
        assert community.density == 0.5
        
        # 邊界值應該成功
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            density=0.0
        )
        assert community.density == 0.0
        
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            density=1.0
        )
        assert community.density == 1.0
        
        # 超出範圍應該失敗
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content="內容",
                density=-0.1
            )
        
        with pytest.raises(ValidationError):
            Community(
                title="社群",
                level=0,
                summary="摘要",
                full_content="內容",
                density=1.1
            )
    
    def test_properties(self):
        """測試屬性方法"""
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        community = Community(
            title="社群",
            level=1,
            entities=["entity-1", "entity-2", "entity-3"],
            relationships=["rel-1", "rel-2"],
            summary="摘要",
            full_content="內容",
            embedding=embedding,
            parent_community_id="parent-1",
            child_communities=["child-1", "child-2"]
        )
        
        # 測試各種屬性
        assert community.entity_count == 3
        assert community.relationship_count == 2
        assert community.child_community_count == 2
        assert community.has_embedding
        assert community.embedding_dimension == 5
        assert community.has_parent
        assert community.has_children
        assert not community.is_leaf_community
        assert not community.is_root_community
        
        # 測試葉子社群
        leaf_community = Community(
            title="葉子社群",
            level=2,
            summary="摘要",
            full_content="內容",
            parent_community_id="parent-1"
        )
        assert leaf_community.is_leaf_community
        assert not leaf_community.is_root_community
        
        # 測試根社群
        root_community = Community(
            title="根社群",
            level=0,
            summary="摘要",
            full_content="內容",
            child_communities=["child-1"]
        )
        assert not root_community.is_leaf_community
        assert root_community.is_root_community
    
    def test_add_entity(self):
        """測試新增實體"""
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容"
        )
        
        # 新增實體
        community.add_entity("entity-1")
        assert "entity-1" in community.entities
        
        # 重複新增應該不會增加
        community.add_entity("entity-1")
        assert community.entities.count("entity-1") == 1
        
        # 新增空字串應該被忽略
        original_count = len(community.entities)
        community.add_entity("")
        community.add_entity("   ")
        assert len(community.entities) == original_count
    
    def test_remove_entity(self):
        """測試移除實體"""
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            entities=["entity-1", "entity-2", "entity-3"]
        )
        
        # 移除存在的實體
        result = community.remove_entity("entity-2")
        assert result is True
        assert "entity-2" not in community.entities
        assert len(community.entities) == 2
        
        # 移除不存在的實體
        result = community.remove_entity("entity-4")
        assert result is False
        assert len(community.entities) == 2
    
    def test_add_relationship(self):
        """測試新增關係"""
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容"
        )
        
        # 新增關係
        community.add_relationship("rel-1")
        assert "rel-1" in community.relationships
        
        # 重複新增應該不會增加
        community.add_relationship("rel-1")
        assert community.relationships.count("rel-1") == 1
        
        # 新增空字串應該被忽略
        original_count = len(community.relationships)
        community.add_relationship("")
        community.add_relationship("   ")
        assert len(community.relationships) == original_count
    
    def test_remove_relationship(self):
        """測試移除關係"""
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            relationships=["rel-1", "rel-2", "rel-3"]
        )
        
        # 移除存在的關係
        result = community.remove_relationship("rel-2")
        assert result is True
        assert "rel-2" not in community.relationships
        assert len(community.relationships) == 2
        
        # 移除不存在的關係
        result = community.remove_relationship("rel-4")
        assert result is False
        assert len(community.relationships) == 2
    
    def test_add_child_community(self):
        """測試新增子社群"""
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容"
        )
        
        # 新增子社群
        community.add_child_community("child-1")
        assert "child-1" in community.child_communities
        
        # 重複新增應該不會增加
        community.add_child_community("child-1")
        assert community.child_communities.count("child-1") == 1
        
        # 新增空字串應該被忽略
        original_count = len(community.child_communities)
        community.add_child_community("")
        community.add_child_community("   ")
        assert len(community.child_communities) == original_count
    
    def test_remove_child_community(self):
        """測試移除子社群"""
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            child_communities=["child-1", "child-2", "child-3"]
        )
        
        # 移除存在的子社群
        result = community.remove_child_community("child-2")
        assert result is True
        assert "child-2" not in community.child_communities
        assert len(community.child_communities) == 2
        
        # 移除不存在的子社群
        result = community.remove_child_community("child-4")
        assert result is False
        assert len(community.child_communities) == 2
    
    def test_update_rank(self):
        """測試更新排名"""
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容"
        )
        
        # 正常更新
        community.update_rank(0.8)
        assert community.rank == 0.8
        
        # 邊界值
        community.update_rank(0.0)
        assert community.rank == 0.0
        
        community.update_rank(1.0)
        assert community.rank == 1.0
        
        # 無效值應該拋出異常
        with pytest.raises(ValueError):
            community.update_rank(-0.1)
        
        with pytest.raises(ValueError):
            community.update_rank(1.1)
    
    def test_calculate_size(self):
        """測試計算社群大小"""
        community = Community(
            title="社群",
            level=0,
            summary="摘要",
            full_content="內容",
            entities=["entity-1", "entity-2", "entity-3"],
            relationships=["rel-1", "rel-2"]
        )
        
        # 計算大小
        size = community.calculate_size()
        assert size == 5  # 3 實體 + 2 關係
        assert community.size == 5
    
    def test_get_summary_preview(self):
        """測試取得摘要預覽"""
        # 短摘要應該返回完整摘要
        short_summary = "短摘要"
        community = Community(
            title="社群",
            level=0,
            summary=short_summary,
            full_content="內容"
        )
        assert community.get_summary_preview() == short_summary
        
        # 長摘要應該被截斷
        long_summary = "這是一個很長的摘要內容" * 20
        community = Community(
            title="社群",
            level=0,
            summary=long_summary,
            full_content="內容"
        )
        preview = community.get_summary_preview(max_length=50)
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")
    
    def test_serialization(self):
        """測試序列化"""
        community = Community(
            title="測試社群",
            level=1,
            entities=["entity-1", "entity-2"],
            relationships=["rel-1"],
            summary="測試摘要",
            full_content="測試內容",
            rank=0.8,
            size=10,
            density=0.7,
            parent_community_id="parent-1",
            child_communities=["child-1"]
        )
        
        # 測試轉換為字典
        data = community.to_dict()
        assert data["title"] == "測試社群"
        assert data["level"] == 1
        assert set(data["entities"]) == {"entity-1", "entity-2"}
        assert data["relationships"] == ["rel-1"]
        assert data["summary"] == "測試摘要"
        assert data["full_content"] == "測試內容"
        assert data["rank"] == 0.8
        assert data["size"] == 10
        assert data["density"] == 0.7
        assert data["parent_community_id"] == "parent-1"
        assert data["child_communities"] == ["child-1"]
        
        # 測試從字典重建
        new_community = Community.from_dict(data)
        assert new_community.title == community.title
        assert new_community.level == community.level
        assert new_community.entities == community.entities
        assert new_community.relationships == community.relationships
        assert new_community.summary == community.summary
        assert new_community.full_content == community.full_content
        assert new_community.rank == community.rank
        assert new_community.size == community.size
        assert new_community.density == community.density
        assert new_community.parent_community_id == community.parent_community_id
        assert new_community.child_communities == community.child_communities