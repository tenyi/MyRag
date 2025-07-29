"""
文本單元資料模型測試
"""

import numpy as np
import pytest
from pydantic import ValidationError

from chinese_graphrag.models.text_unit import TextUnit


class TestTextUnit:
    """測試文本單元資料模型"""
    
    def test_create_text_unit_with_required_fields(self):
        """測試使用必要欄位建立文本單元"""
        text_unit = TextUnit(
            text="這是測試文本",
            document_id="doc-123",
            chunk_index=0
        )
        
        assert text_unit.text == "這是測試文本"
        assert text_unit.document_id == "doc-123"
        assert text_unit.chunk_index == 0
        assert text_unit.start_position is None
        assert text_unit.end_position is None
        assert text_unit.embedding is None
        assert text_unit.token_count is None
    
    def test_create_text_unit_with_all_fields(self):
        """測試使用所有欄位建立文本單元"""
        embedding = np.array([0.1, 0.2, 0.3])
        
        text_unit = TextUnit(
            text="完整的測試文本",
            document_id="doc-456",
            chunk_index=1,
            start_position=100,
            end_position=200,
            embedding=embedding,
            token_count=50
        )
        
        assert text_unit.text == "完整的測試文本"
        assert text_unit.document_id == "doc-456"
        assert text_unit.chunk_index == 1
        assert text_unit.start_position == 100
        assert text_unit.end_position == 200
        assert np.array_equal(text_unit.embedding, embedding)
        assert text_unit.token_count == 50
    
    def test_text_validation(self):
        """測試文本驗證"""
        # 空文本應該失敗
        with pytest.raises(ValidationError):
            TextUnit(text="", document_id="doc-123", chunk_index=0)
        
        # 只有空白的文本應該失敗
        with pytest.raises(ValidationError):
            TextUnit(text="   ", document_id="doc-123", chunk_index=0)
        
        # 正常文本應該成功，並且會被 strip
        text_unit = TextUnit(text="  正常文本  ", document_id="doc-123", chunk_index=0)
        assert text_unit.text == "正常文本"
    
    def test_document_id_validation(self):
        """測試文件 ID 驗證"""
        # 空 ID 應該失敗
        with pytest.raises(ValidationError):
            TextUnit(text="文本", document_id="", chunk_index=0)
        
        # 只有空白的 ID 應該失敗
        with pytest.raises(ValidationError):
            TextUnit(text="文本", document_id="   ", chunk_index=0)
        
        # 正常 ID 應該成功，並且會被 strip
        text_unit = TextUnit(text="文本", document_id="  doc-123  ", chunk_index=0)
        assert text_unit.document_id == "doc-123"
    
    def test_chunk_index_validation(self):
        """測試塊索引驗證"""
        # 正數應該成功
        text_unit = TextUnit(text="文本", document_id="doc-123", chunk_index=5)
        assert text_unit.chunk_index == 5
        
        # 零應該成功
        text_unit = TextUnit(text="文本", document_id="doc-123", chunk_index=0)
        assert text_unit.chunk_index == 0
        
        # 負數應該失敗
        with pytest.raises(ValidationError):
            TextUnit(text="文本", document_id="doc-123", chunk_index=-1)
    
    def test_position_validation(self):
        """測試位置驗證"""
        # 正常位置應該成功
        text_unit = TextUnit(
            text="文本",
            document_id="doc-123",
            chunk_index=0,
            start_position=10,
            end_position=20
        )
        assert text_unit.start_position == 10
        assert text_unit.end_position == 20
        
        # 結束位置小於等於起始位置應該失敗
        with pytest.raises(ValidationError):
            TextUnit(
                text="文本",
                document_id="doc-123",
                chunk_index=0,
                start_position=20,
                end_position=10
            )
        
        with pytest.raises(ValidationError):
            TextUnit(
                text="文本",
                document_id="doc-123",
                chunk_index=0,
                start_position=20,
                end_position=20
            )
        
        # 負數位置應該失敗
        with pytest.raises(ValidationError):
            TextUnit(
                text="文本",
                document_id="doc-123",
                chunk_index=0,
                start_position=-1
            )
    
    def test_embedding_validation(self):
        """測試 embedding 驗證"""
        # 正確的 numpy array 應該成功
        embedding = np.array([0.1, 0.2, 0.3])
        text_unit = TextUnit(
            text="文本",
            document_id="doc-123",
            chunk_index=0,
            embedding=embedding
        )
        assert np.array_equal(text_unit.embedding, embedding)
        
        # 非 numpy array 應該失敗
        with pytest.raises(ValidationError):
            TextUnit(
                text="文本",
                document_id="doc-123",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3]
            )
        
        # 多維 array 應該失敗
        with pytest.raises(ValidationError):
            TextUnit(
                text="文本",
                document_id="doc-123",
                chunk_index=0,
                embedding=np.array([[0.1, 0.2], [0.3, 0.4]])
            )
        
        # 空 array 應該失敗
        with pytest.raises(ValidationError):
            TextUnit(
                text="文本",
                document_id="doc-123",
                chunk_index=0,
                embedding=np.array([])
            )
    
    def test_token_count_validation(self):
        """測試 token 數量驗證"""
        # 正數應該成功
        text_unit = TextUnit(
            text="文本",
            document_id="doc-123",
            chunk_index=0,
            token_count=100
        )
        assert text_unit.token_count == 100
        
        # 零應該成功
        text_unit = TextUnit(
            text="文本",
            document_id="doc-123",
            chunk_index=0,
            token_count=0
        )
        assert text_unit.token_count == 0
        
        # 負數應該失敗
        with pytest.raises(ValidationError):
            TextUnit(
                text="文本",
                document_id="doc-123",
                chunk_index=0,
                token_count=-1
            )
    
    def test_text_length_property(self):
        """測試文本長度屬性"""
        text_unit = TextUnit(text="測試文本", document_id="doc-123", chunk_index=0)
        assert text_unit.text_length == len("測試文本")
    
    def test_has_embedding_property(self):
        """測試是否有 embedding 屬性"""
        # 沒有 embedding
        text_unit = TextUnit(text="文本", document_id="doc-123", chunk_index=0)
        assert not text_unit.has_embedding
        
        # 有 embedding
        text_unit = TextUnit(
            text="文本",
            document_id="doc-123",
            chunk_index=0,
            embedding=np.array([0.1, 0.2, 0.3])
        )
        assert text_unit.has_embedding
    
    def test_embedding_dimension_property(self):
        """測試 embedding 維度屬性"""
        # 沒有 embedding
        text_unit = TextUnit(text="文本", document_id="doc-123", chunk_index=0)
        assert text_unit.embedding_dimension is None
        
        # 有 embedding
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        text_unit = TextUnit(
            text="文本",
            document_id="doc-123",
            chunk_index=0,
            embedding=embedding
        )
        assert text_unit.embedding_dimension == 5
    
    def test_get_text_preview(self):
        """測試取得文本預覽"""
        # 短文本應該返回完整文本
        short_text = "短文本"
        text_unit = TextUnit(text=short_text, document_id="doc-123", chunk_index=0)
        assert text_unit.get_text_preview() == short_text
        
        # 長文本應該被截斷
        long_text = "這是一個很長的文本內容" * 10
        text_unit = TextUnit(text=long_text, document_id="doc-123", chunk_index=0)
        preview = text_unit.get_text_preview(max_length=20)
        assert len(preview) <= 23  # 20 + "..."
        assert preview.endswith("...")
    
    def test_serialization(self):
        """測試序列化"""
        embedding = np.array([0.1, 0.2, 0.3])
        text_unit = TextUnit(
            text="測試文本",
            document_id="doc-123",
            chunk_index=1,
            start_position=10,
            end_position=20,
            embedding=embedding,
            token_count=50
        )
        
        # 測試轉換為字典
        data = text_unit.to_dict()
        assert data["text"] == "測試文本"
        assert data["document_id"] == "doc-123"
        assert data["chunk_index"] == 1
        assert data["start_position"] == 10
        assert data["end_position"] == 20
        assert data["token_count"] == 50
        
        # 注意：numpy array 在序列化時需要特殊處理
        # 這裡我們主要測試其他欄位的序列化