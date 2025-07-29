#!/usr/bin/env python3
"""簡單測試中文文本處理器"""

from chinese_graphrag.processors.chinese_text_processor import ChineseTextProcessor

def test_basic_functionality():
    """測試基本功能"""
    processor = ChineseTextProcessor()
    
    # 測試分詞
    text = "這是一個測試中文分詞功能的文本。"
    words = processor.segment_text(text)
    print("分詞結果:", words)
    assert len(words) > 0
    assert "測試" in words
    
    # 測試文本清理
    dirty_text = "  這是一個測試文本。  \n\n  包含多餘空格。  "
    cleaned = processor.clean_text(dirty_text)
    print("清理結果:", cleaned)
    assert "這是一個測試文本" in cleaned
    
    # 測試文本分塊
    long_text = "這是一個很長的測試文本。" * 10
    chunks = processor.split_text_into_chunks(long_text, "test_doc")
    print(f"分塊數量: {len(chunks)}")
    assert len(chunks) > 0
    assert all(chunk.document_id == "test_doc" for chunk in chunks)
    
    # 測試統計功能
    stats = processor.get_text_statistics(text)
    print("統計結果:", stats)
    assert stats['char_count'] > 0
    assert stats['chinese_char_count'] > 0
    
    print("所有測試通過！")

if __name__ == "__main__":
    test_basic_functionality()