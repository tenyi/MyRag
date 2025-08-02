#!/usr/bin/env python3
"""驗證需求 5.3 和 5.4 的實作"""

from chinese_graphrag.processors.chinese_text_processor import ChineseTextProcessor

def verify_requirement_5_3():
    """驗證需求 5.3: 中文分詞、停用詞過濾和文本清理"""
    print("=== 驗證需求 5.3 ===")
    processor = ChineseTextProcessor()
    
    # 測試中文分詞
    text = "我愛北京天安門，天安門上太陽升。"
    words = processor.segment_text(text)
    print(f"分詞結果: {words}")
    print(f"停用詞已過濾: {'我' not in words and '上' not in words}")
    
    # 測試文本清理
    dirty_text = "測試@#$%文本！！！包含特殊字符★☆"
    cleaned = processor.clean_text(dirty_text)
    print(f"清理結果: {cleaned}")
    print(f"特殊字符已移除: {'@#$%' not in cleaned and '★☆' not in cleaned}")
    
    # 測試停用詞過濾
    stopwords_test = "這是一個很好的測試"
    words_with_stopwords = processor.segment_text(stopwords_test, remove_stopwords=False)
    words_without_stopwords = processor.segment_text(stopwords_test, remove_stopwords=True)
    print(f"包含停用詞: {words_with_stopwords}")
    print(f"移除停用詞: {words_without_stopwords}")
    print(f"停用詞過濾有效: {len(words_without_stopwords) < len(words_with_stopwords)}")
    
    print("需求 5.3 驗證完成！\n")

def verify_requirement_5_4():
    """驗證需求 5.4: 中文文本分塊策略"""
    print("=== 驗證需求 5.4 ===")
    processor = ChineseTextProcessor()
    
    # 測試中文文本分塊
    long_text = """
    中華人民共和國是世界上人口最多的國家，位於亞洲東部。
    中國有著悠久的歷史和燦爛的文化，是四大文明古國之一。
    
    改革開放以來，中國經濟快速發展，成為世界第二大經濟體。
    科技創新、人工智慧、新能源等領域都取得了重大突破。
    
    中國堅持和平發展道路，致力於構建人類命運共同體。
    """
    
    chunks = processor.split_text_into_chunks(long_text, "test_doc", chunk_size=100)
    print(f"分塊數量: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"塊 {i}: 長度={len(chunk.text)}, 語言={chunk.metadata.get('language')}")
        print(f"  內容預覽: {chunk.text[:50]}...")
    
    # 測試重疊功能
    chunks_with_overlap = processor.split_text_into_chunks(
        long_text, "test_doc", chunk_size=80, chunk_overlap=20
    )
    print(f"\n帶重疊的分塊數量: {len(chunks_with_overlap)}")
    
    # 驗證中文語言特性
    print(f"所有塊都標記為中文: {all(chunk.metadata.get('language') == 'zh' for chunk in chunks)}")
    print(f"所有塊都有句子統計: {all('sentence_count' in chunk.metadata for chunk in chunks)}")
    
    print("需求 5.4 驗證完成！\n")

def main():
    """主函數"""
    print("開始驗證中文文本預處理器需求...")
    verify_requirement_5_3()
    verify_requirement_5_4()
    print("所有需求驗證完成！")

if __name__ == "__main__":
    main()