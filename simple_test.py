#!/usr/bin/env python3
"""簡單測試"""

print("開始測試...")

try:
    from chinese_graphrag.config import GraphRAGConfig, VectorStoreConfig, VectorStoreType
    print("配置模組載入成功")
    
    config = GraphRAGConfig(
        models={}, 
        vector_store=VectorStoreConfig(
            type=VectorStoreType.LANCEDB, 
            uri='./test_data/lancedb'
        )
    )
    print("配置建立成功")
    
    from chinese_graphrag.indexing.engine import GraphRAGIndexer
    print("索引引擎模組載入成功")
    
    print("準備初始化索引引擎...")
    indexer = GraphRAGIndexer(config)
    print("索引引擎初始化成功！")
    
except Exception as e:
    print(f"錯誤: {e}")
    import traceback
    traceback.print_exc()