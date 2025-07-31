#!/usr/bin/env python3
"""調試測試"""

import sys
import time

print("開始調試測試...")

try:
    print("1. 載入配置模組...")
    from chinese_graphrag.config import GraphRAGConfig, VectorStoreConfig, VectorStoreType
    print("   配置模組載入成功")
    
    print("2. 建立配置...")
    config = GraphRAGConfig(
        models={}, 
        vector_store=VectorStoreConfig(
            type=VectorStoreType.LANCEDB, 
            uri='./test_data/lancedb'
        )
    )
    print("   配置建立成功")
    
    print("3. 載入模型選擇器...")
    from chinese_graphrag.config.strategy import ModelSelector
    model_selector = ModelSelector(config)
    print("   模型選擇器載入成功")
    
    print("4. 跳過文件處理器...")
    
    print("5. 載入 Embedding 管理器...")
    from chinese_graphrag.embeddings import EmbeddingManager
    embedding_manager = EmbeddingManager(config)
    print("   Embedding 管理器載入成功")
    
    print("6. 載入向量儲存管理器...")
    from chinese_graphrag.vector_stores import VectorStoreManager
    vector_store_manager = VectorStoreManager(config)
    print("   向量儲存管理器載入成功")
    
    print("7. 載入社群檢測器...")
    from chinese_graphrag.indexing.community_detector import CommunityDetector
    community_detector = CommunityDetector()
    print("   社群檢測器載入成功")
    
    print("8. 載入社群報告生成器...")
    from chinese_graphrag.indexing.community_report_generator import CommunityReportGenerator
    report_generator = CommunityReportGenerator(config)
    print("   社群報告生成器載入成功")
    
    print("所有元件載入成功！")
    
except Exception as e:
    print(f"錯誤: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)