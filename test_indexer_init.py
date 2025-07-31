#!/usr/bin/env python3
"""測試索引引擎初始化"""

from chinese_graphrag.indexing.engine import GraphRAGIndexer
from chinese_graphrag.config import GraphRAGConfig, VectorStoreConfig, VectorStoreType

def test_indexer_init():
    """測試索引引擎初始化"""
    config = GraphRAGConfig(
        models={}, 
        vector_store=VectorStoreConfig(
            type=VectorStoreType.LANCEDB, 
            uri='./test_data/lancedb'
        )
    )
    
    indexer = GraphRAGIndexer(config)
    print('索引引擎初始化成功')
    
    # 檢查各個元件
    assert indexer.config == config
    assert indexer.model_selector is not None
    assert indexer.document_processor is not None
    assert indexer.embedding_manager is not None
    assert indexer.vector_store_manager is not None
    assert indexer.community_detector is not None
    assert indexer.report_generator is not None
    
    print('所有元件初始化成功')
    
    # 檢查初始狀態
    stats = indexer.get_statistics()
    print(f'初始統計: {stats}')
    
    assert stats['documents'] == 0
    assert stats['text_units'] == 0
    assert stats['entities'] == 0
    assert stats['relationships'] == 0
    assert stats['communities'] == 0
    
    print('初始狀態檢查通過')

if __name__ == '__main__':
    test_indexer_init()