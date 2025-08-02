#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from chinese_graphrag.embeddings.chinese_optimized import ChineseOptimizedEmbeddingService, ChineseEmbeddingConfig
from unittest.mock import patch, AsyncMock
import numpy as np

def debug_test():
    """調試測試"""
    try:
        # 創建配置
        config = ChineseEmbeddingConfig(
            primary_model="test-model",
            fallback_models=["fallback-model"],
            enable_preprocessing=True,
            apply_chinese_weighting=True
        )
        
        print(f"Config created: {config}")
        print(f"enable_preprocessing: {config.enable_preprocessing}")
        
        # 創建服務
        with patch('chinese_graphrag.embeddings.chinese_optimized.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            service = ChineseOptimizedEmbeddingService(config=config, device="cpu")
            
        print(f"Service created: {service}")
        print(f"text_processor: {service.text_processor}")
        print(f"text_processor type: {type(service.text_processor)}")
        
        if service.text_processor:
            # 測試 evaluate_text_quality 方法
            test_text = "這是一個測試文本"
            quality = service.text_processor.evaluate_text_quality(test_text)
            print(f"Quality result: {quality}")
            print(f"Overall score: {quality.get('overall_score', 'NOT FOUND')}")
        else:
            print("text_processor is None!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_test()