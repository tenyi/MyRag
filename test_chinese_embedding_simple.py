#!/usr/bin/env python3
"""
簡單的中文優化 Embedding 測試
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_import():
    """測試導入"""
    try:
        from chinese_graphrag.embeddings.chinese_optimized import (
            ChineseEmbeddingConfig,
            ChineseOptimizedEmbeddingService,
            create_chinese_optimized_service
        )
        print("✓ 成功導入中文優化 embedding 模組")
        return True
    except Exception as e:
        print(f"✗ 導入失敗: {e}")
        return False

def test_config():
    """測試配置"""
    try:
        from chinese_graphrag.embeddings.chinese_optimized import ChineseEmbeddingConfig
        
        # 測試預設配置
        config = ChineseEmbeddingConfig()
        print(f"✓ 預設配置創建成功")
        print(f"  - 主要模型: {config.primary_model}")
        print(f"  - 備用模型: {config.fallback_models}")
        print(f"  - 啟用預處理: {config.enable_preprocessing}")
        print(f"  - 中文權重: {config.apply_chinese_weighting}")
        
        # 測試自訂配置
        custom_config = ChineseEmbeddingConfig(
            primary_model="test-model",
            enable_preprocessing=False,
            min_chinese_ratio=0.5
        )
        print(f"✓ 自訂配置創建成功")
        print(f"  - 主要模型: {custom_config.primary_model}")
        print(f"  - 最小中文比例: {custom_config.min_chinese_ratio}")
        
        return True
    except Exception as e:
        print(f"✗ 配置測試失敗: {e}")
        return False

def test_service_creation():
    """測試服務創建"""
    try:
        from chinese_graphrag.embeddings.chinese_optimized import create_chinese_optimized_service
        
        # 注意：這裡不實際載入模型，只測試創建
        service = create_chinese_optimized_service(
            primary_model="BAAI/bge-m3",
            device="cpu"
        )
        
        print(f"✓ 服務創建成功")
        print(f"  - 模型名稱: {service.model_name}")
        print(f"  - 裝置: {service.device}")
        print(f"  - 是否已載入: {service.is_loaded}")
        
        # 測試模型資訊
        info = service.get_model_info()
        print(f"  - 中文優化: {info['chinese_optimized']}")
        print(f"  - 預處理啟用: {info['preprocessing_enabled']}")
        
        return True
    except Exception as e:
        print(f"✗ 服務創建失敗: {e}")
        return False

def test_text_processing():
    """測試文本處理功能"""
    try:
        from chinese_graphrag.embeddings.chinese_optimized import ChineseOptimizedEmbeddingService, ChineseEmbeddingConfig
        
        config = ChineseEmbeddingConfig(enable_preprocessing=True)
        service = ChineseOptimizedEmbeddingService(config=config, device="cpu")
        
        # 測試文本預處理
        test_texts = [
            "這是一個中文測試文本。",
            "This is English text.",
            "",
            "   空白文本   ",
            "混合文本 mixed text 中英文"
        ]
        
        processed = service._preprocess_texts(test_texts)
        print(f"✓ 文本預處理成功")
        print(f"  - 原始文本數: {len(test_texts)}")
        print(f"  - 處理後文本數: {len(processed)}")
        
        # 測試中文權重計算
        chinese_text = "這是一個中文測試文本"
        weight = service._calculate_chinese_weight(chinese_text)
        print(f"  - 中文文本權重: {weight:.3f}")
        
        english_text = "This is an English text"
        weight = service._calculate_chinese_weight(english_text)
        print(f"  - 英文文本權重: {weight:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ 文本處理測試失敗: {e}")
        return False

def main():
    """主函數"""
    print("=== 中文優化 Embedding 服務測試 ===\n")
    
    tests = [
        ("導入測試", test_import),
        ("配置測試", test_config),
        ("服務創建測試", test_service_creation),
        ("文本處理測試", test_text_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} 失敗")
        except Exception as e:
            print(f"✗ {test_name} 異常: {e}")
    
    print(f"\n=== 測試結果 ===")
    print(f"通過: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有測試通過！")
        return 0
    else:
        print("❌ 部分測試失敗")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)