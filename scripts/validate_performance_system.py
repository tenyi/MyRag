#!/usr/bin/env python3
"""
系統效能驗證腳本

快速驗證系統的基本功能和效能
"""

import sys
import time
import traceback
from pathlib import Path

def validate_imports():
    """驗證核心模組導入"""
    print("🔍 驗證核心模組導入...")
    
    try:
        # 測試核心模組導入
        from src.chinese_graphrag.processors.chinese_text_processor import ChineseTextProcessor
        print("✅ ChineseTextProcessor 導入成功")
        
        from src.chinese_graphrag.config.loader import ConfigLoader
        print("✅ ConfigLoader 導入成功")
        
        from src.chinese_graphrag.models.base import BaseModel
        print("✅ BaseModel 導入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模組導入失敗: {e}")
        return False

def validate_chinese_processing():
    """驗證中文處理功能"""
    print("\n🇨🇳 驗證中文處理功能...")
    
    try:
        from src.chinese_graphrag.processors.chinese_text_processor import ChineseTextProcessor
        
        processor = ChineseTextProcessor()
        
        # 測試中文文本處理
        test_text = "人工智慧技術正在快速發展，機器學習已經廣泛應用於各個領域。"
        
        start_time = time.time()
        processed_text = processor.preprocess_text(test_text)
        processing_time = time.time() - start_time
        
        print(f"✅ 中文文本預處理成功 (耗時: {processing_time:.3f}s)")
        print(f"   原文: {test_text}")
        print(f"   處理後: {processed_text}")
        
        # 測試文本分塊
        chunks = processor.split_text(processed_text, chunk_size=50)
        print(f"✅ 文本分塊成功 (分塊數: {len(chunks)})")
        
        return True
        
    except Exception as e:
        print(f"❌ 中文處理功能驗證失敗: {e}")
        traceback.print_exc()
        return False

def validate_config_system():
    """驗證配置系統"""
    print("\n⚙️ 驗證配置系統...")
    
    try:
        from src.chinese_graphrag.config.loader import ConfigLoader
        
        config_loader = ConfigLoader()
        
        # 檢查配置檔案
        config_files = [
            Path("config/settings.yaml"),
            Path("config/dev.yaml"),
            Path("pyproject.toml")
        ]
        
        existing_files = [f for f in config_files if f.exists()]
        print(f"✅ 找到配置檔案: {len(existing_files)}/{len(config_files)}")
        
        for file in existing_files:
            print(f"   - {file}")
        
        return len(existing_files) > 0
        
    except Exception as e:
        print(f"❌ 配置系統驗證失敗: {e}")
        return False

def validate_data_models():
    """驗證資料模型"""
    print("\n📊 驗證資料模型...")
    
    try:
        from src.chinese_graphrag.models.document import Document
        from src.chinese_graphrag.models.entity import Entity
        from src.chinese_graphrag.models.text_unit import TextUnit
        
        # 測試文件模型
        doc = Document(
            id="test_doc",
            title="測試文件",
            content="這是一個測試文件的內容",
            metadata={"source": "test"}
        )
        print(f"✅ Document 模型創建成功: {doc.title}")
        
        # 測試實體模型
        entity = Entity(
            id="test_entity",
            name="人工智慧",
            type="概念",
            description="電腦科學的重要分支"
        )
        print(f"✅ Entity 模型創建成功: {entity.name}")
        
        # 測試文本單元模型
        text_unit = TextUnit(
            id="test_unit",
            text="測試文本單元",
            document_id="test_doc",
            chunk_index=0
        )
        print(f"✅ TextUnit 模型創建成功: {text_unit.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ 資料模型驗證失敗: {e}")
        traceback.print_exc()
        return False

def validate_performance():
    """驗證基本效能"""
    print("\n⚡ 驗證基本效能...")
    
    try:
        from src.chinese_graphrag.processors.chinese_text_processor import ChineseTextProcessor
        
        processor = ChineseTextProcessor()
        
        # 效能測試資料
        test_texts = [
            "人工智慧技術發展迅速" * 10,
            "機器學習應用廣泛" * 20,
            "深度學習模型複雜" * 15
        ]
        
        # 批次處理測試
        start_time = time.time()
        processed_texts = []
        
        for text in test_texts:
            processed = processor.preprocess_text(text)
            chunks = processor.split_text(processed, chunk_size=100)
            processed_texts.append({
                "original_length": len(text),
                "processed_length": len(processed),
                "chunks_count": len(chunks)
            })
        
        total_time = time.time() - start_time
        throughput = len(test_texts) / total_time
        
        print(f"✅ 批次處理效能測試完成")
        print(f"   處理文件數: {len(test_texts)}")
        print(f"   總耗時: {total_time:.3f}s")
        print(f"   吞吐量: {throughput:.2f} docs/s")
        
        # 效能要求檢查
        if throughput >= 5.0:
            print("✅ 效能符合要求 (>= 5 docs/s)")
            return True
        else:
            print("⚠️ 效能低於預期 (< 5 docs/s)")
            return False
        
    except Exception as e:
        print(f"❌ 效能驗證失敗: {e}")
        return False

def main():
    """主函數"""
    print("🚀 開始系統效能驗證")
    print("=" * 50)
    
    # 執行各項驗證
    validations = [
        ("核心模組導入", validate_imports),
        ("中文處理功能", validate_chinese_processing),
        ("配置系統", validate_config_system),
        ("資料模型", validate_data_models),
        ("基本效能", validate_performance)
    ]
    
    results = {}
    
    for name, validator in validations:
        try:
            results[name] = validator()
        except Exception as e:
            print(f"❌ {name} 驗證過程中發生錯誤: {e}")
            results[name] = False
    
    # 生成驗證報告
    print("\n" + "=" * 50)
    print("📊 驗證結果摘要:")
    
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    success_rate = passed_count / total_count * 100
    
    for name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {name}: {status}")
    
    print(f"\n總體結果: {passed_count}/{total_count} 通過 ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 系統驗證通過！系統基本功能正常。")
        return 0
    else:
        print("⚠️ 系統驗證未完全通過，建議檢查失敗項目。")
        return 1

if __name__ == "__main__":
    sys.exit(main())