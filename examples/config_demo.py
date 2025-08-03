#!/usr/bin/env python3
"""
GraphRAG 配置系統示例

展示如何使用配置系統載入和管理 GraphRAG 配置
"""

import os
import tempfile
from pathlib import Path

from chinese_graphrag.config import (
    ConfigLoader,
    GraphRAGConfig,
    ModelSelector,
    TaskType,
    create_default_config,
    load_config,
)


def demo_create_default_config():
    """示例：建立預設配置檔案"""
    print("=== 建立預設配置檔案 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "demo_settings.yaml"
        
        # 建立預設配置
        created_path = create_default_config(config_path)
        print(f"建立配置檔案: {created_path}")
        
        # 讀取並顯示配置內容
        with open(created_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("配置檔案內容（前 500 字元）:")
            print(content[:500] + "..." if len(content) > 500 else content)


def demo_load_config():
    """示例：載入配置"""
    print("\n=== 載入配置 ===")
    
    # 設定必要的環境變數
    os.environ["GRAPHRAG_API_KEY"] = "demo-api-key"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "demo_settings.yaml"
        
        # 建立並載入配置
        create_default_config(config_path)
        
        try:
            config = load_config(config_path)
            print(f"成功載入配置")print(f"模型數量: {len(config.models)}")
            print(f"向量資料庫類型: {config.vector_store.type}")
            print(f"預設 LLM: {config.model_selection.default_llm}")
            print(f"預設 Embedding: {config.model_selection.default_embedding}")
            
            return config
            
        except Exception as e:
            print(f"載入配置失敗: {e}")
            return None
        finally:
            # 清理環境變數
            if "GRAPHRAG_API_KEY" in os.environ:
                del os.environ["GRAPHRAG_API_KEY"]


def demo_model_selection(config: GraphRAGConfig):
    """示例：模型選擇"""
    if not config:
        return
        
    print("\n=== 模型選擇示例 ===")
    
    # 建立模型選擇器
    selector = ModelSelector(config)
    
    # 選擇不同任務的模型
    tasks = [
        TaskType.ENTITY_EXTRACTION,
        TaskType.TEXT_EMBEDDING,
        TaskType.QUESTION_ANSWERING
    ]
    
    for task in tasks:
        print(f"\n任務: {task}")
        
        # 選擇 LLM 模型
        if task in [TaskType.ENTITY_EXTRACTION, TaskType.QUESTION_ANSWERING]:
            llm_name, llm_config = selector.select_llm_model(task)
            print(f"  選擇的 LLM: {llm_name} ({llm_config.model})")
        
        # 選擇 Embedding 模型
        if task in [TaskType.TEXT_EMBEDDING]:
            embedding_name, embedding_config = selector.select_embedding_model(task)
            print(f"  選擇的 Embedding: {embedding_name} ({embedding_config.model})")
            
            # 測試中文上下文
            context = {"language": "zh"}
            zh_embedding_name, zh_embedding_config = selector.select_embedding_model(
                task, context
            )
            print(f"  中文上下文 Embedding: {zh_embedding_name} ({zh_embedding_config.model})")


def demo_model_performance_tracking(config: GraphRAGConfig):
    """示例：模型效能追蹤"""
    if not config:
        return
        
    print("\n=== 模型效能追蹤示例 ===")
    
    selector = ModelSelector(config)
    
    # 模擬記錄模型效能
    model_name = config.model_selection.default_llm
    
    # 記錄多次效能資料
    performance_data = [
        (1.2, True, 0.85, 0.02),
        (1.5, True, 0.90, 0.02),
        (2.1, False, 0.60, 0.02),
        (1.8, True, 0.88, 0.02),
    ]
    
    for response_time, success, quality, cost in performance_data:
        selector.record_model_performance(
            model_name,
            response_time=response_time,
            success=success,
            quality_score=quality,
            cost=cost
        )
    
    # 取得統計資訊
    stats = selector.get_model_statistics(model_name)
    print(f"模型 {model_name} 的統計資訊:")
    print(f"  平均回應時間: {stats['average_response_time']:.2f}s")
    print(f"  成功率: {stats['success_rate']:.2%}")
    print(f"  平均品質分數: {stats['average_quality_score']:.2f}")
    print(f"  平均成本: ${stats['average_cost']:.4f}")


def demo_config_validation():
    """示例：配置驗證"""
    print("\n=== 配置驗證示例 ===")
    
    # 測試無效配置
    invalid_configs = [
        {
            "description": "缺少預設模型",
            "config": {
                "models": {
                    "some_model": {
                        "type": "openai_chat",
                        "model": "gpt-4.1"
                    }
                },
                "vector_store": {
                    "type": "lancedb",
                    "uri": "./data/lancedb"
                },
                "model_selection": {
                    "default_llm": "nonexistent_model",
                    "default_embedding": "some_model"
                }
            }
        },
        {
            "description": "無效的模型類型",
            "config": {
                "models": {
                    "invalid_model": {
                        "type": "invalid_type",
                        "model": "test-model"
                    }
                },
                "vector_store": {
                    "type": "lancedb",
                    "uri": "./data/lancedb"
                }
            }
        }
    ]
    
    for test_case in invalid_configs:
        print(f"\n測試: {test_case['description']}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_case['config'], f)
            config_path = Path(f.name)
        
        try:
            loader = ConfigLoader(config_path)
            config = loader.load_config()
            print("  ❌ 應該失敗但成功了")
        except Exception as e:
            print(f"  ✅ 正確捕獲錯誤: {e}")
        finally:
            config_path.unlink()


def main():
    """主函數"""
    print("GraphRAG 配置系統示例")
    print("=" * 50)
    
    # 示例 1: 建立預設配置
    demo_create_default_config()
    
    # 示例 2: 載入配置
    config = demo_load_config()
    
    # 示例 3: 模型選擇
    demo_model_selection(config)
    
    # 示例 4: 效能追蹤
    demo_model_performance_tracking(config)
    
    # 示例 5: 配置驗證
    demo_config_validation()
    
    print("\n示例完成！")


if __name__ == "__main__":
    main()