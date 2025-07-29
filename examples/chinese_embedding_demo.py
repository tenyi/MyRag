#!/usr/bin/env python3
"""
中文優化 Embedding 服務示例

展示如何使用中文優化的 embedding 服務進行文本向量化和品質評估
"""

import asyncio
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from chinese_graphrag.embeddings import (
    ChineseOptimizedEmbeddingService,
    ChineseEmbeddingConfig,
    create_chinese_optimized_service,
    ChineseEmbeddingEvaluator,
    chinese_quality_benchmark
)
from loguru import logger


async def demo_chinese_optimized_embedding():
    """演示中文優化 embedding 服務"""
    
    logger.info("=== 中文優化 Embedding 服務示例 ===")
    
    # 1. 創建中文優化配置
    config = ChineseEmbeddingConfig(
        primary_model="BAAI/bge-m3",
        fallback_models=["text2vec-base-chinese", "m3e-base"],
        enable_preprocessing=True,
        normalize_text=True,
        apply_chinese_weighting=True,
        enable_quality_check=True,
        min_chinese_ratio=0.3
    )
    
    # 2. 創建服務
    logger.info("創建中文優化 embedding 服務...")
    service = ChineseOptimizedEmbeddingService(config=config)
    
    try:
        # 3. 載入模型
        logger.info("載入模型...")
        await service.load_model()
        
        # 4. 準備測試文本
        test_texts = [
            "人工智慧技術在自然語言處理領域取得了顯著進展。",
            "深度學習模型如 BERT、GPT 等在各種 NLP 任務中表現出色。",
            "知識圖譜結合檢索增強生成技術，能夠提供更準確的問答系統。",
            "中文文本處理面臨著分詞、語義理解等多重挑戰。",
            "企業級應用需要高效準確的文本向量化技術支撐。",
            "This is an English text mixed with Chinese: 這是中英文混合的文本。",
            "短文本",
            "這是一個相對較長的中文文本，包含了多個句子和豐富的語義資訊，用於測試模型對不同長度文本的處理能力。"
        ]
        
        # 5. 執行向量化
        logger.info("執行文本向量化...")
        result = await service.embed_texts(test_texts, show_progress=True)
        
        logger.info(f"向量化完成:")
        logger.info(f"  - 文本數量: {len(result.texts)}")
        logger.info(f"  - 向量維度: {result.dimensions}")
        logger.info(f"  - 處理時間: {result.processing_time:.3f}秒")
        logger.info(f"  - 使用模型: {result.model_name}")
        
        # 6. 顯示向量統計資訊
        import numpy as np
        norms = np.linalg.norm(result.embeddings, axis=1)
        logger.info(f"  - 向量範數統計: 平均={np.mean(norms):.3f}, 標準差={np.std(norms):.3f}")
        
        # 7. 計算文本相似度示例
        logger.info("\n計算文本相似度示例:")
        similar_texts = [
            "人工智慧技術發展迅速",
            "AI技術進步很快"
        ]
        
        similarity = await service.compute_similarity(
            similar_texts[0], 
            similar_texts[1], 
            method="cosine"
        )
        logger.info(f"  '{similar_texts[0]}' 與 '{similar_texts[1]}' 的相似度: {similarity:.3f}")
        
        # 8. 執行中文品質評估
        logger.info("\n執行中文品質評估...")
        quality_result = await service.evaluate_chinese_quality(test_texts)
        
        if 'error' not in quality_result:
            logger.info("品質評估結果:")
            logger.info(f"  - 綜合品質分數: {quality_result['overall_quality']['overall']:.3f}")
            logger.info(f"  - 文本品質: {quality_result['overall_quality']['text_quality']:.3f}")
            logger.info(f"  - Embedding品質: {quality_result['overall_quality']['embedding_quality']:.3f}")
            logger.info(f"  - 中文特化品質: {quality_result['overall_quality']['chinese_quality']:.3f}")
            
            # 顯示中文指標
            chinese_metrics = quality_result['chinese_metrics']
            logger.info(f"  - 平均中文比例: {chinese_metrics['chinese_ratio_mean']:.3f}")
            logger.info(f"  - 中文文本數量: {chinese_metrics['chinese_text_count']}/{chinese_metrics['total_text_count']}")
        else:
            logger.error(f"品質評估失敗: {quality_result['error']}")
        
        # 9. 顯示模型資訊
        logger.info("\n模型資訊:")
        model_info = service.get_model_info()
        logger.info(f"  - 主要模型: {model_info['primary_model']}")
        logger.info(f"  - 當前服務: {model_info['current_service']}")
        logger.info(f"  - 中文優化: {model_info['chinese_optimized']}")
        logger.info(f"  - 預處理啟用: {model_info['preprocessing_enabled']}")
        logger.info(f"  - 中文權重啟用: {model_info['chinese_weighting_enabled']}")
        logger.info(f"  - 可用備用服務: {model_info['available_fallback_services']}")
        
        # 10. 效能指標
        logger.info("\n效能指標:")
        metrics = service.get_metrics()
        logger.info(f"  - 總請求數: {metrics.total_requests}")
        logger.info(f"  - 平均處理時間: {metrics.average_processing_time:.3f}秒")
        logger.info(f"  - 成功率: {metrics.success_rate:.3f}")
        logger.info(f"  - 錯誤數: {metrics.error_count}")
        
    except Exception as e:
        logger.error(f"示例執行失敗: {e}")
        raise
    
    finally:
        # 11. 清理資源
        logger.info("清理資源...")
        await service.unload_model()


async def demo_chinese_embedding_evaluation():
    """演示中文 embedding 評估功能"""
    
    logger.info("\n=== 中文 Embedding 評估示例 ===")
    
    try:
        # 創建多個服務進行比較
        services = []
        
        # BGE-M3 服務
        bge_service = create_chinese_optimized_service(
            primary_model="BAAI/bge-m3",
            enable_preprocessing=True,
            apply_chinese_weighting=True
        )
        services.append(bge_service)
        
        # 如果有其他模型可用，也可以添加
        # text2vec_service = create_chinese_optimized_service(
        #     primary_model="text2vec-base-chinese",
        #     enable_preprocessing=True,
        #     apply_chinese_weighting=False
        # )
        # services.append(text2vec_service)
        
        # 執行中文品質基準測試
        logger.info("執行中文品質基準測試...")
        benchmark_result = await chinese_quality_benchmark(
            services=services,
            output_dir="evaluation_results"
        )
        
        logger.info("基準測試完成:")
        logger.info(f"  - 評估服務數: {benchmark_result['services_evaluated']}")
        
        if 'best_service' in benchmark_result:
            best = benchmark_result['best_service']
            logger.info(f"  - 最佳服務: {best['name']} (分數: {best['score']:.3f})")
        
        # 顯示各服務的詳細結果
        for service_name, result in benchmark_result['results'].items():
            if 'overall_score' in result:
                logger.info(f"  - {service_name}: {result['overall_score']:.3f}")
            else:
                logger.warning(f"  - {service_name}: 評估失敗")
        
    except Exception as e:
        logger.error(f"評估示例執行失敗: {e}")
        raise
    
    finally:
        # 清理所有服務
        for service in services:
            try:
                await service.unload_model()
            except:
                pass


async def main():
    """主函數"""
    try:
        # 執行基本示例
        await demo_chinese_optimized_embedding()
        
        # 執行評估示例
        await demo_chinese_embedding_evaluation()
        
        logger.info("\n=== 所有示例執行完成 ===")
        
    except Exception as e:
        logger.error(f"示例執行失敗: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # 配置日誌
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 執行示例
    exit_code = asyncio.run(main())
    sys.exit(exit_code)