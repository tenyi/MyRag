#!/usr/bin/env python3
"""
Embedding 服務架構示範

展示如何使用多種 embedding 模型和管理器
"""

import asyncio
import os
from pathlib import Path

# 添加 src 目錄到 Python 路徑
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chinese_graphrag.embeddings import (
    EmbeddingManager,
    BGEM3EmbeddingService,
    LocalEmbeddingService,
    OpenAIEmbeddingService,
    EmbeddingEvaluator,
    quick_benchmark,
    create_bge_m3_service,
    create_text2vec_service,
    create_openai_service
)

from loguru import logger


async def demo_basic_usage():
    """基本使用示範"""
    logger.info("=== 基本使用示範 ===")
    
    # 建立 BGE-M3 服務（如果可用）
    try:
        bge_service = create_bge_m3_service(device="cpu")  # 使用 CPU 以確保相容性
        
        async with bge_service:
            # 測試文本
            texts = [
                "這是一個中文測試文本。",
                "人工智慧技術發展迅速。",
                "知識圖譜在 NLP 中很重要。"
            ]
            
            logger.info(f"使用 {bge_service.model_name} 進行向量化...")
            result = await bge_service.embed_texts(texts)
            
            logger.info(f"向量化完成:")
            logger.info(f"  - 文本數量: {len(result.texts)}")
            logger.info(f"  - 向量維度: {result.dimensions}")
            logger.info(f"  - 處理時間: {result.processing_time:.3f}秒")
            logger.info(f"  - 向量形狀: {result.embeddings.shape}")
            
            # 計算相似度
            similarity = await bge_service.compute_similarity(texts[0], texts[1])
            logger.info(f"文本相似度: {similarity:.4f}")
            
    except ImportError as e:
        logger.warning(f"BGE-M3 服務不可用: {e}")
    except Exception as e:
        logger.error(f"BGE-M3 服務錯誤: {e}")


async def demo_manager_usage():
    """管理器使用示範"""
    logger.info("=== 管理器使用示範 ===")
    
    # 建立管理器
    manager = EmbeddingManager(enable_fallback=True)
    
    # 註冊多個服務
    services_to_register = []
    
    # 嘗試註冊 BGE-M3 服務
    try:
        bge_service = create_bge_m3_service(device="cpu")
        services_to_register.append(("bge-m3", bge_service, True))
    except ImportError:
        logger.warning("BGE-M3 服務不可用，跳過註冊")
    
    # 嘗試註冊 text2vec 服務
    try:
        text2vec_service = create_text2vec_service(device="cpu")
        services_to_register.append(("text2vec", text2vec_service, False))
    except ImportError:
        logger.warning("text2vec 服務不可用，跳過註冊")
    
    # 嘗試註冊 OpenAI 服務（如果有 API 金鑰）
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_service = create_openai_service()
            services_to_register.append(("openai", openai_service, False))
        except ImportError:
            logger.warning("OpenAI 服務不可用，跳過註冊")
    
    # 註冊服務
    for name, service, is_default in services_to_register:
        manager.register_service(name, service, set_as_default=is_default)
        logger.info(f"註冊服務: {name}")
    
    if not services_to_register:
        logger.warning("沒有可用的 embedding 服務")
        return
    
    try:
        # 載入所有模型
        logger.info("載入所有模型...")
        load_results = await manager.load_all_models()
        
        for service_name, success in load_results.items():
            status = "成功" if success else "失敗"
            logger.info(f"  - {service_name}: {status}")
        
        # 列出服務
        services_info = manager.list_services()
        logger.info(f"已註冊 {len(services_info)} 個服務:")
        for info in services_info:
            default_mark = " (預設)" if info['is_default'] else ""
            logger.info(f"  - {info['service_name']}: {info['model_name']}{default_mark}")
        
        # 測試向量化
        texts = ["測試文本1", "測試文本2", "測試文本3"]
        
        logger.info("使用預設服務進行向量化...")
        result = await manager.embed_texts(texts)
        logger.info(f"使用模型: {result.model_name}")
        logger.info(f"向量形狀: {result.embeddings.shape}")
        
        # 測試智慧路由
        if len(services_to_register) > 1:
            logger.info("測試智慧路由...")
            result = await manager.smart_route_request(texts, strategy="fastest")
            logger.info(f"智慧路由選擇模型: {result.model_name}")
        
        # 健康檢查
        logger.info("執行健康檢查...")
        health_results = await manager.health_check_all()
        for service_name, health in health_results.items():
            logger.info(f"  - {service_name}: {health['status']}")
        
        # 效能指標
        metrics_summary = manager.get_metrics_summary()
        logger.info(f"效能指標摘要:")
        logger.info(f"  - 總服務數: {metrics_summary['total_services']}")
        logger.info(f"  - 已載入服務數: {metrics_summary['loaded_services']}")
        
    finally:
        # 卸載所有模型
        await manager.unload_all_models()
        logger.info("所有模型已卸載")


async def demo_evaluation():
    """效能評估示範"""
    logger.info("=== 效能評估示範 ===")
    
    # 建立可用的服務列表
    available_services = []
    
    # 嘗試建立 BGE-M3 服務
    try:
        bge_service = create_bge_m3_service(device="cpu")
        available_services.append(bge_service)
        logger.info("添加 BGE-M3 服務到評估列表")
    except ImportError:
        logger.warning("BGE-M3 服務不可用")
    
    # 嘗試建立 text2vec 服務
    try:
        text2vec_service = create_text2vec_service(device="cpu")
        available_services.append(text2vec_service)
        logger.info("添加 text2vec 服務到評估列表")
    except ImportError:
        logger.warning("text2vec 服務不可用")
    
    if not available_services:
        logger.warning("沒有可用的服務進行評估")
        return
    
    try:
        # 執行快速基準測試
        logger.info(f"開始評估 {len(available_services)} 個服務...")
        
        # 建立輸出目錄
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        # 執行基準測試
        report = await quick_benchmark(available_services, str(output_dir))
        
        # 顯示結果摘要
        logger.info("評估完成！")
        logger.info(f"總測試數: {report['evaluation_summary']['total_tests']}")
        logger.info(f"測試模型數: {report['evaluation_summary']['models_tested']}")
        
        # 顯示各模型效能
        for model_name, performance in report['model_performance'].items():
            summary = performance['summary']
            logger.info(f"{model_name} 效能:")
            logger.info(f"  - 平均處理時間: {summary['average_processing_time']:.4f}秒")
            logger.info(f"  - 平均吞吐量: {summary['average_throughput']:.2f} 文本/秒")
            logger.info(f"  - 記憶體增加: {summary['average_memory_increase_mb']:.2f}MB")
            logger.info(f"  - 成功率: {summary['success_rate']:.3f}")
        
    except Exception as e:
        logger.error(f"評估過程中發生錯誤: {e}")


async def main():
    """主函數"""
    logger.info("開始 Embedding 服務架構示範")
    
    try:
        # 基本使用示範
        await demo_basic_usage()
        
        print("\n" + "="*50 + "\n")
        
        # 管理器使用示範
        await demo_manager_usage()
        
        print("\n" + "="*50 + "\n")
        
        # 效能評估示範
        await demo_evaluation()
        
    except KeyboardInterrupt:
        logger.info("示範被用戶中斷")
    except Exception as e:
        logger.error(f"示範過程中發生錯誤: {e}")
    
    logger.info("Embedding 服務架構示範完成")


if __name__ == "__main__":
    # 設定日誌格式
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 執行示範
    asyncio.run(main())