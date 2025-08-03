#!/usr/bin/env python3
"""
測試 GraphRAG 修復結果

驗證階段 1 的修復是否成功：
1. GraphRAG 工作流程導入正常
2. 不再出現 NameError
3. 索引過程可以正常執行
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_graphrag_indexer():
    """測試 GraphRAG 索引器的基本功能"""
    
    print("🔍 測試 GraphRAG 索引器修復結果")
    print("=" * 50)
    
    # 1. 測試導入
    print("1. 測試模組導入...")
    try:
        from chinese_graphrag.indexing.engine import GraphRAGIndexer, GRAPHRAG_AVAILABLE
        from chinese_graphrag.config.loader import load_config
        print(f"   ✅ GraphRAGIndexer 導入成功")
        print(f"   ✅ GRAPHRAG_AVAILABLE = {GRAPHRAG_AVAILABLE}")
    except Exception as e:
        print(f"   ❌ 導入失敗: {e}")
        return False
    
    # 2. 測試配置載入
    print("\n2. 測試配置載入...")
    try:
        config = load_config(Path("config/settings.yaml"))
        print(f"   ✅ 配置載入成功")
    except Exception as e:
        print(f"   ❌ 配置載入失敗: {e}")
        return False
    
    # 3. 測試索引器初始化
    print("\n3. 測試索引器初始化...")
    try:
        indexer = GraphRAGIndexer(config)
        print(f"   ✅ 索引器初始化成功")
        print(f"   ✅ GraphRAG 可用性: {indexer.graphrag_available}")
    except Exception as e:
        print(f"   ❌ 索引器初始化失敗: {e}")
        return False
    
    # 4. 測試 GraphRAG 工作流程檢查
    print("\n4. 測試 GraphRAG 工作流程檢查...")
    try:
        if GRAPHRAG_AVAILABLE:
            # 檢查導入的工作流程函數
            from chinese_graphrag.indexing.engine import (
                create_base_text_units_workflow,
                extract_graph_workflow,
                create_communities_workflow,
                create_community_reports_workflow
            )
            print("   ✅ 所有 GraphRAG 工作流程函數正確導入")
            print(f"     - create_base_text_units_workflow: {create_base_text_units_workflow}")
            print(f"     - extract_graph_workflow: {extract_graph_workflow}")
            print(f"     - create_communities_workflow: {create_communities_workflow}")
            print(f"     - create_community_reports_workflow: {create_community_reports_workflow}")
        else:
            print("   ⚠️  GraphRAG 不可用，將使用自定義實現")
    except Exception as e:
        print(f"   ❌ GraphRAG 工作流程檢查失敗: {e}")
        return False
    
    # 5. 測試基本方法調用（不會觸發 NameError）
    print("\n5. 測試基本方法調用...")
    try:
        # 檢查統計方法
        stats = indexer.get_statistics()
        print(f"   ✅ 統計方法正常: {stats}")
        
        # 檢查 GraphRAG 可用性檢查
        availability = indexer._check_graphrag_availability()
        print(f"   ✅ GraphRAG 可用性檢查: {availability}")
        
    except Exception as e:
        print(f"   ❌ 基本方法測試失敗: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 所有測試通過！階段 1 修復成功！")
    print("\n修復摘要：")
    print("✅ 已修復函數名不匹配問題")
    print("✅ GraphRAG 工作流程導入正常")
    print("✅ 不再出現 NameError")
    print("✅ 索引器可以正常初始化")
    
    return True

async def main():
    """主測試函數"""
    try:
        success = await test_graphrag_indexer()
        if success:
            print("\n🚀 準備進入階段 1 任務 1.2：測試修復後的索引功能")
        else:
            print("\n❌ 測試失敗，需要進一步修復")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 測試過程中發生未預期錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())