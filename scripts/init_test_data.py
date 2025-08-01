#!/usr/bin/env python3
"""
測試資料初始化腳本

用於設定和管理測試環境的資料。
"""

import argparse
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_data_manager import TestDataManager


def init_test_data(data_dir: str = None, reset: bool = False):
    """初始化測試資料"""
    print("🚀 開始初始化測試資料...")
    
    # 建立測試資料管理器
    manager = TestDataManager(data_dir)
    
    if reset:
        print("⚠️  重設現有測試資料...")
        manager.reset_all_data(confirm=True)
    
    # 建立所有測試資料
    print("📄 建立範例文件...")
    documents = manager.create_sample_documents()
    
    print("⚙️ 建立測試配置...")
    configs = manager.create_test_configs()
    
    print("🧪 建立測試夾具...")
    fixtures = manager.create_test_fixtures()
    
    # 驗證資料
    print("✅ 驗證資料完整性...")
    integrity_results = manager.validate_data_integrity()
    valid_count = sum(integrity_results.values())
    total_count = len(integrity_results)
    
    if valid_count == total_count:
        print(f"✅ 所有資料驗證通過 ({valid_count}/{total_count})")
    else:
        print(f"⚠️  部分資料驗證失敗 ({valid_count}/{total_count})")
        for name, is_valid in integrity_results.items():
            if not is_valid:
                print(f"  ❌ {name}")
    
    # 匯出清單
    print("📋 匯出資料清單...")
    manifest_file = manager.export_data_manifest()
    
    print(f"✨ 測試資料初始化完成！")
    print(f"   資料目錄: {manager.base_dir}")
    print(f"   清單檔案: {manifest_file}")
    print(f"   總資料數: {len(manager.registry)}")
    
    return manager


def clean_test_data(data_dir: str = None, max_age_hours: int = 24):
    """清理測試資料"""
    print("🧹 開始清理測試資料...")
    
    manager = TestDataManager(data_dir)
    
    # 清理臨時資料
    cleaned_count = manager.cleanup_temporary_data(max_age_hours)
    print(f"✅ 已清理 {cleaned_count} 個過期臨時目錄")
    
    return cleaned_count


def validate_test_data(data_dir: str = None):
    """驗證測試資料"""
    print("🔍 開始驗證測試資料...")
    
    manager = TestDataManager(data_dir)
    
    # 檢查資料完整性
    integrity_results = manager.validate_data_integrity()
    
    print("驗證結果:")
    for name, is_valid in integrity_results.items():
        status = "✅" if is_valid else "❌"
        data_info = manager.get_data_info(name)
        print(f"  {status} {name} ({data_info.data_type})")
    
    valid_count = sum(integrity_results.values())
    total_count = len(integrity_results)
    print(f"\n總計: {valid_count}/{total_count} 有效")
    
    return integrity_results


def list_test_data(data_dir: str = None, data_type: str = None):
    """列出測試資料"""
    print("📋 測試資料清單:")
    
    manager = TestDataManager(data_dir)
    
    data_list = manager.list_data(data_type=data_type)
    
    if not data_list:
        print("  (無資料)")
        return
    
    # 按類型分組顯示
    by_type = {}
    for data_info in data_list:
        if data_info.data_type not in by_type:
            by_type[data_info.data_type] = []
        by_type[data_info.data_type].append(data_info)
    
    for dtype, items in by_type.items():
        print(f"\n{dtype.upper()}:")
        for data_info in items:
            size_mb = data_info.size_bytes / (1024 * 1024)
            print(f"  📄 {data_info.name}")
            print(f"     {data_info.description}")
            print(f"     檔案: {data_info.file_path}")
            print(f"     大小: {size_mb:.2f} MB")
            print(f"     標籤: {', '.join(data_info.tags)}")
            print()


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description="測試資料管理工具")
    parser.add_argument("--data-dir", help="測試資料目錄")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # init 命令
    init_parser = subparsers.add_parser("init", help="初始化測試資料")
    init_parser.add_argument("--reset", action="store_true", help="重設現有資料")
    
    # clean 命令
    clean_parser = subparsers.add_parser("clean", help="清理測試資料")
    clean_parser.add_argument("--max-age-hours", type=int, default=24, 
                             help="清理超過指定小時數的臨時資料")
    
    # validate 命令
    validate_parser = subparsers.add_parser("validate", help="驗證測試資料")
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出測試資料")
    list_parser.add_argument("--type", help="篩選資料類型")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "init":
            init_test_data(args.data_dir, args.reset)
        elif args.command == "clean":
            clean_test_data(args.data_dir, args.max_age_hours)
        elif args.command == "validate":
            validate_test_data(args.data_dir)
        elif args.command == "list":
            list_test_data(args.data_dir, args.type)
        
    except Exception as e:
        print(f"❌ 執行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()