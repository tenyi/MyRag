#!/usr/bin/env python3
"""Chinese GraphRAG 主入口點"""

def main():
    """啟動 Chinese GraphRAG CLI"""
    try:
        from src.chinese_graphrag.cli.main import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"導入錯誤: {e}")
        print("請確保已正確安裝依賴: uv sync")
        exit(1)
    except Exception as e:
        print(f"啟動失敗: {e}")
        exit(1)

if __name__ == "__main__":
    main()
