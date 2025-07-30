# 開發常用指令

## 環境設定
```bash
# 安裝依賴
uv sync

# 安裝開發依賴  
uv sync --extra dev
```

## 開發工具
```bash
# 執行測試
uv run pytest

# 測試覆蓋率
uv run pytest --cov=src/chinese_graphrag --cov-report=html

# 程式碼格式化
uv run black src/ tests/
uv run isort src/ tests/

# 型別檢查
uv run mypy src/

# 程式碼檢查
uv run flake8 src/ tests/
```

## 系統操作
```bash
# 執行主程式
uv run python main.py
uv run chinese-graphrag --help

# 初始化系統
uv run python -m chinese_graphrag.cli init

# 索引文件
uv run chinese-graphrag index --input ./documents --output ./data

# 執行查詢
uv run chinese-graphrag query "您的中文問題"
```

## 任務完成檢查清單
每次完成開發任務後，需要執行以下檢查：
1. 程式碼格式化：`uv run black src/ tests/ && uv run isort src/ tests/`
2. 型別檢查：`uv run mypy src/`
3. 程式碼檢查：`uv run flake8 src/ tests/`
4. 執行測試：`uv run pytest`
5. 測試覆蓋率確認：目標 >90%