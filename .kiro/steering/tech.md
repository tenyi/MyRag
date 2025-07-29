# 技術堆疊與建置系統

## 建置系統

- **套件管理器**：uv (現代 Python 套件管理工具)
- **Python 版本**：3.12+
- **建置後端**：hatchling

## 核心技術堆疊

### 主要依賴

- **GraphRAG**：Microsoft GraphRAG 框架 (>=2.4.0)
- **資料驗證**：Pydantic (>=2.0.0) - 強型別資料模型
- **中文處理**：jieba (>=0.42.1) - 中文分詞
- **向量資料庫**：LanceDB (>=0.5.0)
- **Embedding**：sentence-transformers (>=5.0.0)
- **Web 框架**：FastAPI + Uvicorn
- **CLI 工具**：Click
- **日誌系統**：Loguru
- **資料處理**：pandas, numpy

### 開發工具

- **測試框架**：pytest + pytest-cov + pytest-asyncio
- **程式碼格式化**：black + isort
- **程式碼檢查**：flake8 + mypy
- **Git hooks**：pre-commit
- Always reply in zh-TW.

## 常用指令

### 環境設定

```bash
# 安裝依賴
uv sync

# 安裝開發依賴
uv sync --extra dev

```

### 開發指令

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

# 執行主程式
uv run python main.py
uv run chinese-graphrag --help
```

### 系統操作

```bash
# 初始化系統
uv run python -m chinese_graphrag.cli init

# 索引文件
uv run chinese-graphrag index --input ./documents --output ./data

# 執行查詢
uv run chinese-graphrag query "您的中文問題"
```

## 程式碼品質標準

- **行長度**：88 字元 (black 標準)
- **型別提示**：必須使用完整的型別提示
- **測試覆蓋率**：目標 >90%
- **文件語言**：中文註解和文件字串
