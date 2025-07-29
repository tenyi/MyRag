# 中文 GraphRAG 系統

一個基於 Microsoft GraphRAG 框架的知識圖譜檢索增強生成系統，專門針對中文文件處理進行優化。

## 特色功能

- 🇨🇳 **中文優化**：專門針對中文文件處理設計
- 🔄 **多模型支援**：支援多種 LLM 和 Embedding 模型
- 💾 **向量資料庫**：持久化儲存和高效檢索
- 📊 **知識圖譜**：自動建構實體關係和社群結構
- ⚡ **效能優化**：成本控制和智慧路由
- 🛠️ **易於部署**：完整的配置和管理系統

## 系統需求

- Python 3.12+
- uv 套件管理工具

## 快速開始

### 1. 安裝依賴

```bash
uv sync
```

### 2. 配置環境

```bash
# 複製配置範例檔案
cp .env.example .env
cp config/settings.yaml.example config/settings.yaml

# 編輯 .env 檔案，填入您的 API 金鑰
```

### 3. 初始化系統

```bash
# 啟動虛擬環境
uv run python -m chinese_graphrag.cli init
```

### 4. 索引文件

```bash
# 索引中文文件
uv run chinese-graphrag index --input ./documents --output ./data
```

### 5. 查詢系統

```bash
# 執行查詢
uv run chinese-graphrag query "您的中文問題"
```

## 專案結構

```
chinese-graphrag/
├── src/chinese_graphrag/     # 主要程式碼
│   ├── config/              # 配置管理
│   ├── models/              # 資料模型
│   ├── processors/          # 文件處理
│   ├── embeddings/          # Embedding 服務
│   ├── vector_stores/       # 向量資料庫
│   ├── indexing/            # 索引引擎
│   ├── query/               # 查詢引擎
│   └── cli/                 # 命令列介面
├── tests/                   # 測試檔案
├── config/                  # 配置檔案
├── data/                    # 資料目錄
└── logs/                    # 日誌目錄
```

## 開發

### 安裝開發依賴

```bash
uv sync --extra dev
```

### 執行測試

```bash
uv run pytest
```

### 程式碼格式化

```bash
uv run black src/ tests/
uv run isort src/ tests/
```

### 類型檢查

```bash
uv run mypy src/
```

## 授權

MIT License