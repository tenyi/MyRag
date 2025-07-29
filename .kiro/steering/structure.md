# 專案結構與架構模式

## 目錄結構

```text
chinese-graphrag/
├── src/chinese_graphrag/     # 主要程式碼 (src layout)
│   ├── __init__.py          # 套件初始化
│   ├── cli/                 # 命令列介面
│   ├── config/              # 配置管理
│   ├── models/              # 資料模型 (Pydantic)
│   ├── processors/          # 文件處理器
│   ├── embeddings/          # Embedding 服務
│   ├── vector_stores/       # 向量資料庫介面
│   ├── indexing/            # 索引引擎
│   └── query/               # 查詢引擎
├── tests/                   # 測試檔案 (鏡像 src 結構)
├── config/                  # 配置檔案範本
├── data/                    # 資料目錄
├── logs/                    # 日誌目錄
└── ragtest/                 # 測試資料與範例
```

## 架構模式

### 資料模型層 (models/)

- **基礎模型**：`BaseModel` 提供共同功能
- **領域模型**：Entity, Document, Relationship, Community, TextUnit
- **設計原則**：
  - 使用 Pydantic 進行資料驗證
  - 包含中文驗證邏輯
  - 提供序列化/反序列化方法
  - 自動時間戳記管理

### 處理器層 (processors/)

- 文件預處理
- 實體提取
- 關係識別
- 社群偵測

### 服務層

- **Embedding 服務** (embeddings/)：向量化處理
- **向量儲存** (vector_stores/)：資料持久化
- **索引引擎** (indexing/)：建構知識圖譜
- **查詢引擎** (query/)：檢索與生成

## 程式碼慣例

### 檔案命名

- 模組檔案：小寫加底線 (`entity_processor.py`)
- 類別名稱：PascalCase (`EntityProcessor`)
- 函數名稱：小寫加底線 (`extract_entities`)
- 常數：大寫加底線 (`DEFAULT_BATCH_SIZE`)

### 文件字串格式

```python
"""
模組/類別/函數的中文描述

詳細說明功能和用途
"""
```

### 型別提示

- 所有公開函數必須有完整型別提示
- 使用 `typing` 模組的泛型類型
- 複雜類型使用 `TypeAlias` 定義

### 錯誤處理

- 使用具體的異常類型
- 提供中文錯誤訊息
- 記錄適當的日誌等級

### 測試結構

- 測試檔案鏡像 src 結構
- 測試類別使用 `Test` 前綴
- 測試方法使用描述性中文名稱
- 每個模型/功能都有對應的測試檔案

## 配置管理

- 使用 YAML 格式配置檔案
- 環境變數透過 `.env` 檔案管理
- 配置範本放在 `config/` 目錄
- 支援多環境配置

## 日誌規範

- 使用 Loguru 進行日誌管理
- 日誌檔案存放在 `logs/` 目錄
- 支援結構化日誌 (JSON 格式)
- 不同模組使用不同的日誌記錄器
