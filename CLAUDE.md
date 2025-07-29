# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

這是一個基於 Microsoft GraphRAG 框架的知識圖譜檢索增強生成系統，專門針對中文文件處理進行優化。系統採用模組化架構，整合了中文 embedding 模型（BGE-M3）、向量資料庫儲存、以及完整的索引和查詢管道。

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

# 執行特定測試檔案
uv run pytest tests/test_processors/test_chinese_text_processor.py

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

## 架構設計

### 目錄結構
- `src/chinese_graphrag/`: 主要程式碼（src layout）
  - `cli/`: 命令列介面
  - `config/`: 配置管理
  - `models/`: 資料模型（Pydantic）
  - `processors/`: 文件處理器
  - `embeddings/`: Embedding 服務
  - `vector_stores/`: 向量資料庫介面
  - `indexing/`: 索引引擎
  - `query/`: 查詢引擎
- `tests/`: 測試檔案（鏡像 src 結構）
- `config/`: 配置檔案範本
- `data/`: 資料目錄
- `logs/`: 日誌目錄
- `ragtest/`: 測試資料與範例

### 核心元件
1. **文件處理模組** (`processors/`): 處理各種格式的中文文件輸入（txt、pdf、docx、md）
2. **中文文本處理器** (`ChineseTextProcessor`): 使用 jieba 進行中文分詞和文本預處理
3. **BGE-M3 Embedding 服務**: 提供中文優化的向量化服務
4. **向量資料庫管理器**: 使用 LanceDB 管理向量資料的儲存和檢索
5. **GraphRAG 索引引擎**: 整合 GraphRAG 的索引流程，建構知識圖譜
6. **查詢引擎**: 處理中文查詢並返回結構化結果

### 資料模型
- **Document**: 文件模型，包含 id、title、content、metadata 等
- **TextUnit**: 文本單元模型，用於文本分塊
- **Entity**: 實體模型，包含實體資訊和 embedding
- **Relationship**: 關係模型，描述實體間的關係
- **Community**: 社群模型，表示實體群組和層次結構

## 技術堆疊

- **語言**: Python 3.12+
- **套件管理**: uv
- **核心框架**: Microsoft GraphRAG (>=2.4.0)
- **中文處理**: jieba (>=0.42.1)
- **資料驗證**: Pydantic (>=2.0.0)
- **向量資料庫**: LanceDB (>=0.5.0)
- **Embedding**: sentence-transformers (>=5.0.0)
- **測試框架**: pytest + pytest-cov + pytest-asyncio
- **程式碼品質**: black + isort + flake8 + mypy

## 開發慣例

### 命名規範
- 模組檔案：snake_case (`chinese_text_processor.py`)
- 類別名稱：PascalCase (`ChineseTextProcessor`)
- 函數名稱：snake_case (`extract_entities`)
- 常數：UPPER_SNAKE_CASE (`DEFAULT_BATCH_SIZE`)

### 程式碼風格
- 行長度：88 字元（black 標準）
- 必須使用完整的型別提示
- 使用中文註解和文件字串
- 測試覆蓋率目標 >90%

### 文件字串格式
使用中文描述功能和用途：
```python
def process_text(text: str) -> List[str]:
    """處理中文文本。
    
    Args:
        text: 待處理的中文文本
        
    Returns:
        處理後的文本片段列表
        
    Raises:
        ValueError: 當輸入文本為空時
    """
```

### 測試結構
- 測試檔案鏡像 src 結構
- 測試類別使用 `Test` 前綴
- 每個模組/功能都有對應的測試檔案
- 特別關注中文文本處理測試

## 配置管理

- 使用 YAML 格式配置檔案 (`config/settings.yaml`)
- 環境變數透過 `.env` 檔案管理
- 支援多環境配置
- 配置範本放在 `config/` 目錄

## 錯誤處理

定義了具體的異常類型：
- `DocumentProcessingError`: 文件處理相關錯誤
- `EmbeddingServiceError`: Embedding 服務錯誤
- `DatabaseError`: 資料庫相關錯誤

使用重試機制處理暫時性錯誤，提供降級處理和詳細錯誤記錄。

## 中文特定處理

- 使用 jieba 進行中文分詞
- 支援繁體/簡體中文處理
- 針對中文語言特性進行文本分割
- 中文實體識別和關係提取
- 中文自然語言查詢處理

## 效能考量

- 支援批次處理和並行處理
- 查詢響應時間目標 < 30秒
- 支援水平擴展和負載均衡
- 記憶體優化和分頁處理機制