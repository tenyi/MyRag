# Chinese GraphRAG API 文件索引

> 生成時間：2025-08-01 21:57:27 (Asia/Taipei)
> 版本：v1.3.0

## 文件概覽

本目錄包含 Chinese GraphRAG API 的完整文件，涵蓋 API 索引、資料模型參考和使用指南。

## 文件結構

### 📋 [API 索引](./api_index.md)
**完整的 API 端點索引和功能說明**

- ✅ 所有 API 端點的詳細列表
- ✅ 端點分類和功能描述
- ✅ HTTP 方法和路徑
- ✅ 請求/回應模型
- ✅ 錯誤處理說明
- ✅ CLI 命令支援
- ✅ 部署和監控指南

**適用對象**: 開發者、系統管理員、API 使用者

### 📊 [資料模型參考](./api_models_reference.md)
**詳細的資料模型定義和驗證規則**

- ✅ 所有請求和回應模型
- ✅ 欄位定義和型別說明
- ✅ 驗證規則和限制
- ✅ 序列化配置
- ✅ 使用範例和最佳實踐

**適用對象**: 前端開發者、API 整合開發者

### 🚀 [使用指南](./api_usage_guide.md)
**實用的 API 使用範例和最佳實踐**

- ✅ 快速開始指南
- ✅ 完整工作流程範例
- ✅ 進階使用技巧
- ✅ 錯誤處理策略
- ✅ 效能優化建議
- ✅ 測試和除錯方法

**適用對象**: 應用程式開發者、系統整合者

## 快速導航

### 🎯 我想要...

#### 了解 API 有哪些功能
👉 查看 [API 索引](./api_index.md)

#### 知道如何發送請求
👉 查看 [資料模型參考](./api_models_reference.md)

#### 看實際的程式碼範例
👉 查看 [使用指南](./api_usage_guide.md)

#### 部署 API 服務
👉 查看 [API 索引](./api_index.md)

#### 監控系統狀態
👉 查看 [API 索引](./api_index.md)

#### 處理錯誤和異常
👉 查看 [使用指南](./api_usage_guide.md)

## API 功能概覽

### 🔍 核心功能

| 功能 | 端點 | 說明 |
|------|------|------|
| **文件索引** | `POST /api/v1/index` | 建立文件索引，支援多種格式 |
| **語義查詢** | `POST /api/v1/query` | 執行智慧查詢，支援多種搜索策略 |
| **批次處理** | `POST /api/v1/query/batch` | 批次查詢處理，提高效率 |
| **配置管理** | `GET/PUT /api/v1/config` | 動態配置管理 |
| **系統監控** | `GET /api/v1/monitoring/*` | 完整的監控和警告系統 |

### 🛠️ 管理功能

| 功能 | 端點 | 說明 |
|------|------|------|
| **健康檢查** | `GET /health` | 系統健康狀態檢查 |
| **任務管理** | `GET /api/v1/index/{task_id}` | 索引任務狀態追蹤 |
| **歷史記錄** | `GET /api/v1/query/history` | 查詢歷史和統計 |
| **系統診斷** | `POST /api/v1/monitoring/test` | 系統測試和診斷 |

## 技術規格

### 🏗️ 架構特點

- **RESTful API**: 標準的 REST 架構設計
- **異步處理**: 支援長時間運行的任務
- **批次操作**: 高效的批次處理能力
- **實時監控**: 完整的系統監控和警告
- **中文優化**: 專門針對中文文本處理優化

### 📋 支援格式

- **輸入格式**: TXT, MD, PDF, DOCX, HTML
- **查詢類型**: 語義搜索、關鍵字搜索、圖譜搜索、混合搜索
- **回應格式**: JSON (UTF-8 編碼)
- **認證方式**: API Key, JWT Token

### ⚡ 效能指標

- **查詢響應**: < 500ms (一般查詢)
- **索引速度**: ~2-5 文件/秒 (取決於文件大小)
- **並發支援**: 最多 100 個並發連接
- **批次處理**: 最多 100 個查詢/批次

## 開發環境設定

### 🐍 Python 環境

```bash
# 確保 Python 3.11+
python --version

# 安裝依賴
uv sync

# 啟動開發服務器
uv run chinese-graphrag api server --reload
```

### 🐳 Docker 環境

```bash
# 建構映像
docker build -t chinese-graphrag-api .

# 執行容器
docker run -p 8000:8000 chinese-graphrag-api
```

### 🧪 測試環境

```bash
# 執行 API 測試
uv run pytest tests/api/

# 執行效能測試
uv run chinese-graphrag api perf

# 驗證 API 配置
uv run chinese-graphrag api validate
```

## 常見使用場景

### 📚 知識庫搜索

```python
# 建立企業知識庫索引
client.create_index(
    input_path="./company_docs",
    file_types=["pdf", "docx", "md"]
)

# 智慧查詢
result = client.query(
    "公司的休假政策是什麼？",
    query_type="semantic"
)
```

### 🔬 學術研究

```python
# 學術論文索引
client.create_index(
    input_path="./research_papers",
    file_types=["pdf"],
    batch_size=16  # 較小批次以處理大型 PDF
)

# 研究查詢
result = client.query(
    "深度學習在醫療影像診斷中的最新進展",
    query_type="graph",
    max_results=20
)
```

### 📖 文檔問答

```python
# 技術文檔索引
client.create_index(
    input_path="./tech_docs",
    file_types=["md", "txt", "html"]
)

# 批次問答
questions = [
    "如何安裝這個軟體？",
    "有哪些配置選項？",
    "如何進行故障排除？"
]

results = client.execute_batch_query(questions)
```

## 故障排除

### ❗ 常見問題

#### 服務無法啟動
```bash
# 檢查端口占用
lsof -i :8000

# 檢查配置
chinese-graphrag doctor

# 查看詳細日誌
chinese-graphrag api server --log-level debug
```

#### 查詢結果不準確
```bash
# 檢查索引狀態
chinese-graphrag show-index

# 重建索引
chinese-graphrag index --force-rebuild

# 調整查詢參數
# 使用不同的 query_type 或調整 max_results
```

#### 效能問題
```bash
# 檢查系統資源
chinese-graphrag api perf

# 調整配置
# 增加 batch_size 或 parallel_workers

# 使用批次處理
# 避免頻繁的單個請求
```

### 🔧 除錯工具

- **健康檢查**: `GET /health/detailed`
- **系統測試**: `POST /api/v1/monitoring/test`
- **配置驗證**: `chinese-graphrag api validate`
- **效能測試**: `chinese-graphrag api perf`

## 社群資源

### 📞 支援管道

- **GitHub Issues**: [回報問題](https://github.com/your-org/chinese-graphrag/issues)
- **討論區**: [GitHub Discussions](https://github.com/your-org/chinese-graphrag/discussions)
- **文件**: [完整文件](https://docs.chinese-graphrag.com)

### 🤝 貢獻指南

- **程式碼貢獻**: 查看 [貢獻指南](./contributing_guide.md)
- **文件改進**: 提交 Pull Request
- **問題回報**: 使用 Issue 模板

---

**📝 文件維護**
- 最後更新：2025-08-01
- 維護者：Chinese GraphRAG 開發團隊
- 文件版本：1.3.0

**🔄 自動更新**
此文件由 Serena 自動生成和維護，基於最新的程式碼結構分析。