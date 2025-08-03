# Chinese GraphRAG 配置檔案說明

本目錄包含 Chinese GraphRAG 系統的配置檔案範本和範例。

## 檔案說明

### 核心配置檔案

- **`settings.yaml`** - 完整的系統配置範本
  - 包含所有可用的配置選項
  - 包含詳細的註解說明
  - 適合作為自訂配置的起點

- **`.env.template`** - 環境變數範本檔案
  - 列出所有支援的環境變數
  - 包含變數類型和說明
  - 複製為 `.env` 後填入實際值

### 環境特定配置

- **`dev.yaml`** - 開發環境配置
  - 適用於本地開發和測試
  - 使用較小的資源配置
  - 啟用除錯功能

- **`prod.yaml`** - 生產環境配置
  - 適用於生產部署
  - 優化效能和可靠性
  - 包含監控和安全設定

## 使用方法

### 1. 基本設定

1. 複製環境變數範本：
   ```bash
   cp .env.template .env
   ```

2. 編輯 `.env` 檔案，填入必要的配置：
   ```bash
   # 必需設定
   GRAPHRAG_API_KEY=your-api-key-here
   
   # 可選設定
   GRAPHRAG_MODEL=gpt-4.1-mini
   GRAPHRAG_DEVICE=auto
   ```

3. 選擇合適的配置檔案：
   ```bash
   # 開發環境
   chinese-graphrag --config config/dev.yaml
   
   # 生產環境
   chinese-graphrag --config config/prod.yaml
   
   # 自訂配置
   chinese-graphrag --config config/settings.yaml
   ```

### 2. 配置檔案結構

配置檔案採用 YAML 格式，主要包含以下區塊：

```yaml
# 模型配置
models:
  model_name:
    type: "model_type"
    model: "model_path"
    # 其他參數...

# 向量資料庫配置
vector_store:
  type: "lancedb"
  uri: "./data/lancedb"
  # 其他參數...

# 處理配置
chunks:
  size: 1000
  overlap: 200
  # 其他參數...

# 其他配置區塊...
```

### 3. 環境變數替換

配置檔案支援環境變數替換語法：

```yaml
# 使用環境變數
api_key: "${GRAPHRAG_API_KEY}"

# 使用環境變數並提供預設值
device: "${GRAPHRAG_DEVICE:auto}"

# 複雜範例
model_config:
  type: "${MODEL_TYPE:openai_chat}"
  api_key: "${GRAPHRAG_API_KEY}"
  max_tokens: "${MAX_TOKENS:4000}"
```

## 配置選項說明

### 模型配置 (models)

支援的模型類型：

#### LLM 模型
- `openai_chat` - OpenAI Chat 模型
- `azure_openai_chat` - Azure OpenAI Chat 模型
- `anthropic` - Anthropic Claude 模型
- `local` - 本地模型

#### Embedding 模型
- `bge_m3` - BGE-M3 中文優化模型
- `openai_embedding` - OpenAI Embedding 模型
- `azure_openai_embedding` - Azure OpenAI Embedding 模型
- `ollama` - Ollama Embedding 模型

### 向量資料庫配置 (vector_store)

支援的向量資料庫：
- `lancedb` - LanceDB（推薦）
- `faiss` - FAISS
- `chroma` - ChromaDB

### 中文處理配置 (chinese_processing)

- `tokenizer` - 中文分詞器（jieba）
- `enable_traditional_chinese` - 是否支援繁體中文
- `enable_pos_tagging` - 是否啟用詞性標註
- `enable_ner` - 是否啟用命名實體識別

### 文本分塊配置 (chunks)

- `size` - 文本塊大小（標記數）
- `overlap` - 文本塊重疊大小
- `strategy` - 分塊策略（token/sentence）
- `separators` - 分隔符列表

### 索引配置 (indexing)

- `enable_entity_extraction` - 啟用實體提取
- `enable_relationship_extraction` - 啟用關係提取
- `enable_community_detection` - 啟用社群偵測
- `enable_community_reports` - 啟用社群報告生成

### 查詢配置 (query)

- `enable_global_search` - 啟用全域搜尋
- `enable_local_search` - 啟用局部搜尋
- `top_k` - 返回結果數量
- `similarity_threshold` - 相似度閾值

## 最佳實踐

### 1. 環境分離

建議為不同環境使用不同的配置檔案：

```bash
# 開發環境
config/
├── dev.yaml          # 開發配置
├── test.yaml         # 測試配置
└── prod.yaml         # 生產配置
```

### 2. 敏感資訊管理

- 將 API 金鑰等敏感資訊放在環境變數中
- 不要將 `.env` 檔案提交到版本控制
- 使用環境變數範本檔案（`.env.template`）

### 3. 配置驗證

系統會自動驗證配置檔案：

```bash
# 驗證配置檔案
chinese-graphrag validate-config --config config/settings.yaml
```

### 4. 配置繼承

可以使用基礎配置檔案並覆蓋特定選項：

```yaml
# 繼承基礎配置
base_config: "config/settings.yaml"

# 覆蓋特定選項
logging:
  level: "DEBUG"

debug:
  enable_debug_mode: true
```

## 常見問題

### Q: 如何設定多個 API 金鑰？

A: 在環境變數中設定不同的金鑰，並在配置中引用：

```bash
# .env
OPENAI_API_KEY=your-openai-key
AZURE_API_KEY=your-azure-key
```

```yaml
# config.yaml
models:
  openai_model:
    api_key: "${OPENAI_API_KEY}"
  azure_model:
    api_key: "${AZURE_API_KEY}"
```

### Q: 如何優化記憶體使用？

A: 調整以下配置：

```yaml
parallelization:
  num_threads: 2      # 減少執行緒數
  batch_size: 5       # 減少批次大小

chunks:
  size: 500           # 減少文本塊大小

performance:
  memory_limit_mb: 2048  # 設定記憶體限制
```

### Q: 如何設定 Ollama Embedding 模型？

A: 首先確保 Ollama 服務正在運行，然後配置 embedding 模型：

```yaml
models:
  ollama_embedding_model:
    type: ollama
    model: nomic-embed-text    # 或其他支援的 embedding 模型
    api_base: http://localhost:11434
    batch_size: 32
    max_length: 512
    normalize_embeddings: true

model_selection:
  default_embedding: ollama_embedding_model
```

### Q: 如何啟用 GPU 加速？

A: 設定計算設備：

```yaml
models:
  embedding_model:
    device: "cuda"    # 或 "mps" (Apple Silicon)

performance:
  enable_gpu_acceleration: true
  gpu_memory_fraction: 0.8
```

## 支援

如需更多協助，請參考：
- [使用者指南](../docs/user-guide.md)
- [API 文件](../docs/api.md)
- [故障排除](../docs/troubleshooting.md)