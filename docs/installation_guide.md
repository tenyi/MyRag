# 安裝和配置指南

本指南將協助您完整安裝和配置 Chinese GraphRAG 系統，包含所有必要的步驟和配置選項。

## 目錄

- [系統需求](#系統需求)
- [安裝步驟](#安裝步驟)
- [環境配置](#環境配置)
- [模型配置](#模型配置)
- [資料庫配置](#資料庫配置)
- [驗證安裝](#驗證安裝)
- [常見問題](#常見問題)

## 系統需求

### 硬體需求

#### 最低需求

- **CPU**: 4 核心 (Intel i5 或 AMD Ryzen 5 同等級)
- **記憶體**: 8GB RAM
- **磁碟空間**: 20GB 可用空間
- **網路**: 穩定的網際網路連線

#### 建議需求

- **CPU**: 8 核心或更多 (Intel i7/i9 或 AMD Ryzen 7/9)
- **記憶體**: 16GB RAM 或更多
- **磁碟空間**: 100GB SSD
- **GPU**: NVIDIA GPU (可選，用於加速 embedding 計算)

### 軟體需求

- **作業系統**:
  - Linux (Ubuntu 20.04+ 或 CentOS 8+)
  - macOS 11.0+ (Big Sur)
  - Windows 10/11
- **Python**: 3.11 或更高版本
- **uv**: 套件管理工具

## 安裝步驟

### 1. 安裝 Python 和 uv

#### Linux (Ubuntu/Debian)

```bash
# 安裝 Python 3.11+
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### macOS

```bash
# 使用 Homebrew 安裝 Python
brew install python@3.11

# 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
```

#### Windows

```powershell
# 使用 Chocolatey 安裝 Python
choco install python311

# 安裝 uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 下載專案

```bash
# 克隆專案（如果使用 Git）
git clone https://github.com/your-org/chinese-graphrag.git
cd chinese-graphrag

# 或下載並解壓縮專案檔案
```

### 3. 安裝依賴套件

```bash
# 安裝基本依賴
uv sync

# 安裝開發依賴（如果需要開發）
uv sync --extra dev

# 安裝文件依賴（如果需要建立文件）
uv sync --extra docs
```

### 4. 驗證安裝

```bash
# 檢查 CLI 工具是否正常運作
uv run chinese-graphrag --help

# 檢查版本
uv run chinese-graphrag --version
```

## 環境配置

### 1. 建立環境變數檔案

```bash
# 複製環境變數範本
cp .env.template .env
```

### 2. 編輯環境變數

編輯 `.env` 檔案，設定必要的 API 金鑰和配置：

```bash
# OpenAI API 配置
GRAPHRAG_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# Azure OpenAI 配置（可選）
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# 系統配置
GRAPHRAG_LOG_LEVEL=INFO
GRAPHRAG_DATA_DIR=./data
GRAPHRAG_CACHE_DIR=./cache

# 效能配置
GRAPHRAG_MAX_WORKERS=4
GRAPHRAG_BATCH_SIZE=32
GRAPHRAG_MEMORY_LIMIT=8GB
```

### 3. 建立配置檔案

```bash
# 複製配置範本
cp config/settings.yaml.example config/settings.yaml
```

## 模型配置

### 1. OpenAI 模型配置

編輯 `config/settings.yaml`：

```yaml
models:
  # 預設 LLM 模型
  default_chat_model:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_chat
    model: gpt-5-mini  # 或 gpt-4.1
    model_supports_json: true
    max_tokens: 2000
    temperature: 0.0
    timeout: 60
    max_retries: 3
  
  # 預設 Embedding 模型
  openai_embedding_model:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small
    batch_size: 32
    normalize_embeddings: true
```

### 2. 中文優化模型配置

```yaml
models:
  # 中文優化 Embedding 模型（BGE-M3）
  chinese_embedding_model:
    type: bge_m3
    model: BAAI/bge-m3
    device: auto  # auto, cpu, cuda, mps
    batch_size: 32
    max_length: 512
    normalize_embeddings: true
    cache_enabled: true
```

### 3. Azure OpenAI 配置

```yaml
models:
  # Azure OpenAI 配置
  azure_chat_model:
    type: azure_openai_chat
    model: gpt-4
    api_base: ${AZURE_OPENAI_ENDPOINT}
    api_version: ${AZURE_OPENAI_API_VERSION}
    deployment_name: gpt-4-deployment
    api_key: ${AZURE_OPENAI_API_KEY}
```

## 資料庫配置

### 1. 向量資料庫配置

#### LanceDB（預設）

```yaml
vector_store:
  type: lancedb
  uri: ./data/lancedb
  table_name: embeddings
  metric: cosine
```

#### Chroma（可選）

```yaml
vector_store:
  type: chroma
  persist_directory: ./data/chroma
  collection_name: chinese_graphrag
```

### 2. 圖形資料庫配置

```yaml
graph_store:
  type: parquet
  base_dir: ./data/graph
  entities_file: entities.parquet
  relationships_file: relationships.parquet
  communities_file: communities.parquet
```

## 驗證安裝

### 1. 系統初始化

```bash
# 初始化系統配置
uv run chinese-graphrag init

# 驗證配置
uv run chinese-graphrag validate-config
```

### 2. 測試基本功能

```bash
# 建立測試資料目錄
mkdir -p test_data

# 建立測試文件
echo "這是一個測試文件，用於驗證 Chinese GraphRAG 系統的功能。" > test_data/test.txt

# 執行索引測試
uv run chinese-graphrag index --input test_data --output ./data/test

# 執行查詢測試
uv run chinese-graphrag query "什麼是測試文件？"
```

### 3. 檢查系統狀態

```bash
# 檢查系統狀態
uv run chinese-graphrag status

# 檢查模型狀態
uv run chinese-graphrag model-status

# 檢查資料庫連線
uv run chinese-graphrag db-status
```

## 常見問題

### Q1: 安裝 uv 時遇到權限問題

**解決方案**：

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# 或使用 pip 安裝
pip install uv
```

### Q2: Python 版本不符合需求

**解決方案**：

```bash
# 檢查 Python 版本
python --version

# 如果版本過舊，請安裝 Python 3.11+
# Ubuntu/Debian
sudo apt install python3.11

# macOS
brew install python@3.11

# Windows
# 從 python.org 下載並安裝最新版本
```

### Q3: 記憶體不足錯誤

**解決方案**：

1. 調整批次大小：

```yaml
models:
  chinese_embedding_model:
    batch_size: 16  # 降低批次大小
```

2. 啟用記憶體優化：

```yaml
performance:
  memory_optimization: true
  max_memory_usage: 6GB  # 設定記憶體限制
```

### Q4: GPU 加速無法使用

**解決方案**：

```bash
# 檢查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 安裝 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q5: API 金鑰配置錯誤

**解決方案**：

1. 檢查 `.env` 檔案中的 API 金鑰是否正確
2. 確認 API 金鑰有足夠的配額
3. 測試 API 連線：

```bash
uv run chinese-graphrag test-api
```

### Q6: 中文文件處理問題

**解決方案**：

1. 確認文件編碼為 UTF-8
2. 檢查中文分詞工具是否正確安裝：

```bash
python -c "import jieba; print('jieba 安裝成功')"
```

## 下一步

安裝完成後，您可以：

1. 閱讀 [API 使用文件](api_usage_guide.md)
2. 查看 [程式碼範例和教學](examples_and_tutorials.md)
3. 了解 [架構和設計文件](architecture_design.md)
4. 參考 [故障排除指南](troubleshooting_guide.md)

如果遇到其他問題，請查看 [故障排除指南](troubleshooting_guide.md) 或提交 Issue。
