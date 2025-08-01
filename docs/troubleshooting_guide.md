# 故障排除指南

本指南提供 Chinese GraphRAG 系統常見問題的診斷和解決方案，幫助您快速解決使用過程中遇到的問題。

## 目錄

- [診斷工具](#診斷工具)
- [安裝問題](#安裝問題)
- [配置問題](#配置問題)
- [模型問題](#模型問題)
- [索引問題](#索引問題)
- [查詢問題](#查詢問題)
- [效能問題](#效能問題)
- [API 問題](#api-問題)
- [資料庫問題](#資料庫問題)
- [日誌分析](#日誌分析)
- [聯繫支援](#聯繫支援)

## 診斷工具

### 系統診斷命令

```bash
# 檢查系統整體狀態
uv run chinese-graphrag status

# 檢查配置是否正確
uv run chinese-graphrag validate-config

# 檢查模型狀態
uv run chinese-graphrag model-status

# 檢查資料庫連線
uv run chinese-graphrag db-status

# 執行系統自檢
uv run chinese-graphrag self-check
```

### 日誌查看

```bash
# 查看最新日誌
tail -f logs/chinese_graphrag.log

# 查看錯誤日誌
grep "ERROR" logs/chinese_graphrag.log

# 查看特定時間範圍的日誌
grep "2024-01-01" logs/chinese_graphrag.log
```

## 安裝問題

### Q1: uv 安裝失敗

**症狀**：

```
curl: command not found
或
Permission denied
```

**解決方案**：

1. **Linux/macOS**：

```bash
# 方法 1: 使用 pip 安裝
pip install uv

# 方法 2: 手動下載安裝
wget https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
tar -xzf uv-x86_64-unknown-linux-gnu.tar.gz
sudo mv uv /usr/local/bin/
```

2. **Windows**：

```powershell
# 使用 Scoop
scoop install uv

# 或使用 Chocolatey
choco install uv
```

### Q2: Python 版本不相容

**症狀**：

```
ERROR: This package requires Python >=3.11
```

**解決方案**：

1. **檢查 Python 版本**：

```bash
python --version
python3 --version
```

2. **安裝正確版本**：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-pip

# macOS
brew install python@3.11

# 建立別名
alias python=python3.11
```

### Q3: 依賴套件安裝失敗

**症狀**：

```
ERROR: Could not install packages due to an EnvironmentError
```

**解決方案**：

1. **清理快取**：

```bash
uv cache clean
pip cache purge
```

2. **使用不同的索引**：

```bash
uv sync --index-url https://pypi.org/simple/
```

3. **手動安裝問題套件**：

```bash
uv add package-name --no-deps
```

## 配置問題

### Q4: 配置檔案載入失敗

**症狀**：

```
ConfigurationError: Could not load settings.yaml
```

**解決方案**：

1. **檢查檔案是否存在**：

```bash
ls -la config/settings.yaml
```

2. **驗證 YAML 語法**：

```bash
python -c "import yaml; yaml.safe_load(open('config/settings.yaml'))"
```

3. **使用預設配置**：

```bash
cp config/settings.yaml.example config/settings.yaml
```

### Q5: 環境變數未載入

**症狀**：

```
KeyError: 'GRAPHRAG_API_KEY'
```

**解決方案**：

1. **檢查 .env 檔案**：

```bash
cat .env | grep GRAPHRAG_API_KEY
```

2. **手動設定環境變數**：

```bash
export GRAPHRAG_API_KEY="your-api-key"
```

3. **驗證環境變數**：

```bash
echo $GRAPHRAG_API_KEY
```

## 模型問題

### Q6: OpenAI API 連線失敗

**症狀**：

```
OpenAIError: The api_key client option must be set
```

**解決方案**：

1. **檢查 API 金鑰**：

```bash
# 測試 API 金鑰
curl -H "Authorization: Bearer $GRAPHRAG_API_KEY" \
     https://api.openai.com/v1/models
```

2. **檢查網路連線**：

```bash
ping api.openai.com
```

3. **使用代理伺服器**：

```bash
export https_proxy=http://proxy.example.com:8080
```

### Q7: BGE-M3 模型載入失敗

**症狀**：

```
OSError: Can't load tokenizer for 'BAAI/bge-m3'
```

**解決方案**：

1. **手動下載模型**：

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
```

2. **使用本地模型路徑**：

```yaml
models:
  chinese_embedding_model:
    type: bge_m3
    model: /path/to/local/bge-m3
```

3. **檢查磁碟空間**：

```bash
df -h ~/.cache/huggingface/
```

### Q8: GPU 記憶體不足

**症狀**：

```
CUDA out of memory
```

**解決方案**：

1. **降低批次大小**：

```yaml
models:
  chinese_embedding_model:
    batch_size: 8  # 從 32 降低到 8
```

2. **使用 CPU**：

```yaml
models:
  chinese_embedding_model:
    device: cpu
```

3. **清理 GPU 記憶體**：

```python
import torch
torch.cuda.empty_cache()
```

## 索引問題

### Q9: 文件讀取失敗

**症狀**：

```
FileNotFoundError: No such file or directory
```

**解決方案**：

1. **檢查檔案路徑**：

```bash
ls -la /path/to/documents/
```

2. **檢查檔案權限**：

```bash
chmod 644 /path/to/documents/*
```

3. **檢查檔案編碼**：

```bash
file -i /path/to/documents/file.txt
```

### Q10: 中文文件處理錯誤

**症狀**：

```
UnicodeDecodeError: 'utf-8' codec can't decode
```

**解決方案**：

1. **轉換檔案編碼**：

```bash
iconv -f gbk -t utf-8 input.txt > output.txt
```

2. **自動檢測編碼**：

```python
import chardet
with open('file.txt', 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']
```

3. **配置編碼檢測**：

```yaml
document_processing:
  auto_detect_encoding: true
  fallback_encoding: gbk
```

### Q11: 索引過程中斷

**症狀**：

```
IndexingError: Process interrupted
```

**解決方案**：

1. **檢查磁碟空間**：

```bash
df -h ./data/
```

2. **恢復索引**：

```bash
uv run chinese-graphrag index --resume --task-id idx_abc123
```

3. **增加記憶體限制**：

```yaml
performance:
  max_memory_usage: 16GB
```

## 查詢問題

### Q12: 查詢回應時間過長

**症狀**：查詢超過 30 秒沒有回應

**解決方案**：

1. **檢查索引狀態**：

```bash
uv run chinese-graphrag index-status
```

2. **優化查詢參數**：

```yaml
query:
  max_tokens: 1000  # 降低 token 數量
  timeout: 30       # 設定超時時間
```

3. **使用快取**：

```yaml
query:
  cache_enabled: true
  cache_ttl: 3600
```

### Q13: 查詢結果不準確

**症狀**：回答與問題不相關

**解決方案**：

1. **檢查索引品質**：

```bash
uv run chinese-graphrag validate-index
```

2. **調整搜尋參數**：

```yaml
query:
  similarity_threshold: 0.7  # 提高相似度閾值
  max_results: 10           # 增加搜尋結果數量
```

3. **重新索引**：

```bash
uv run chinese-graphrag index --rebuild
```

## 效能問題

### Q14: 記憶體使用過高

**症狀**：系統記憶體使用率超過 90%

**解決方案**：

1. **啟用記憶體優化**：

```yaml
performance:
  memory_optimization: true
  batch_processing: true
```

2. **調整批次大小**：

```yaml
models:
  chinese_embedding_model:
    batch_size: 16
```

3. **使用分頁處理**：

```yaml
indexing:
  chunk_processing: true
  max_chunks_per_batch: 100
```

### Q15: CPU 使用率過高

**症狀**：CPU 使用率持續 100%

**解決方案**：

1. **限制並行處理**：

```yaml
performance:
  max_workers: 2  # 降低工作執行緒數量
```

2. **使用 GPU 加速**：

```yaml
models:
  chinese_embedding_model:
    device: cuda
```

3. **調整處理優先級**：

```bash
nice -n 10 uv run chinese-graphrag index
```

## API 問題

### Q16: API 服務無法啟動

**症狀**：

```
OSError: [Errno 48] Address already in use
```

**解決方案**：

1. **檢查埠號佔用**：

```bash
lsof -i :8000
netstat -tulpn | grep 8000
```

2. **使用不同埠號**：

```bash
uv run chinese-graphrag api start --port 8080
```

3. **終止佔用程序**：

```bash
kill -9 $(lsof -t -i:8000)
```

### Q17: API 請求超時

**症狀**：

```
TimeoutError: Request timed out
```

**解決方案**：

1. **增加超時時間**：

```python
import requests
response = requests.post(url, json=data, timeout=60)
```

2. **使用異步請求**：

```python
import asyncio
import aiohttp

async def async_request():
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            return await response.json()
```

## 資料庫問題

### Q18: 向量資料庫連線失敗

**症狀**：

```
ConnectionError: Could not connect to vector database
```

**解決方案**：

1. **檢查資料庫檔案**：

```bash
ls -la ./data/lancedb/
```

2. **重建資料庫**：

```bash
uv run chinese-graphrag db-init --force
```

3. **檢查權限**：

```bash
chmod -R 755 ./data/
```

### Q19: 資料庫損壞

**症狀**：

```
DatabaseError: Database file is corrupted
```

**解決方案**：

1. **備份資料**：

```bash
cp -r ./data/ ./data_backup/
```

2. **修復資料庫**：

```bash
uv run chinese-graphrag db-repair
```

3. **從備份恢復**：

```bash
uv run chinese-graphrag db-restore --backup ./data_backup/
```

## 日誌分析

### 常見錯誤模式

1. **記憶體不足**：

```
MemoryError: Unable to allocate array
OutOfMemoryError: Java heap space
```

2. **網路連線問題**：

```
ConnectionError: HTTPSConnectionPool
TimeoutError: Request timed out
```

3. **API 限制**：

```
RateLimitError: Rate limit exceeded
QuotaExceededError: Quota exceeded
```

### 日誌級別說明

- `DEBUG`: 詳細的除錯資訊
- `INFO`: 一般資訊訊息
- `WARNING`: 警告訊息
- `ERROR`: 錯誤訊息
- `CRITICAL`: 嚴重錯誤

### 啟用詳細日誌

```bash
# 設定日誌級別
export GRAPHRAG_LOG_LEVEL=DEBUG

# 或在配置檔案中設定
logging:
  level: DEBUG
  file: logs/debug.log
```

## 效能調優建議

### 1. 硬體優化

- **記憶體**: 建議 16GB 以上
- **CPU**: 多核心處理器
- **儲存**: 使用 SSD
- **GPU**: NVIDIA GPU（可選）

### 2. 軟體配置

```yaml
performance:
  # 記憶體優化
  memory_optimization: true
  max_memory_usage: 12GB
  
  # 並行處理
  max_workers: 4
  batch_processing: true
  
  # 快取設定
  cache_enabled: true
  cache_size: 1000
  
  # GPU 加速
  gpu_acceleration: true
```

### 3. 監控指標

```bash
# 監控系統資源
htop
nvidia-smi  # GPU 監控

# 監控應用程式
uv run chinese-graphrag monitor
```

## 聯繫支援

如果以上解決方案都無法解決您的問題，請：

### 1. 收集診斷資訊

```bash
# 生成診斷報告
uv run chinese-graphrag generate-report

# 收集系統資訊
uv run chinese-graphrag system-info > system_info.txt
```

### 2. 提供詳細資訊

- 作業系統和版本
- Python 版本
- 錯誤訊息和堆疊追蹤
- 配置檔案（移除敏感資訊）
- 重現步驟

### 3. 提交問題

- **GitHub Issues**: <https://github.com/your-org/chinese-graphrag/issues>
- **電子郵件**: <support@example.com>
- **社群論壇**: <https://community.example.com>

### 4. 緊急支援

對於生產環境的緊急問題：

- **緊急熱線**: +1-xxx-xxx-xxxx
- **企業支援**: <enterprise@example.com>

## 預防措施

### 1. 定期維護

```bash
# 每週執行系統檢查
uv run chinese-graphrag health-check

# 清理快取和日誌
uv run chinese-graphrag cleanup

# 更新依賴套件
uv sync --upgrade
```

### 2. 備份策略

```bash
# 備份配置和資料
tar -czf backup_$(date +%Y%m%d).tar.gz config/ data/

# 自動備份腳本
crontab -e
# 0 2 * * * /path/to/backup_script.sh
```

### 3. 監控告警

```yaml
monitoring:
  alerts:
    memory_threshold: 80%
    cpu_threshold: 90%
    disk_threshold: 85%
    error_rate_threshold: 5%
```

這個故障排除指南涵蓋了大部分常見問題。如果您遇到未列出的問題，請參考系統日誌並聯繫技術支援。
