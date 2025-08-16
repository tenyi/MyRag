# 中文 GraphRAG 系統生產部署指南

## 概述

本指南提供中文 GraphRAG 系統在生產環境中的完整部署流程，包括系統需求、安裝步驟、配置設定、監控維護等內容。

## 系統需求

### 硬體需求

#### 最低配置
- **CPU**: 4 核心 2.0GHz 以上
- **記憶體**: 8GB RAM
- **儲存空間**: 50GB 可用空間
- **網路**: 穩定的網際網路連線

#### 建議配置
- **CPU**: 8 核心 3.0GHz 以上
- **記憶體**: 16GB RAM
- **儲存空間**: 200GB SSD
- **網路**: 高速網際網路連線
- **GPU**: NVIDIA GPU（可選，用於加速 Embedding）

#### 生產環境配置
- **CPU**: 16 核心 3.5GHz 以上
- **記憶體**: 32GB RAM
- **儲存空間**: 500GB NVMe SSD
- **網路**: 企業級網路連線
- **GPU**: NVIDIA RTX 4090 或同等級（推薦）

### 軟體需求

#### 作業系統
- **Linux**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **macOS**: 12.0+ (開發環境)
- **Windows**: Windows Server 2019+ (不推薦生產環境)

#### 必要軟體
- **Python**: 3.11 或 3.12
- **uv**: 最新版本套件管理器
- **Git**: 版本控制
- **curl**: 網路工具
- **systemd**: 服務管理 (Linux)

#### 可選軟體
- **Docker**: 容器化部署
- **Nginx**: 反向代理
- **Redis**: 快取服務
- **PostgreSQL**: 關聯式資料庫

## 部署前準備

### 1. 環境檢查

```bash
# 檢查 Python 版本
python3 --version

# 檢查 uv 安裝
uv --version

# 檢查系統資源
free -h
df -h
nproc
```

### 2. 建立部署用戶

```bash
# 建立專用用戶
sudo useradd -m -s /bin/bash graphrag
sudo usermod -aG sudo graphrag

# 切換到部署用戶
sudo su - graphrag
```

### 3. 準備部署目錄

```bash
# 建立應用目錄
mkdir -p ~/chinese-graphrag
cd ~/chinese-graphrag

# 下載應用程式碼
git clone <repository-url> .

# 或者上傳應用程式檔案
# scp -r ./chinese-graphrag user@server:~/
```

## 自動化部署

### 使用部署腳本

```bash
# 執行自動化部署
uv run python scripts/deploy_production.py \
    --config deployment_config.yaml \
    --deployment-dir /opt/chinese-graphrag \
    --backup-dir /opt/backups

# 檢查部署狀態
uv run python scripts/deploy_production.py --dry-run
```

### 部署配置

編輯 `deployment_config.yaml` 檔案：

```yaml
# 基本設定
deployment_dir: "/opt/chinese-graphrag"
backup_dir: "/opt/backups"

# 系統需求
required_disk_space_gb: 50
required_memory_gb: 8

# 生產配置
production_config:
  workers: 4
  api:
    host: "0.0.0.0"
    port: 8000
  logging:
    level: "INFO"
  monitoring:
    enabled: true
```

## 手動部署步驟

### 1. 建立部署結構

```bash
# 建立目錄結構
sudo mkdir -p /opt/chinese-graphrag/{app,config,data,logs,scripts,backups}
sudo chown -R graphrag:graphrag /opt/chinese-graphrag
```

### 2. 複製應用程式檔案

```bash
# 複製源碼
cp -r src/ /opt/chinese-graphrag/app/
cp -r config/ /opt/chinese-graphrag/config/
cp pyproject.toml /opt/chinese-graphrag/app/
cp main.py /opt/chinese-graphrag/app/
```

### 3. 安裝依賴

```bash
cd /opt/chinese-graphrag/app
uv sync --frozen
```

### 4. 配置環境

```bash
# 複製環境變數範本
cp config/.env.example /opt/chinese-graphrag/config/.env.production

# 編輯環境變數
nano /opt/chinese-graphrag/config/.env.production
```

環境變數設定：

```bash
# 基本設定
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API 設定
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# 資料庫設定
DATA_DIR=/opt/chinese-graphrag/data
VECTOR_DB_PATH=/opt/chinese-graphrag/data/vector_db
GRAPH_DB_PATH=/opt/chinese-graphrag/data/graph_db

# 日誌設定
LOG_DIR=/opt/chinese-graphrag/logs

# OpenAI API 設定（必要）
OPENAI_API_KEY=your_openai_api_key_here

# Azure OpenAI 設定（可選）
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_api_key
```

### 5. 建立服務腳本

建立 systemd 服務檔案：

```bash
sudo nano /etc/systemd/system/chinese-graphrag.service
```

服務配置：

```ini
[Unit]
Description=Chinese GraphRAG System
After=network.target

[Service]
Type=exec
User=graphrag
Group=graphrag
WorkingDirectory=/opt/chinese-graphrag/app
Environment=PATH=/opt/chinese-graphrag/app/.venv/bin
EnvironmentFile=/opt/chinese-graphrag/config/.env.production
ExecStart=/opt/chinese-graphrag/app/.venv/bin/uvicorn src.chinese_graphrag.api.app:app --host 0.0.0.0 --port 8000 --workers 4
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=chinese-graphrag

[Install]
WantedBy=multi-user.target
```

### 6. 啟動服務

```bash
# 重新載入 systemd
sudo systemctl daemon-reload

# 啟用服務
sudo systemctl enable chinese-graphrag

# 啟動服務
sudo systemctl start chinese-graphrag

# 檢查狀態
sudo systemctl status chinese-graphrag
```

## 配置設定

### 應用程式配置

編輯 `/opt/chinese-graphrag/config/production.yaml`：

```yaml
# 應用程式設定
environment: production
debug: false

# API 設定
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_enabled: false
  rate_limiting: true

# 資料庫設定
database:
  vector_db:
    type: "lancedb"
    path: "/opt/chinese-graphrag/data/vector_db"
  graph_db:
    path: "/opt/chinese-graphrag/data/graph_db"

# Embedding 設定
embedding:
  model: "BAAI/bge-m3"
  device: "auto"
  batch_size: 32
  cache_enabled: true

# LLM 設定
llm:
  provider: "openai"
  model: "gpt-5-mini"
  #temperature: 0.7
  #max_tokens: 2000

# 中文處理設定
chinese:
  enable_traditional: true
  enable_simplified: true

# 監控設定
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_port: 8080
```

### 日誌配置

建立日誌配置檔案 `/opt/chinese-graphrag/config/logging.yaml`：

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /opt/chinese-graphrag/logs/app.log
    maxBytes: 104857600  # 100MB
    backupCount: 10

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: json
    filename: /opt/chinese-graphrag/logs/error.log
    maxBytes: 104857600  # 100MB
    backupCount: 5

loggers:
  chinese_graphrag:
    level: INFO
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

## 反向代理設定

### Nginx 配置

安裝 Nginx：

```bash
sudo apt update
sudo apt install nginx
```

建立 Nginx 配置：

```bash
sudo nano /etc/nginx/sites-available/chinese-graphrag
```

配置內容：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL 配置
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # 安全標頭
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # 上傳大小限制
    client_max_body_size 10M;

    # API 代理
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超時設定
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }

    # 健康檢查
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
        access_log off;
    }

    # 監控指標
    location /metrics {
        proxy_pass http://127.0.0.1:9090/metrics;
        allow 127.0.0.1;
        deny all;
    }

    # 靜態檔案
    location /static/ {
        alias /opt/chinese-graphrag/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

啟用配置：

```bash
sudo ln -s /etc/nginx/sites-available/chinese-graphrag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 監控設定

### 系統監控

啟動監控服務：

```bash
# 建立監控腳本
cp scripts/monitoring_tools.py /opt/chinese-graphrag/scripts/

# 建立監控服務
sudo nano /etc/systemd/system/chinese-graphrag-monitor.service
```

監控服務配置：

```ini
[Unit]
Description=Chinese GraphRAG Monitoring
After=chinese-graphrag.service

[Service]
Type=exec
User=graphrag
Group=graphrag
WorkingDirectory=/opt/chinese-graphrag
ExecStart=/opt/chinese-graphrag/app/.venv/bin/python scripts/monitoring_tools.py monitor
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

啟動監控：

```bash
sudo systemctl enable chinese-graphrag-monitor
sudo systemctl start chinese-graphrag-monitor
```

### 日誌監控

設定 logrotate：

```bash
sudo nano /etc/logrotate.d/chinese-graphrag
```

配置內容：

```
/opt/chinese-graphrag/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 graphrag graphrag
    postrotate
        systemctl reload chinese-graphrag
    endscript
}
```

## 備份設定

### 自動備份

建立備份腳本：

```bash
cp scripts/backup_recovery.py /opt/chinese-graphrag/scripts/

# 建立備份配置
nano /opt/chinese-graphrag/config/backup.yaml
```

備份配置：

```yaml
backup_dir: "/opt/backups"
data_dir: "/opt/chinese-graphrag/data"
config_dir: "/opt/chinese-graphrag/config"

retention:
  daily: 7
  weekly: 4
  monthly: 12

compression: true
verify_backup: true
```

設定 cron 任務：

```bash
crontab -e
```

添加備份任務：

```cron
# 每日備份（凌晨 2 點）
0 2 * * * /opt/chinese-graphrag/app/.venv/bin/python /opt/chinese-graphrag/scripts/backup_recovery.py backup --type daily

# 週備份（週日凌晨 3 點）
0 3 * * 0 /opt/chinese-graphrag/app/.venv/bin/python /opt/chinese-graphrag/scripts/backup_recovery.py backup --type weekly

# 月備份（每月 1 號凌晨 4 點）
0 4 1 * * /opt/chinese-graphrag/app/.venv/bin/python /opt/chinese-graphrag/scripts/backup_recovery.py backup --type monthly

# 清理舊備份（每週一凌晨 5 點）
0 5 * * 1 /opt/chinese-graphrag/app/.venv/bin/python /opt/chinese-graphrag/scripts/backup_recovery.py cleanup
```

## 安全設定

### 防火牆配置

```bash
# 安裝 ufw
sudo apt install ufw

# 預設規則
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 允許 SSH
sudo ufw allow ssh

# 允許 HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# 允許應用程式埠（僅限本地）
sudo ufw allow from 127.0.0.1 to any port 8000
sudo ufw allow from 127.0.0.1 to any port 8080
sudo ufw allow from 127.0.0.1 to any port 9090

# 啟用防火牆
sudo ufw enable
```

### API 安全

設定 API 金鑰：

```bash
# 生成 API 金鑰
openssl rand -hex 32

# 添加到環境變數
echo "API_KEY=your_generated_api_key" >> /opt/chinese-graphrag/config/.env.production
```

### 檔案權限

```bash
# 設定適當的檔案權限
sudo chown -R graphrag:graphrag /opt/chinese-graphrag
sudo chmod -R 755 /opt/chinese-graphrag
sudo chmod 600 /opt/chinese-graphrag/config/.env.production
sudo chmod 600 /opt/chinese-graphrag/config/backup.yaml
```

## 效能調優

### 系統調優

編輯系統限制：

```bash
sudo nano /etc/security/limits.conf
```

添加配置：

```
graphrag soft nofile 65536
graphrag hard nofile 65536
graphrag soft nproc 32768
graphrag hard nproc 32768
```

### 應用程式調優

調整 uvicorn 設定：

```bash
# 在環境變數中設定
echo "UVICORN_WORKERS=4" >> /opt/chinese-graphrag/config/.env.production
echo "UVICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker" >> /opt/chinese-graphrag/config/.env.production
echo "UVICORN_MAX_REQUESTS=1000" >> /opt/chinese-graphrag/config/.env.production
echo "UVICORN_MAX_REQUESTS_JITTER=100" >> /opt/chinese-graphrag/config/.env.production
```

### 資料庫調優

LanceDB 調優：

```yaml
# 在 production.yaml 中添加
database:
  vector_db:
    type: "lancedb"
    path: "/opt/chinese-graphrag/data/vector_db"
    cache_size: 1000
    write_batch_size: 1000
    index_cache_size: 500
```

## 部署驗證

### 健康檢查

```bash
# 檢查服務狀態
sudo systemctl status chinese-graphrag

# 檢查 API 健康
curl http://localhost:8080/health

# 檢查監控指標
curl http://localhost:9090/metrics

# 執行完整健康檢查
uv run python scripts/monitoring_tools.py health
```

### 功能測試

```bash
# 測試 API 端點
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{"query": "什麼是人工智慧？"}'

# 測試索引功能
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{"documents": ["測試文件內容"]}'
```

### 效能測試

```bash
# 使用 ab 進行壓力測試
ab -n 100 -c 10 http://localhost:8000/health

# 使用自訂效能測試
uv run python scripts/performance_stress_test.py \
  --test-dir ./performance_results \
  --duration 300
```

## 維護操作

### 日常維護

```bash
# 檢查系統狀態
sudo systemctl status chinese-graphrag
sudo systemctl status chinese-graphrag-monitor

# 檢查日誌
sudo journalctl -u chinese-graphrag -f
tail -f /opt/chinese-graphrag/logs/app.log

# 檢查資源使用
htop
df -h
free -h
```

### 更新部署

```bash
# 停止服務
sudo systemctl stop chinese-graphrag

# 備份當前版本
uv run python /opt/chinese-graphrag/scripts/backup_recovery.py backup --type update

# 更新程式碼
cd /opt/chinese-graphrag/app
git pull origin main

# 更新依賴
uv sync

# 重啟服務
sudo systemctl start chinese-graphrag

# 驗證更新
curl http://localhost:8080/health
```

### 故障排除

常見問題和解決方案：

1. **服務無法啟動**
   ```bash
   # 檢查日誌
   sudo journalctl -u chinese-graphrag -n 50
   
   # 檢查配置
   uv run python -c "import yaml; yaml.safe_load(open('/opt/chinese-graphrag/config/production.yaml'))"
   ```

2. **記憶體不足**
   ```bash
   # 檢查記憶體使用
   free -h
   ps aux --sort=-%mem | head
   
   # 調整 worker 數量
   nano /opt/chinese-graphrag/config/.env.production
   # 修改 API_WORKERS=2
   ```

3. **磁碟空間不足**
   ```bash
   # 清理日誌
   uv run python /opt/chinese-graphrag/scripts/monitoring_tools.py cleanup --log-days 7
   
   # 清理備份
   uv run python /opt/chinese-graphrag/scripts/backup_recovery.py cleanup
   ```

## 災難恢復

### 完整系統恢復

```bash
# 1. 安裝基礎環境
# （按照部署步驟 1-3）

# 2. 恢復最新備份
uv run python /opt/chinese-graphrag/scripts/backup_recovery.py list
uv run python /opt/chinese-graphrag/scripts/backup_recovery.py restore <backup_name>

# 3. 重啟服務
sudo systemctl start chinese-graphrag
sudo systemctl start chinese-graphrag-monitor

# 4. 驗證恢復
curl http://localhost:8080/health
```

### 資料恢復

```bash
# 僅恢復資料
uv run python /opt/chinese-graphrag/scripts/backup_recovery.py restore <backup_name> --target /opt/chinese-graphrag/data

# 重建索引（如果需要）
curl -X POST http://localhost:8000/api/rebuild-index \
  -H "Authorization: Bearer your_api_key"
```

## 最佳實踐

### 部署檢查清單

- [ ] 系統需求檢查完成
- [ ] 安全設定配置完成
- [ ] 監控系統運行正常
- [ ] 備份機制設定完成
- [ ] 日誌輪轉配置完成
- [ ] 防火牆規則設定完成
- [ ] SSL 憑證配置完成
- [ ] API 金鑰設定完成
- [ ] 效能調優完成
- [ ] 健康檢查通過
- [ ] 功能測試通過
- [ ] 文件更新完成

### 監控指標

重要監控指標：

- **系統指標**: CPU、記憶體、磁碟使用率
- **應用指標**: 回應時間、錯誤率、吞吐量
- **業務指標**: 查詢成功率、索引文件數、用戶活躍度

### 警報設定

建議設定的警報：

- CPU 使用率 > 80%
- 記憶體使用率 > 85%
- 磁碟使用率 > 90%
- API 回應時間 > 5 秒
- 錯誤率 > 5%
- 服務不可用

## 結論

本指南提供了中文 GraphRAG 系統的完整生產部署流程。遵循這些步驟和最佳實踐，可以確保系統在生產環境中穩定、安全、高效地運行。

定期檢查和維護是保持系統健康運行的關鍵。建議建立定期的維護計劃，包括系統更新、備份驗證、效能監控等。

如有問題，請參考故障排除章節或聯繫技術支援團隊。