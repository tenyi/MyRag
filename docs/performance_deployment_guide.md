# 效能優化系統部署指南

本指南將協助您在不同環境中部署和配置 Chinese GraphRAG 效能優化系統。

## 目錄

- [系統需求](#系統需求)
- [安裝步驟](#安裝步驟)
- [環境配置](#環境配置)
- [部署模式](#部署模式)
- [監控設定](#監控設定)
- [故障排除](#故障排除)
- [效能調優](#效能調優)
- [維護指南](#維護指南)

## 系統需求

### 最低需求

- **CPU**: 4 核心
- **記憶體**: 8GB RAM
- **磁碟空間**: 50GB 可用空間
- **Python**: 3.8 或更高版本
- **作業系統**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10+

### 建議需求

- **CPU**: 8 核心或更多
- **記憶體**: 16GB RAM 或更多
- **磁碟空間**: 200GB SSD
- **GPU**: NVIDIA GPU (可選，用於加速)
- **網路**: 穩定的網際網路連線

### 依賴服務

- **Redis** (可選): 用於分散式快取
- **PostgreSQL/MySQL** (可選): 用於持久化儲存
- **Prometheus** (可選): 用於監控
- **Grafana** (可選): 用於視覺化

## 安裝步驟

### 1. 環境準備

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 升級 pip
pip install --upgrade pip
```

### 2. 安裝依賴

```bash
# 安裝基本依賴
pip install -r requirements.txt

# 安裝效能優化相關依賴
pip install psutil GPUtil redis pyyaml

# 安裝可選依賴
pip install prometheus-client grafana-api
```

### 3. 配置檔案設定

```bash
# 複製配置範本
cp config/performance_optimization.yaml.example config/performance_optimization.yaml

# 編輯配置檔案
nano config/performance_optimization.yaml
```

### 4. 初始化系統

```python
# 初始化腳本 (init_performance.py)
from src.chinese_graphrag.performance import load_performance_config, OptimizerManager

async def initialize_system():
    # 載入配置
    config = load_performance_config()
    
    # 初始化優化管理器
    manager = OptimizerManager(config.optimization)
    await manager.initialize()
    
    print("效能優化系統初始化完成")
    return manager

# 執行初始化
import asyncio
asyncio.run(initialize_system())
```

## 環境配置

### 開發環境

```yaml
# config/performance_optimization.yaml
environments:
  development:
    batch_optimization:
      enabled: true
      default_batch_size: 8
      parallel_workers: 2
      memory_threshold_mb: 512.0
    
    query_optimization:
      cache_enabled: true
      cache_backend: "memory"
      cache_ttl_seconds: 300
      cache_max_size: 1000
    
    cost_optimization:
      tracking_enabled: true
      budget_limit_usd: 10.0
      strict_budget_enforcement: false
    
    performance_monitoring:
      enabled: true
      collection_interval: 10.0
      alert_thresholds:
        cpu_usage: 90.0
        memory_usage: 90.0
```

### 測試環境

```yaml
environments:
  staging:
    batch_optimization:
      enabled: true
      default_batch_size: 16
      parallel_workers: 2
      memory_threshold_mb: 1024.0
    
    query_optimization:
      cache_enabled: true
      cache_backend: "redis"
      redis:
        url: "redis://staging-redis:6379"
      cache_ttl_seconds: 1800
      cache_max_size: 5000
    
    cost_optimization:
      tracking_enabled: true
      budget_limit_usd: 50.0
      strict_budget_enforcement: true
    
    performance_monitoring:
      enabled: true
      collection_interval: 5.0
```

### 生產環境

```yaml
environments:
  production:
    batch_optimization:
      enabled: true
      default_batch_size: 32
      parallel_workers: 4
      memory_threshold_mb: 2048.0
      enable_adaptive_sizing: true
    
    query_optimization:
      cache_enabled: true
      cache_backend: "hybrid"
      redis:
        url: "redis://prod-redis-cluster:6379"
        connection_pool_size: 20
      cache_ttl_seconds: 3600
      cache_max_size: 20000
      enable_preloading: true
    
    cost_optimization:
      tracking_enabled: true
      budget_limit_usd: 500.0
      strict_budget_enforcement: true
      alert_threshold: 0.8
    
    performance_monitoring:
      enabled: true
      collection_interval: 5.0
      enable_gpu_monitoring: true
```

## 部署模式

### 1. 單機部署

適用於小型應用或開發環境。

```python
# single_node_deployment.py
import asyncio
from src.chinese_graphrag.performance import OptimizerManager, load_performance_config

async def deploy_single_node():
    # 載入配置
    config = load_performance_config(environment="production")
    
    # 建立優化管理器
    async with OptimizerManager(config.optimization) as manager:
        print("單機部署完成，系統已啟動")
        
        # 保持系統運行
        try:
            while True:
                await asyncio.sleep(60)
                status = manager.get_performance_status()
                print(f"系統狀態: {status['running']}")
        except KeyboardInterrupt:
            print("系統正在關閉...")

if __name__ == "__main__":
    asyncio.run(deploy_single_node())
```

### 2. 分散式部署

適用於大型應用或高可用性需求。

#### 協調器節點

```python
# coordinator_node.py
import asyncio
from src.chinese_graphrag.performance import OptimizerManager, load_performance_config

async def deploy_coordinator():
    config = load_performance_config(environment="production")
    
    # 設定為協調器模式
    config.distributed = {
        "enabled": True,
        "mode": "coordinator",
        "coordinator": {
            "redis_url": "redis://coordinator:6379",
            "heartbeat_interval": 30.0,
            "worker_timeout": 120.0
        }
    }
    
    async with OptimizerManager(config.optimization) as manager:
        print("協調器節點已啟動")
        
        # 監控工作節點
        while True:
            await asyncio.sleep(30)
            # 檢查工作節點狀態
            # 實作工作節點健康檢查邏輯

if __name__ == "__main__":
    asyncio.run(deploy_coordinator())
```

#### 工作節點

```python
# worker_node.py
import asyncio
import os
from src.chinese_graphrag.performance import OptimizerManager, load_performance_config

async def deploy_worker():
    config = load_performance_config(environment="production")
    
    # 設定為工作節點模式
    node_id = os.getenv("NODE_ID", "worker-1")
    config.distributed = {
        "enabled": True,
        "mode": "worker",
        "worker": {
            "node_id": node_id,
            "coordinator_url": "redis://coordinator:6379",
            "capabilities": ["embedding", "query", "inference"],
            "max_concurrent_tasks": 10
        }
    }
    
    async with OptimizerManager(config.optimization) as manager:
        print(f"工作節點 {node_id} 已啟動")
        
        # 註冊到協調器
        # 實作工作節點註冊邏輯
        
        # 處理任務
        while True:
            await asyncio.sleep(10)
            # 從協調器取得任務並處理

if __name__ == "__main__":
    asyncio.run(deploy_worker())
```

### 3. Docker 部署

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 複製需求檔案
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY src/ src/
COPY config/ config/
COPY examples/ examples/

# 設定環境變數
ENV PYTHONPATH=/app
ENV PERF_ENVIRONMENT=production

# 暴露埠號
EXPOSE 8000

# 啟動命令
CMD ["python", "examples/performance_optimization_example.py"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  # Redis 快取
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  # 協調器節點
  coordinator:
    build: .
    environment:
      - PERF_ENVIRONMENT=production
      - PERF_REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    command: python coordinator_node.py

  # 工作節點
  worker-1:
    build: .
    environment:
      - PERF_ENVIRONMENT=production
      - NODE_ID=worker-1
      - PERF_REDIS_URL=redis://redis:6379
    depends_on:
      - coordinator
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    command: python worker_node.py

  worker-2:
    build: .
    environment:
      - PERF_ENVIRONMENT=production
      - NODE_ID=worker-2
      - PERF_REDIS_URL=redis://redis:6379
    depends_on:
      - coordinator
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    command: python worker_node.py

  # Prometheus 監控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  # Grafana 儀表板
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 4. Kubernetes 部署

#### deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: performance-optimizer
  labels:
    app: performance-optimizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: performance-optimizer
  template:
    metadata:
      labels:
        app: performance-optimizer
    spec:
      containers:
      - name: optimizer
        image: chinese-graphrag/performance-optimizer:latest
        ports:
        - containerPort: 8000
        env:
        - name: PERF_ENVIRONMENT
          value: "production"
        - name: PERF_REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: performance-config
      - name: logs-volume
        persistentVolumeClaim:
          claimName: logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: performance-optimizer-service
spec:
  selector:
    app: performance-optimizer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 監控設定

### 1. Prometheus 配置

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'performance-optimizer'
    static_configs:
      - targets: ['coordinator:8000', 'worker-1:8000', 'worker-2:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Grafana 儀表板

```json
{
  "dashboard": {
    "title": "Chinese GraphRAG Performance",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU Usage"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "memory_usage_mb",
            "legendFormat": "Memory Usage"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "cache_hit_rate",
            "legendFormat": "Hit Rate"
          }
        ]
      },
      {
        "title": "Batch Processing Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "batch_throughput_ops_per_sec",
            "legendFormat": "Throughput"
          }
        ]
      }
    ]
  }
}
```

### 3. 警報規則

```yaml
# monitoring/alert_rules.yml
groups:
  - name: performance_alerts
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes"

      - alert: LowCacheHitRate
        expr: cache_hit_rate < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is below 70% for more than 10 minutes"

      - alert: BudgetExceeded
        expr: cost_budget_usage_percent > 90
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Budget limit exceeded"
          description: "Cost budget usage is above 90%"
```

## 故障排除

### 常見問題

#### 1. 系統啟動失敗

**症狀**: 系統無法正常啟動

**檢查步驟**:
```bash
# 檢查配置檔案
python -c "from src.chinese_graphrag.performance import load_performance_config; print('Config OK')"

# 檢查依賴
pip list | grep -E "(psutil|redis|yaml)"

# 檢查日誌
tail -f logs/performance/performance.log
```

**常見解決方案**:
- 檢查配置檔案語法
- 確認所有依賴已安裝
- 檢查檔案權限
- 確認 Redis 連線

#### 2. 效能下降

**症狀**: 系統效能明顯下降

**診斷工具**:
```python
# 效能診斷腳本
from src.chinese_graphrag.performance import OptimizerManager, load_performance_config

async def diagnose_performance():
    config = load_performance_config()
    manager = OptimizerManager(config.optimization)
    
    # 取得效能報告
    report = manager.get_optimization_report(60)
    print(f"效能報告: {report}")
    
    # 檢查資源使用
    if manager.performance_monitor:
        metrics = manager.performance_monitor.get_current_metrics()
        print(f"當前指標: {metrics}")

# 執行診斷
import asyncio
asyncio.run(diagnose_performance())
```

#### 3. 記憶體洩漏

**症狀**: 記憶體使用持續增長

**檢查方法**:
```python
# 記憶體分析
import gc
import psutil
import tracemalloc

# 啟用記憶體追蹤
tracemalloc.start()

# 執行一段時間後檢查
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

# 檢查垃圾回收
print(f"Garbage collection counts: {gc.get_count()}")

# 系統記憶體使用
process = psutil.Process()
print(f"Process memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

## 效能調優

### 1. 批次大小優化

```python
# 自動批次大小調優
async def optimize_batch_size():
    optimizer = BatchOptimizer()
    
    # 測試不同批次大小
    batch_sizes = [8, 16, 32, 64, 128]
    results = {}
    
    for size in batch_sizes:
        optimizer.default_batch_size = size
        
        # 執行基準測試
        start_time = time.time()
        await optimizer.process_batch(test_data, test_function)
        end_time = time.time()
        
        results[size] = end_time - start_time
    
    # 選擇最佳批次大小
    optimal_size = min(results, key=results.get)
    print(f"最佳批次大小: {optimal_size}")
    
    return optimal_size
```

### 2. 快取策略優化

```python
# 快取效能分析
def analyze_cache_performance():
    optimizer = QueryOptimizer()
    
    # 取得快取統計
    stats = optimizer.get_cache_stats()
    
    print(f"快取命中率: {stats['hit_rate']:.2%}")
    print(f"快取大小: {stats['cache_size']}")
    print(f"平均查詢時間: {stats['avg_query_time']:.3f}s")
    
    # 建議優化策略
    if stats['hit_rate'] < 0.7:
        print("建議: 增加快取 TTL 或快取大小")
    
    if stats['avg_query_time'] > 0.1:
        print("建議: 啟用預載或優化快取鍵設計")
```

### 3. 成本優化

```python
# 成本分析和優化
def optimize_costs():
    optimizer = CostOptimizer()
    
    # 分析使用模式
    usage_stats = optimizer.get_usage_stats(24 * 60)  # 24小時
    
    # 識別高成本操作
    high_cost_operations = [
        op for op in usage_stats['operations'] 
        if op['cost'] > usage_stats['avg_cost'] * 2
    ]
    
    print(f"高成本操作: {len(high_cost_operations)}")
    
    # 建議替代模型
    for op in high_cost_operations:
        recommendation = optimizer.get_model_recommendation(
            operation_type=op['type'],
            priority="cost"
        )
        print(f"操作 {op['type']} 建議使用: {recommendation['recommended_model']}")
```

## 維護指南

### 1. 定期維護任務

```bash
#!/bin/bash
# maintenance.sh - 定期維護腳本

# 清理舊日誌
find logs/ -name "*.log" -mtime +30 -delete

# 清理快取
redis-cli FLUSHDB

# 備份配置
cp config/performance_optimization.yaml backups/config_$(date +%Y%m%d).yaml

# 更新統計資料
python scripts/update_performance_stats.py

# 檢查系統健康
python scripts/health_check.py
```

### 2. 監控檢查清單

- [ ] CPU 使用率 < 80%
- [ ] 記憶體使用率 < 85%
- [ ] 磁碟使用率 < 90%
- [ ] 快取命中率 > 70%
- [ ] 錯誤率 < 5%
- [ ] 回應時間 < 5 秒
- [ ] 預算使用率 < 90%

### 3. 升級程序

```bash
# 升級步驟
# 1. 備份當前配置
cp -r config/ backups/config_backup_$(date +%Y%m%d)/

# 2. 停止服務
docker-compose down

# 3. 更新程式碼
git pull origin main

# 4. 更新依賴
pip install -r requirements.txt --upgrade

# 5. 遷移配置（如需要）
python scripts/migrate_config.py

# 6. 重新啟動服務
docker-compose up -d

# 7. 驗證升級
python scripts/verify_upgrade.py
```

### 4. 備份策略

```python
# backup_manager.py
import shutil
import datetime
from pathlib import Path

class BackupManager:
    def __init__(self, backup_path="backups"):
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(exist_ok=True)
    
    def backup_config(self):
        """備份配置檔案"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_path / f"config_{timestamp}"
        shutil.copytree("config", backup_dir)
        print(f"配置已備份至: {backup_dir}")
    
    def backup_logs(self):
        """備份日誌檔案"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_path / f"logs_{timestamp}"
        shutil.copytree("logs", backup_dir)
        print(f"日誌已備份至: {backup_dir}")
    
    def cleanup_old_backups(self, days=30):
        """清理舊備份"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        for backup_dir in self.backup_path.iterdir():
            if backup_dir.is_dir():
                dir_time = datetime.datetime.fromtimestamp(backup_dir.stat().st_mtime)
                if dir_time < cutoff_date:
                    shutil.rmtree(backup_dir)
                    print(f"已刪除舊備份: {backup_dir}")

# 使用範例
backup_manager = BackupManager()
backup_manager.backup_config()
backup_manager.backup_logs()
backup_manager.cleanup_old_backups()
```

## 總結

本部署指南涵蓋了 Chinese GraphRAG 效能優化系統的完整部署流程，包括：

1. **系統需求和安裝**: 確保環境符合要求
2. **環境配置**: 針對不同環境進行適當配置
3. **部署模式**: 支援單機、分散式、容器化部署
4. **監控設定**: 建立完整的監控和警報系統
5. **故障排除**: 提供常見問題的解決方案
6. **效能調優**: 指導如何優化系統效能
7. **維護指南**: 確保系統長期穩定運行

遵循本指南，您可以成功部署和維護一個高效能的 Chinese GraphRAG 系統。