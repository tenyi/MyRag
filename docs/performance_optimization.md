# 效能優化系統

Chinese GraphRAG 系統的效能優化模組提供全面的效能提升解決方案，包括批次處理優化、查詢快取、成本控制、效能監控等功能。

## 目錄

- [概述](#概述)
- [核心功能](#核心功能)
- [快速開始](#快速開始)
- [詳細配置](#詳細配置)
- [使用指南](#使用指南)
- [效能監控](#效能監控)
- [基準測試](#基準測試)
- [最佳實踐](#最佳實踐)
- [故障排除](#故障排除)

## 概述

效能優化系統旨在提升 Chinese GraphRAG 系統的整體效能，降低運行成本，並提供詳細的效能監控和分析功能。

### 主要特性

- **批次處理優化**: 智慧批次大小調整、並行處理、記憶體管理
- **查詢快取系統**: 多層快取、智慧預載、快取失效策略
- **成本優化**: 模型使用追蹤、成本控制、智慧模型選擇
- **效能監控**: 即時監控、歷史分析、警報系統
- **基準測試**: 自動化測試、效能比較、報告生成

## 核心功能

### 1. 批次處理優化 (BatchOptimizer)

批次處理優化器提供智慧的批次處理功能，自動調整批次大小和並行度以達到最佳效能。

#### 主要功能

- 動態批次大小調整
- 記憶體感知處理
- 並行工作者管理
- 錯誤處理和重試
- 效能統計追蹤

#### 使用範例

```python
from src.chinese_graphrag.performance import BatchOptimizer

# 初始化批次優化器
optimizer = BatchOptimizer(
    default_batch_size=32,
    max_batch_size=128,
    parallel_workers=4,
    memory_threshold_mb=1024.0
)

# 處理批次資料
async def process_texts(texts):
    return await optimizer.process_batch(
        items=texts,
        process_func=your_embedding_function
    )
```

### 2. 查詢優化 (QueryOptimizer)

查詢優化器提供多層快取系統，大幅提升重複查詢的效能。

#### 主要功能

- 記憶體快取 (LRU)
- Redis 分散式快取
- 語義快取
- 智慧預載
- 快取統計分析

#### 使用範例

```python
from src.chinese_graphrag.performance import QueryOptimizer

# 初始化查詢優化器
optimizer = QueryOptimizer(
    cache_ttl=3600,  # 1小時
    max_cache_size=10000,
    enable_preloading=True
)

# 優化查詢
async def optimized_query(query):
    # 嘗試從快取取得結果
    cached_result = await optimizer.get_cached_result(query)
    if cached_result:
        return cached_result
    
    # 執行查詢並快取結果
    result = await your_query_function(query)
    await optimizer.cache_result(query, result)
    return result
```

### 3. 成本優化 (CostOptimizer)

成本優化器追蹤模型使用情況，提供成本控制和智慧模型選擇建議。

#### 主要功能

- 使用情況追蹤
- 成本計算和預算控制
- 品質評估
- 智慧模型選擇
- 成本分析報告

#### 使用範例

```python
from src.chinese_graphrag.performance import CostOptimizer

# 初始化成本優化器
optimizer = CostOptimizer(
    budget_limit=100.0,  # $100 預算
    quality_threshold=0.8,
    storage_path="logs/cost_tracking.json"
)

# 追蹤模型使用
await optimizer.track_usage(
    model_name="gpt-5-mini",
    input_tokens=1000,
    output_tokens=500,
    operation_type="text_generation"
)

# 取得模型建議
recommendation = await optimizer.get_model_recommendation(
    operation_type="embedding",
    expected_tokens=2000
)
```

### 4. 效能監控 (PerformanceMonitor)

效能監控器提供即時的系統效能監控和歷史分析功能。

#### 主要功能

- 系統資源監控 (CPU, 記憶體, 磁碟, GPU)
- 應用程式效能指標
- 自訂指標收集
- 警報系統
- 歷史資料分析

#### 使用範例

```python
from src.chinese_graphrag.performance import PerformanceMonitor

# 初始化效能監控器
monitor = PerformanceMonitor(
    collection_interval=5.0,  # 5秒收集一次
    history_size=1000,
    storage_path="logs/performance"
)

# 啟動監控
monitor.start_monitoring()

# 設定警報
monitor.set_alert_threshold(
    "cpu_usage", 80.0, 
    lambda metric, value, threshold: print(f"CPU 使用率過高: {value}%")
)

# 取得當前指標
current_metrics = monitor.get_current_metrics()
```

## 快速開始

### 1. 安裝依賴

```bash
pip install psutil GPUtil redis
```

### 2. 基本使用

```python
import asyncio
from src.chinese_graphrag.performance import OptimizerManager, OptimizationConfig

async def main():
    # 建立配置
    config = OptimizationConfig(
        batch_enabled=True,
        query_cache_enabled=True,
        cost_tracking_enabled=True,
        monitoring_enabled=True
    )
    
    # 初始化管理器
    async with OptimizerManager(config) as manager:
        # 批次處理優化
        results = await manager.optimize_batch_processing(
            items=["文本1", "文本2", "文本3"],
            process_func=your_process_function
        )
        
        # 查詢優化
        result = await manager.optimize_query(
            query="測試查詢",
            query_func=your_query_function
        )
        
        # 取得效能報告
        report = manager.get_optimization_report()
        print(f"優化統計: {report}")

# 執行
asyncio.run(main())
```

## 詳細配置

### OptimizationConfig 參數

```python
config = OptimizationConfig(
    # 批次處理設定
    batch_enabled=True,
    batch_size=32,                    # 預設批次大小
    max_batch_size=128,               # 最大批次大小
    parallel_workers=4,               # 並行工作者數量
    memory_threshold_mb=1024.0,       # 記憶體閾值 (MB)
    
    # 查詢快取設定
    query_cache_enabled=True,
    cache_ttl_seconds=3600,           # 快取存活時間 (秒)
    cache_max_size=10000,             # 最大快取項目數
    preload_enabled=True,             # 啟用預載
    
    # 成本追蹤設定
    cost_tracking_enabled=True,
    budget_limit_usd=100.0,           # 預算限制 (USD)
    quality_threshold=0.8,            # 品質閾值
    
    # 效能監控設定
    monitoring_enabled=True,
    monitoring_interval=5.0,          # 監控間隔 (秒)
    alert_thresholds={                # 警報閾值
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "error_rate": 5.0
    },
    
    # 儲存設定
    storage_path="logs/performance"   # 資料儲存路徑
)
```

## 使用指南

### 批次處理最佳化

1. **選擇合適的批次大小**

   ```python
   # 小批次 - 適合記憶體受限環境
   optimizer = BatchOptimizer(default_batch_size=16, max_batch_size=32)
   
   # 大批次 - 適合高效能環境
   optimizer = BatchOptimizer(default_batch_size=64, max_batch_size=256)
   ```

2. **並行處理調整**

   ```python
   # CPU 密集型任務
   optimizer = BatchOptimizer(parallel_workers=cpu_count())
   
   # I/O 密集型任務
   optimizer = BatchOptimizer(parallel_workers=cpu_count() * 2)
   ```

3. **記憶體管理**

   ```python
   # 啟用記憶體監控
   optimizer = BatchOptimizer(
       memory_threshold_mb=1024.0,
       enable_memory_monitoring=True
   )
   ```

### 查詢快取策略

1. **快取層級選擇**

   ```python
   # 僅記憶體快取 - 適合單機部署
   optimizer = QueryOptimizer(cache_backend="memory")
   
   # Redis 快取 - 適合分散式部署
   optimizer = QueryOptimizer(
       cache_backend="redis",
       redis_url="redis://localhost:6379"
   )
   ```

2. **快取失效策略**

   ```python
   # 時間基礎失效
   optimizer = QueryOptimizer(cache_ttl=3600)  # 1小時
   
   # 版本基礎失效
   optimizer = QueryOptimizer(version_based_invalidation=True)
   ```

3. **預載策略**

   ```python
   # 啟用智慧預載
   optimizer = QueryOptimizer(
       enable_preloading=True,
       preload_patterns=["常見查詢*", "熱門主題*"]
   )
   ```

### 成本控制

1. **預算管理**

   ```python
   optimizer = CostOptimizer(
       budget_limit=50.0,  # $50 月預算
       alert_threshold=0.8  # 80% 時發出警報
   )
   ```

2. **模型選擇策略**

   ```python
   # 成本優先
   recommendation = await optimizer.get_model_recommendation(
       operation_type="embedding",
       priority="cost"
   )
   
   # 品質優先
   recommendation = await optimizer.get_model_recommendation(
       operation_type="text_generation",
       priority="quality"
   )
   
   # 平衡模式
   recommendation = await optimizer.get_model_recommendation(
       operation_type="inference",
       priority="balanced"
   )
   ```

## 效能監控

### 監控指標

系統自動收集以下效能指標：

- **系統資源**: CPU 使用率、記憶體使用量、磁碟使用率
- **GPU 資源**: GPU 使用率、GPU 記憶體使用量 (如果可用)
- **應用程式**: 回應時間、吞吐量、錯誤率
- **快取**: 命中率、快取大小
- **資料庫**: 查詢時間、連線數量

### 自訂指標

```python
# 註冊自訂指標收集器
monitor.register_custom_collector(
    "active_users",
    lambda: get_active_user_count()
)

monitor.register_custom_collector(
    "queue_size",
    lambda: get_processing_queue_size()
)
```

### 警報設定

```python
# 設定 CPU 使用率警報
monitor.set_alert_threshold(
    "cpu_usage", 80.0,
    lambda metric, value, threshold: send_alert_email(
        f"CPU 使用率過高: {value}% (閾值: {threshold}%)"
    )
)

# 設定記憶體使用警報
monitor.set_alert_threshold(
    "memory_usage", 85.0,
    lambda metric, value, threshold: scale_up_resources()
)
```

## 基準測試

### 執行基準測試

```python
from src.chinese_graphrag.performance import BenchmarkRunner

runner = BenchmarkRunner(monitor)

# 定義測試配置
test_configs = [
    {
        "name": "嵌入效能測試",
        "func": embedding_benchmark,
        "params": {"batch_size": 32}
    },
    {
        "name": "查詢效能測試", 
        "func": query_benchmark,
        "params": {"query_type": "semantic"}
    }
]

# 執行比較測試
results = runner.run_comparative_benchmark(test_configs, iterations=10)

# 匯出結果
runner.export_results("benchmark_results.json", format="json")
```

### 自訂基準測試

```python
async def custom_benchmark():
    """自訂基準測試函數"""
    # 執行測試邏輯
    start_time = time.time()
    
    # 模擬工作負載
    await process_large_dataset()
    
    end_time = time.time()
    return {
        "processing_time": end_time - start_time,
        "items_processed": 1000
    }

# 執行自訂測試
result = await runner.run_benchmark(
    test_name="自訂效能測試",
    test_func=custom_benchmark,
    test_params={},
    iterations=5
)
```

## 最佳實踐

### 1. 批次處理優化

- **合理設定批次大小**: 根據可用記憶體和處理能力調整
- **監控記憶體使用**: 避免記憶體溢出
- **錯誤處理**: 實作適當的錯誤處理和重試機制
- **效能測試**: 定期執行基準測試以驗證優化效果

### 2. 查詢快取

- **快取鍵設計**: 使用有意義且唯一的快取鍵
- **快取失效**: 實作適當的快取失效策略
- **快取預熱**: 在系統啟動時預載常用資料
- **監控命中率**: 定期檢查快取命中率並調整策略

### 3. 成本控制

- **設定預算警報**: 在達到預算閾值時及時通知
- **模型選擇**: 根據任務需求選擇合適的模型
- **使用追蹤**: 詳細記錄模型使用情況
- **定期審查**: 定期審查成本報告並優化使用策略

### 4. 效能監控

- **設定合理閾值**: 根據系統特性設定警報閾值
- **歷史分析**: 利用歷史資料分析效能趨勢
- **自訂指標**: 收集業務相關的自訂指標
- **定期檢查**: 定期檢查監控資料並調整配置

## 故障排除

### 常見問題

#### 1. 批次處理效能不佳

**症狀**: 批次處理速度慢於預期

**可能原因**:

- 批次大小設定不當
- 並行工作者數量不足
- 記憶體不足導致頻繁 GC

**解決方案**:

```python
# 調整批次大小
optimizer.adjust_batch_size(factor=1.5)

# 增加並行工作者
optimizer.parallel_workers = min(cpu_count() * 2, 8)

# 啟用記憶體監控
optimizer.enable_memory_monitoring = True
```

#### 2. 快取命中率低

**症狀**: 快取命中率低於預期

**可能原因**:

- 快取鍵設計不當
- TTL 設定過短
- 快取大小限制

**解決方案**:

```python
# 調整 TTL
optimizer.cache_ttl = 7200  # 2小時

# 增加快取大小
optimizer.max_cache_size = 20000

# 檢查快取鍵設計
optimizer.enable_cache_key_analysis = True
```

#### 3. 成本超出預算

**症狀**: 模型使用成本超出預算

**可能原因**:

- 模型選擇不當
- 使用量超出預期
- 缺乏成本控制

**解決方案**:

```python
# 啟用嚴格預算控制
optimizer.strict_budget_enforcement = True

# 調整模型選擇策略
optimizer.model_selection_priority = "cost"

# 設定使用量限制
optimizer.set_usage_limit("gpt-4", max_tokens_per_day=10000)
```

#### 4. 監控資料異常

**症狀**: 監控資料顯示異常值

**可能原因**:

- 監控間隔設定不當
- 系統負載波動
- 監控器配置錯誤

**解決方案**:

```python
# 調整監控間隔
monitor.collection_interval = 10.0  # 10秒

# 啟用資料平滑
monitor.enable_data_smoothing = True

# 檢查監控器配置
monitor.validate_configuration()
```

### 除錯工具

#### 1. 效能分析

```python
# 啟用詳細日誌
import logging
logging.getLogger("chinese_graphrag.performance").setLevel(logging.DEBUG)

# 效能分析器
from src.chinese_graphrag.performance.profiler import PerformanceProfiler

profiler = PerformanceProfiler()
with profiler:
    # 執行需要分析的程式碼
    await your_function()

# 檢視分析結果
profiler.print_stats()
```

#### 2. 記憶體分析

```python
# 記憶體使用分析
from src.chinese_graphrag.performance.memory_analyzer import MemoryAnalyzer

analyzer = MemoryAnalyzer()
analyzer.start_monitoring()

# 執行程式碼
await your_memory_intensive_function()

# 生成記憶體報告
report = analyzer.generate_report()
print(report)
```

#### 3. 快取分析

```python
# 快取效能分析
cache_stats = optimizer.get_detailed_cache_stats()
print(f"快取命中率: {cache_stats['hit_rate']:.2%}")
print(f"平均查詢時間: {cache_stats['avg_query_time']:.3f}s")
print(f"快取大小: {cache_stats['cache_size']} 項目")
```

## 進階功能

### 1. 自適應優化

系統支援自適應優化，根據運行時效能自動調整參數：

```python
# 啟用自適應優化
config = OptimizationConfig(
    adaptive_optimization=True,
    adaptation_interval=300,  # 5分鐘調整一次
    adaptation_sensitivity=0.1  # 調整敏感度
)
```

### 2. 分散式優化

支援分散式環境下的效能優化：

```python
# 分散式配置
config = OptimizationConfig(
    distributed_mode=True,
    coordinator_url="redis://coordinator:6379",
    node_id="worker-1"
)
```

### 3. A/B 測試

支援效能優化策略的 A/B 測試：

```python
# A/B 測試配置
ab_test = ABTestManager()
ab_test.create_experiment(
    name="batch_size_test",
    variants={
        "small_batch": {"batch_size": 16},
        "large_batch": {"batch_size": 64}
    },
    traffic_split=0.5
)
```

## 總結

Chinese GraphRAG 的效能優化系統提供了全面的效能提升解決方案。通過合理配置和使用這些功能，您可以：

- 提升系統處理效能 2-5 倍
- 降低運行成本 20-40%
- 改善系統穩定性和可靠性
- 獲得詳細的效能洞察和分析

建議從基本配置開始，逐步啟用更多進階功能，並根據實際使用情況調整參數以達到最佳效能。
