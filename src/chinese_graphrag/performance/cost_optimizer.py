"""
模型成本和品質優化器

提供 LLM 和 Embedding 模型使用量監控、智慧模型選擇、成本控制和品質評估功能
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import threading

from loguru import logger


class ModelType(Enum):
    """模型類型"""
    LLM = "llm"
    EMBEDDING = "embedding"


class TaskType(Enum):
    """任務類型"""
    QUERY = "query"
    INDEXING = "indexing"
    EMBEDDING = "embedding"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    type: ModelType
    cost_per_token: float
    max_tokens: int
    quality_score: float  # 0-1 之間的品質分數
    latency_ms: float     # 平均延遲（毫秒）
    availability: float   # 可用性 0-1
    supported_tasks: List[TaskType]
    
    # 成本相關
    daily_budget: Optional[float] = None
    monthly_budget: Optional[float] = None
    
    # 效能相關
    throughput_limit: Optional[int] = None  # 每分鐘請求數限制
    concurrent_limit: Optional[int] = None  # 並發請求限制


@dataclass
class UsageRecord:
    """使用記錄"""
    timestamp: float
    model_name: str
    model_type: ModelType
    task_type: TaskType
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency_ms: float
    quality_score: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class CostBudget:
    """成本預算"""
    daily_limit: float
    monthly_limit: float
    current_daily_usage: float = 0.0
    current_monthly_usage: float = 0.0
    last_reset_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    last_reset_month: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m"))
    
    def is_within_budget(self, additional_cost: float) -> bool:
        """檢查是否在預算範圍內"""
        return (self.current_daily_usage + additional_cost <= self.daily_limit and
                self.current_monthly_usage + additional_cost <= self.monthly_limit)
    
    def add_usage(self, cost: float):
        """添加使用量"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_month = datetime.now().strftime("%Y-%m")
        
        # 重設日預算
        if current_date != self.last_reset_date:
            self.current_daily_usage = 0.0
            self.last_reset_date = current_date
        
        # 重設月預算
        if current_month != self.last_reset_month:
            self.current_monthly_usage = 0.0
            self.last_reset_month = current_month
        
        self.current_daily_usage += cost
        self.current_monthly_usage += cost


class ModelUsageTracker:
    """模型使用量追蹤器"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """初始化使用量追蹤器
        
        Args:
            storage_path: 儲存路徑
        """
        self.storage_path = Path(storage_path) if storage_path else Path("logs/model_usage")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 使用記錄
        self._usage_records: deque = deque(maxlen=10000)  # 保留最近10000條記錄
        self._usage_lock = threading.RLock()
        
        # 統計資料
        self._daily_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(float))
        self._monthly_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(float))
        
        # 即時統計
        self._realtime_stats: Dict[str, Any] = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0,
            "success_rate": 0.0
        }
        
        logger.info(f"模型使用量追蹤器初始化完成，儲存路徑: {self.storage_path}")
    
    def record_usage(self, 
                    model_name: str,
                    model_type: ModelType,
                    task_type: TaskType,
                    input_tokens: int,
                    output_tokens: int,
                    cost: float,
                    latency_ms: float,
                    quality_score: Optional[float] = None,
                    success: bool = True,
                    error_message: Optional[str] = None):
        """記錄模型使用量
        
        Args:
            model_name: 模型名稱
            model_type: 模型類型
            task_type: 任務類型
            input_tokens: 輸入 token 數
            output_tokens: 輸出 token 數
            cost: 成本
            latency_ms: 延遲（毫秒）
            quality_score: 品質分數
            success: 是否成功
            error_message: 錯誤訊息
        """
        record = UsageRecord(
            timestamp=time.time(),
            model_name=model_name,
            model_type=model_type,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            quality_score=quality_score,
            success=success,
            error_message=error_message
        )
        
        with self._usage_lock:
            self._usage_records.append(record)
            self._update_stats(record)
        
        # 異步儲存記錄
        asyncio.create_task(self._save_record(record))
        
        logger.debug(f"記錄模型使用: {model_name} - {task_type.value}, "
                    f"tokens: {input_tokens + output_tokens}, cost: ${cost:.4f}")
    
    def _update_stats(self, record: UsageRecord):
        """更新統計資料"""
        date_key = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d")
        month_key = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m")
        
        # 更新日統計
        daily_stats = self._daily_stats[date_key]
        daily_stats["total_requests"] += 1
        daily_stats["total_tokens"] += record.total_tokens
        daily_stats["total_cost"] += record.cost
        daily_stats["total_latency"] += record.latency_ms
        daily_stats["successful_requests"] += 1 if record.success else 0
        
        # 更新月統計
        monthly_stats = self._monthly_stats[month_key]
        monthly_stats["total_requests"] += 1
        monthly_stats["total_tokens"] += record.total_tokens
        monthly_stats["total_cost"] += record.cost
        monthly_stats["total_latency"] += record.latency_ms
        monthly_stats["successful_requests"] += 1 if record.success else 0
        
        # 更新即時統計
        self._realtime_stats["total_requests"] += 1
        self._realtime_stats["total_tokens"] += record.total_tokens
        self._realtime_stats["total_cost"] += record.cost
        
        # 計算平均值
        total_requests = self._realtime_stats["total_requests"]
        self._realtime_stats["avg_latency"] = (
            (self._realtime_stats["avg_latency"] * (total_requests - 1) + record.latency_ms) / total_requests
        )
        
        successful_requests = sum(1 for r in self._usage_records if r.success)
        self._realtime_stats["success_rate"] = successful_requests / total_requests if total_requests > 0 else 0
    
    async def _save_record(self, record: UsageRecord):
        """儲存使用記錄"""
        try:
            date_str = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d")
            log_file = self.storage_path / f"usage_{date_str}.jsonl"
            
            record_data = {
                "timestamp": record.timestamp,
                "model_name": record.model_name,
                "model_type": record.model_type.value,
                "task_type": record.task_type.value,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "total_tokens": record.total_tokens,
                "cost": record.cost,
                "latency_ms": record.latency_ms,
                "quality_score": record.quality_score,
                "success": record.success,
                "error_message": record.error_message
            }
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"儲存使用記錄失敗: {e}")
    
    def get_usage_stats(self, 
                       period: str = "today",
                       model_name: Optional[str] = None,
                       task_type: Optional[TaskType] = None) -> Dict[str, Any]:
        """取得使用統計
        
        Args:
            period: 統計期間 ("today", "month", "all")
            model_name: 特定模型名稱
            task_type: 特定任務類型
        
        Returns:
            統計資料
        """
        with self._usage_lock:
            if period == "today":
                date_key = datetime.now().strftime("%Y-%m-%d")
                base_stats = self._daily_stats.get(date_key, {})
            elif period == "month":
                month_key = datetime.now().strftime("%Y-%m")
                base_stats = self._monthly_stats.get(month_key, {})
            else:  # all
                base_stats = self._realtime_stats.copy()
            
            # 過濾記錄
            filtered_records = list(self._usage_records)
            
            if model_name:
                filtered_records = [r for r in filtered_records if r.model_name == model_name]
            
            if task_type:
                filtered_records = [r for r in filtered_records if r.task_type == task_type]
            
            # 計算過濾後的統計
            if filtered_records:
                total_cost = sum(r.cost for r in filtered_records)
                total_tokens = sum(r.total_tokens for r in filtered_records)
                avg_latency = sum(r.latency_ms for r in filtered_records) / len(filtered_records)
                success_rate = sum(1 for r in filtered_records if r.success) / len(filtered_records)
                
                # 按模型分組統計
                model_stats = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
                for record in filtered_records:
                    model_stats[record.model_name]["requests"] += 1
                    model_stats[record.model_name]["tokens"] += record.total_tokens
                    model_stats[record.model_name]["cost"] += record.cost
                
                return {
                    "period": period,
                    "total_requests": len(filtered_records),
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                    "avg_latency_ms": avg_latency,
                    "success_rate": success_rate,
                    "model_breakdown": dict(model_stats),
                    "cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0
                }
            else:
                return {
                    "period": period,
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "avg_latency_ms": 0.0,
                    "success_rate": 0.0,
                    "model_breakdown": {},
                    "cost_per_token": 0.0
                }
    
    def get_top_models(self, limit: int = 10, metric: str = "cost") -> List[Tuple[str, float]]:
        """取得使用量最高的模型
        
        Args:
            limit: 返回數量限制
            metric: 排序指標 ("cost", "tokens", "requests")
        
        Returns:
            模型使用量排序列表
        """
        model_metrics = defaultdict(float)
        
        with self._usage_lock:
            for record in self._usage_records:
                if metric == "cost":
                    model_metrics[record.model_name] += record.cost
                elif metric == "tokens":
                    model_metrics[record.model_name] += record.total_tokens
                elif metric == "requests":
                    model_metrics[record.model_name] += 1
        
        return sorted(model_metrics.items(), key=lambda x: x[1], reverse=True)[:limit]


class CostOptimizer:
    """成本優化器
    
    提供智慧模型選擇、成本控制和品質評估功能
    """
    
    def __init__(self, usage_tracker: Optional[ModelUsageTracker] = None):
        """初始化成本優化器
        
        Args:
            usage_tracker: 使用量追蹤器
        """
        self.usage_tracker = usage_tracker or ModelUsageTracker()
        
        # 模型配置
        self._model_configs: Dict[str, ModelConfig] = {}
        
        # 預算管理
        self._budgets: Dict[str, CostBudget] = {}
        
        # 模型選擇策略
        self._selection_strategies: Dict[TaskType, Callable] = {}
        
        # 品質評估器
        self._quality_evaluators: Dict[TaskType, Callable] = {}
        
        # 效能歷史
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("成本優化器初始化完成")
    
    def register_model(self, config: ModelConfig):
        """註冊模型配置
        
        Args:
            config: 模型配置
        """
        self._model_configs[config.name] = config
        
        # 初始化預算
        if config.daily_budget or config.monthly_budget:
            self._budgets[config.name] = CostBudget(
                daily_limit=config.daily_budget or float('inf'),
                monthly_limit=config.monthly_budget or float('inf')
            )
        
        logger.info(f"註冊模型: {config.name} ({config.type.value})")
    
    def register_selection_strategy(self, task_type: TaskType, strategy: Callable):
        """註冊模型選擇策略
        
        Args:
            task_type: 任務類型
            strategy: 選擇策略函數
        """
        self._selection_strategies[task_type] = strategy
        logger.info(f"註冊選擇策略: {task_type.value}")
    
    def register_quality_evaluator(self, task_type: TaskType, evaluator: Callable):
        """註冊品質評估器
        
        Args:
            task_type: 任務類型
            evaluator: 評估函數
        """
        self._quality_evaluators[task_type] = evaluator
        logger.info(f"註冊品質評估器: {task_type.value}")
    
    def select_optimal_model(self, 
                           task_type: TaskType,
                           input_size: int,
                           quality_requirement: float = 0.8,
                           cost_priority: float = 0.5) -> Optional[str]:
        """選擇最佳模型
        
        Args:
            task_type: 任務類型
            input_size: 輸入大小（token 數）
            quality_requirement: 品質要求（0-1）
            cost_priority: 成本優先級（0-1，1表示完全優先考慮成本）
        
        Returns:
            最佳模型名稱
        """
        # 過濾支援該任務的模型
        candidate_models = [
            config for config in self._model_configs.values()
            if task_type in config.supported_tasks and config.quality_score >= quality_requirement
        ]
        
        if not candidate_models:
            logger.warning(f"沒有找到支援 {task_type.value} 任務的模型")
            return None
        
        # 檢查預算限制
        candidate_models = [
            config for config in candidate_models
            if self._check_budget_availability(config.name, input_size)
        ]
        
        if not candidate_models:
            logger.warning("所有候選模型都超出預算限制")
            return None
        
        # 使用自訂選擇策略
        if task_type in self._selection_strategies:
            try:
                selected_model = self._selection_strategies[task_type](
                    candidate_models, input_size, quality_requirement, cost_priority
                )
                if selected_model:
                    return selected_model
            except Exception as e:
                logger.warning(f"自訂選擇策略失敗: {e}")
        
        # 預設選擇策略：平衡成本和品質
        best_model = None
        best_score = -1
        
        for config in candidate_models:
            # 計算綜合分數
            cost_score = 1 / (1 + config.cost_per_token * input_size)  # 成本越低分數越高
            quality_score = config.quality_score
            latency_score = 1 / (1 + config.latency_ms / 1000)  # 延遲越低分數越高
            
            # 加入歷史效能
            historical_performance = self._get_historical_performance(config.name)
            performance_score = historical_performance.get("avg_quality", config.quality_score)
            
            # 綜合分數
            composite_score = (
                cost_priority * cost_score +
                (1 - cost_priority) * 0.6 * quality_score +
                (1 - cost_priority) * 0.2 * latency_score +
                (1 - cost_priority) * 0.2 * performance_score
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = config.name
        
        logger.debug(f"選擇最佳模型: {best_model} (分數: {best_score:.3f})")
        return best_model
    
    def _check_budget_availability(self, model_name: str, estimated_tokens: int) -> bool:
        """檢查預算可用性
        
        Args:
            model_name: 模型名稱
            estimated_tokens: 估計 token 數
        
        Returns:
            是否在預算範圍內
        """
        if model_name not in self._budgets:
            return True
        
        config = self._model_configs[model_name]
        estimated_cost = config.cost_per_token * estimated_tokens
        
        return self._budgets[model_name].is_within_budget(estimated_cost)
    
    def _get_historical_performance(self, model_name: str) -> Dict[str, float]:
        """取得歷史效能資料
        
        Args:
            model_name: 模型名稱
        
        Returns:
            歷史效能統計
        """
        if model_name not in self._performance_history:
            return {"avg_quality": 0.8, "avg_latency": 1000, "success_rate": 0.95}
        
        history = list(self._performance_history[model_name])
        if not history:
            return {"avg_quality": 0.8, "avg_latency": 1000, "success_rate": 0.95}
        
        avg_quality = sum(record.get("quality", 0.8) for record in history) / len(history)
        avg_latency = sum(record.get("latency", 1000) for record in history) / len(history)
        success_rate = sum(1 for record in history if record.get("success", True)) / len(history)
        
        return {
            "avg_quality": avg_quality,
            "avg_latency": avg_latency,
            "success_rate": success_rate
        }
    
    def record_model_usage(self,
                          model_name: str,
                          task_type: TaskType,
                          input_tokens: int,
                          output_tokens: int,
                          latency_ms: float,
                          success: bool = True,
                          quality_score: Optional[float] = None,
                          error_message: Optional[str] = None):
        """記錄模型使用情況
        
        Args:
            model_name: 模型名稱
            task_type: 任務類型
            input_tokens: 輸入 token 數
            output_tokens: 輸出 token 數
            latency_ms: 延遲（毫秒）
            success: 是否成功
            quality_score: 品質分數
            error_message: 錯誤訊息
        """
        if model_name not in self._model_configs:
            logger.warning(f"未知模型: {model_name}")
            return
        
        config = self._model_configs[model_name]
        cost = config.cost_per_token * (input_tokens + output_tokens)
        
        # 記錄到使用量追蹤器
        self.usage_tracker.record_usage(
            model_name=model_name,
            model_type=config.type,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            quality_score=quality_score,
            success=success,
            error_message=error_message
        )
        
        # 更新預算
        if model_name in self._budgets:
            self._budgets[model_name].add_usage(cost)
        
        # 更新效能歷史
        self._performance_history[model_name].append({
            "timestamp": time.time(),
            "quality": quality_score or config.quality_score,
            "latency": latency_ms,
            "success": success,
            "cost": cost
        })
    
    async def evaluate_response_quality(self,
                                      task_type: TaskType,
                                      input_text: str,
                                      output_text: str,
                                      model_name: str) -> float:
        """評估回應品質
        
        Args:
            task_type: 任務類型
            input_text: 輸入文本
            output_text: 輸出文本
            model_name: 模型名稱
        
        Returns:
            品質分數（0-1）
        """
        if task_type in self._quality_evaluators:
            try:
                quality_score = await self._quality_evaluators[task_type](
                    input_text, output_text, model_name
                )
                return max(0.0, min(1.0, quality_score))
            except Exception as e:
                logger.warning(f"品質評估失敗: {e}")
        
        # 預設品質評估（基於長度和基本規則）
        return self._default_quality_evaluation(input_text, output_text)
    
    def _default_quality_evaluation(self, input_text: str, output_text: str) -> float:
        """預設品質評估"""
        if not output_text or not output_text.strip():
            return 0.0
        
        # 基本品質指標
        length_score = min(1.0, len(output_text) / max(1, len(input_text) * 0.5))
        
        # 檢查是否包含中文
        chinese_chars = sum(1 for char in output_text if '\u4e00' <= char <= '\u9fff')
        chinese_score = min(1.0, chinese_chars / max(1, len(output_text) * 0.3))
        
        # 綜合分數
        return (length_score * 0.4 + chinese_score * 0.6)
    
    def get_cost_analysis(self, period: str = "today") -> Dict[str, Any]:
        """取得成本分析
        
        Args:
            period: 分析期間
        
        Returns:
            成本分析結果
        """
        usage_stats = self.usage_tracker.get_usage_stats(period)
        top_models = self.usage_tracker.get_top_models(limit=5, metric="cost")
        
        # 預算狀態
        budget_status = {}
        for model_name, budget in self._budgets.items():
            budget_status[model_name] = {
                "daily_usage": budget.current_daily_usage,
                "daily_limit": budget.daily_limit,
                "daily_remaining": budget.daily_limit - budget.current_daily_usage,
                "monthly_usage": budget.current_monthly_usage,
                "monthly_limit": budget.monthly_limit,
                "monthly_remaining": budget.monthly_limit - budget.current_monthly_usage
            }
        
        return {
            "usage_stats": usage_stats,
            "top_models_by_cost": top_models,
            "budget_status": budget_status,
            "cost_optimization_suggestions": self._generate_cost_suggestions()
        }
    
    def _generate_cost_suggestions(self) -> List[str]:
        """生成成本優化建議"""
        suggestions = []
        
        # 分析高成本模型
        top_models = self.usage_tracker.get_top_models(limit=3, metric="cost")
        for model_name, cost in top_models:
            if model_name in self._model_configs:
                config = self._model_configs[model_name]
                if config.cost_per_token > 0.01:  # 高成本閾值
                    suggestions.append(f"考慮為高成本模型 {model_name} 尋找替代方案")
        
        # 檢查預算使用情況
        for model_name, budget in self._budgets.items():
            daily_usage_rate = budget.current_daily_usage / budget.daily_limit
            if daily_usage_rate > 0.8:
                suggestions.append(f"模型 {model_name} 日預算使用率過高 ({daily_usage_rate:.1%})")
        
        return suggestions
    
    def optimize_model_selection(self):
        """優化模型選擇策略"""
        # 分析歷史效能資料
        performance_analysis = {}
        
        for model_name, history in self._performance_history.items():
            if len(history) >= 10:  # 需要足夠的資料
                recent_history = list(history)[-20:]  # 最近20次
                
                avg_quality = sum(record["quality"] for record in recent_history) / len(recent_history)
                avg_latency = sum(record["latency"] for record in recent_history) / len(recent_history)
                success_rate = sum(1 for record in recent_history if record["success"]) / len(recent_history)
                avg_cost = sum(record["cost"] for record in recent_history) / len(recent_history)
                
                performance_analysis[model_name] = {
                    "avg_quality": avg_quality,
                    "avg_latency": avg_latency,
                    "success_rate": success_rate,
                    "avg_cost": avg_cost,
                    "efficiency_score": avg_quality / (avg_cost + 0.001)  # 品質/成本比
                }
        
        # 更新模型配置中的品質分數
        for model_name, analysis in performance_analysis.items():
            if model_name in self._model_configs:
                config = self._model_configs[model_name]
                # 使用移動平均更新品質分數
                config.quality_score = 0.7 * config.quality_score + 0.3 * analysis["avg_quality"]
                config.latency_ms = 0.7 * config.latency_ms + 0.3 * analysis["avg_latency"]
        
        logger.info("模型選擇策略已優化")
        return performance_analysis