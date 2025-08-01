"""
效能優化配置載入器

支援從 YAML、JSON 檔案載入配置，並提供環境變數覆蓋功能
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

from loguru import logger

from .optimizer_manager import OptimizationConfig


@dataclass
class PerformanceConfig:
    """完整的效能配置"""
    
    # 基本優化配置
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # 進階設定
    distributed: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    integrations: Dict[str, Any] = field(default_factory=dict)
    development: Dict[str, Any] = field(default_factory=dict)
    
    # 環境特定設定
    environments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "optimization": self.optimization.to_dict(),
            "distributed": self.distributed,
            "security": self.security,
            "integrations": self.integrations,
            "development": self.development,
            "environments": self.environments
        }


class ConfigLoader:
    """配置載入器"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 environment: str = "production"):
        """初始化配置載入器
        
        Args:
            config_path: 配置檔案路徑
            environment: 環境名稱 (development, staging, production)
        """
        self.config_path = config_path
        self.environment = environment
        self._config_cache: Optional[PerformanceConfig] = None
        
        # 預設配置檔案路徑
        if not self.config_path:
            self.config_path = self._find_default_config()
        
        logger.info(f"配置載入器初始化，環境: {environment}, 配置檔案: {self.config_path}")
    
    def _find_default_config(self) -> str:
        """尋找預設配置檔案"""
        possible_paths = [
            "config/performance_optimization.yaml",
            "config/performance_optimization.yml", 
            "config/performance.yaml",
            "config/performance.yml",
            "performance_config.yaml",
            "performance_config.yml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        # 如果找不到配置檔案，返回預設路徑
        return "config/performance_optimization.yaml"
    
    def load_config(self, force_reload: bool = False) -> PerformanceConfig:
        """載入配置
        
        Args:
            force_reload: 強制重新載入配置
        
        Returns:
            效能配置物件
        """
        if self._config_cache and not force_reload:
            return self._config_cache
        
        try:
            # 載入基礎配置
            base_config = self._load_config_file()
            
            # 應用環境特定配置
            config = self._apply_environment_config(base_config)
            
            # 應用環境變數覆蓋
            config = self._apply_environment_variables(config)
            
            # 驗證配置
            self._validate_config(config)
            
            # 建立配置物件
            performance_config = self._create_performance_config(config)
            
            # 快取配置
            self._config_cache = performance_config
            
            logger.info("配置載入完成")
            return performance_config
            
        except Exception as e:
            logger.error(f"配置載入失敗: {e}")
            # 返回預設配置
            return PerformanceConfig()
    
    def _load_config_file(self) -> Dict[str, Any]:
        """載入配置檔案"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            logger.warning(f"配置檔案不存在: {config_file}，使用預設配置")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"不支援的配置檔案格式: {config_file.suffix}")
            
            logger.info(f"配置檔案載入成功: {config_file}")
            return config or {}
            
        except Exception as e:
            logger.error(f"配置檔案載入失敗: {e}")
            return {}
    
    def _apply_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """應用環境特定配置"""
        if "environments" not in config:
            return config
        
        env_config = config["environments"].get(self.environment, {})
        if not env_config:
            logger.info(f"未找到環境 {self.environment} 的特定配置")
            return config
        
        # 深度合併環境配置
        merged_config = self._deep_merge(config.copy(), env_config)
        
        logger.info(f"已應用環境 {self.environment} 的配置")
        return merged_config
    
    def _apply_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """應用環境變數覆蓋"""
        env_mappings = {
            # 批次處理設定
            "PERF_BATCH_SIZE": ("batch_optimization", "default_batch_size", int),
            "PERF_MAX_BATCH_SIZE": ("batch_optimization", "max_batch_size", int),
            "PERF_PARALLEL_WORKERS": ("batch_optimization", "parallel_workers", int),
            "PERF_MEMORY_THRESHOLD": ("batch_optimization", "memory_threshold_mb", float),
            
            # 快取設定
            "PERF_CACHE_ENABLED": ("query_optimization", "cache_enabled", bool),
            "PERF_CACHE_TTL": ("query_optimization", "cache_ttl_seconds", int),
            "PERF_CACHE_SIZE": ("query_optimization", "cache_max_size", int),
            "PERF_REDIS_URL": ("query_optimization", "redis", "url", str),
            
            # 成本設定
            "PERF_BUDGET_LIMIT": ("cost_optimization", "budget_limit_usd", float),
            "PERF_QUALITY_THRESHOLD": ("cost_optimization", "quality_threshold", float),
            "PERF_STRICT_BUDGET": ("cost_optimization", "strict_budget_enforcement", bool),
            
            # 監控設定
            "PERF_MONITORING_ENABLED": ("performance_monitoring", "enabled", bool),
            "PERF_MONITORING_INTERVAL": ("performance_monitoring", "collection_interval", float),
            "PERF_CPU_THRESHOLD": ("performance_monitoring", "alert_thresholds", "cpu_usage", float),
            "PERF_MEMORY_THRESHOLD": ("performance_monitoring", "alert_thresholds", "memory_usage", float),
            
            # 儲存設定
            "PERF_STORAGE_PATH": ("storage", "base_path", str),
            "PERF_LOG_LEVEL": ("storage", "logging", "level", str),
        }
        
        for env_var, path_info in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # 解析路徑和類型
                    *path_parts, value_type = path_info
                    
                    # 轉換值類型
                    if value_type == bool:
                        parsed_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == int:
                        parsed_value = int(env_value)
                    elif value_type == float:
                        parsed_value = float(env_value)
                    else:
                        parsed_value = env_value
                    
                    # 設定配置值
                    self._set_nested_value(config, path_parts, parsed_value)
                    
                    logger.info(f"環境變數覆蓋: {env_var} = {parsed_value}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"環境變數 {env_var} 值無效: {env_value}, 錯誤: {e}")
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合併字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any):
        """設定巢狀字典值"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _validate_config(self, config: Dict[str, Any]):
        """驗證配置"""
        # 驗證批次處理配置
        if "batch_optimization" in config:
            batch_config = config["batch_optimization"]
            if batch_config.get("default_batch_size", 0) <= 0:
                raise ValueError("default_batch_size 必須大於 0")
            if batch_config.get("max_batch_size", 0) < batch_config.get("default_batch_size", 1):
                raise ValueError("max_batch_size 必須大於等於 default_batch_size")
            if batch_config.get("parallel_workers", 0) <= 0:
                raise ValueError("parallel_workers 必須大於 0")
        
        # 驗證快取配置
        if "query_optimization" in config:
            cache_config = config["query_optimization"]
            if cache_config.get("cache_ttl_seconds", 0) <= 0:
                logger.warning("cache_ttl_seconds 應該大於 0")
            if cache_config.get("cache_max_size", 0) <= 0:
                logger.warning("cache_max_size 應該大於 0")
        
        # 驗證成本配置
        if "cost_optimization" in config:
            cost_config = config["cost_optimization"]
            if cost_config.get("budget_limit_usd") is not None:
                if cost_config["budget_limit_usd"] <= 0:
                    raise ValueError("budget_limit_usd 必須大於 0")
            quality_threshold = cost_config.get("quality_threshold", 0.8)
            if not 0 <= quality_threshold <= 1:
                raise ValueError("quality_threshold 必須在 0 到 1 之間")
        
        logger.info("配置驗證通過")
    
    def _create_performance_config(self, config: Dict[str, Any]) -> PerformanceConfig:
        """建立效能配置物件"""
        # 建立基本優化配置
        optimization_config = self._create_optimization_config(config)
        
        # 建立完整配置
        performance_config = PerformanceConfig(
            optimization=optimization_config,
            distributed=config.get("distributed", {}),
            security=config.get("security", {}),
            integrations=config.get("integrations", {}),
            development=config.get("development", {}),
            environments=config.get("environments", {})
        )
        
        return performance_config
    
    def _create_optimization_config(self, config: Dict[str, Any]) -> OptimizationConfig:
        """建立優化配置物件"""
        # 批次處理設定
        batch_config = config.get("batch_optimization", {})
        
        # 查詢優化設定
        query_config = config.get("query_optimization", {})
        
        # 成本優化設定
        cost_config = config.get("cost_optimization", {})
        
        # 效能監控設定
        monitoring_config = config.get("performance_monitoring", {})
        
        # 儲存設定
        storage_config = config.get("storage", {})
        
        return OptimizationConfig(
            # 批次處理
            batch_enabled=batch_config.get("enabled", True),
            batch_size=batch_config.get("default_batch_size", 32),
            max_batch_size=batch_config.get("max_batch_size", 128),
            parallel_workers=batch_config.get("parallel_workers", 4),
            memory_threshold_mb=batch_config.get("memory_threshold_mb", 1024.0),
            
            # 查詢快取
            query_cache_enabled=query_config.get("cache_enabled", True),
            cache_ttl_seconds=query_config.get("cache_ttl_seconds", 3600),
            cache_max_size=query_config.get("cache_max_size", 10000),
            preload_enabled=query_config.get("enable_preloading", True),
            
            # 成本優化
            cost_tracking_enabled=cost_config.get("tracking_enabled", True),
            budget_limit_usd=cost_config.get("budget_limit_usd"),
            quality_threshold=cost_config.get("quality_threshold", 0.8),
            
            # 效能監控
            monitoring_enabled=monitoring_config.get("enabled", True),
            monitoring_interval=monitoring_config.get("collection_interval", 5.0),
            alert_thresholds=monitoring_config.get("alert_thresholds", {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "error_rate": 5.0
            }),
            
            # 儲存
            storage_path=storage_config.get("base_path", "logs/performance")
        )
    
    def save_config(self, config: PerformanceConfig, output_path: str):
        """儲存配置到檔案
        
        Args:
            config: 效能配置物件
            output_path: 輸出檔案路徑
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        try:
            if output_file.suffix.lower() in ['.yaml', '.yml']:
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            elif output_file.suffix.lower() == '.json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支援的輸出格式: {output_file.suffix}")
            
            logger.info(f"配置已儲存至: {output_file}")
            
        except Exception as e:
            logger.error(f"配置儲存失敗: {e}")
            raise
    
    def reload_config(self) -> PerformanceConfig:
        """重新載入配置"""
        logger.info("重新載入配置...")
        return self.load_config(force_reload=True)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """取得配置摘要"""
        config = self.load_config()
        
        return {
            "environment": self.environment,
            "config_path": self.config_path,
            "batch_optimization": {
                "enabled": config.optimization.batch_enabled,
                "batch_size": config.optimization.batch_size,
                "parallel_workers": config.optimization.parallel_workers
            },
            "query_optimization": {
                "cache_enabled": config.optimization.query_cache_enabled,
                "cache_ttl": config.optimization.cache_ttl_seconds,
                "cache_size": config.optimization.cache_max_size
            },
            "cost_optimization": {
                "tracking_enabled": config.optimization.cost_tracking_enabled,
                "budget_limit": config.optimization.budget_limit_usd,
                "quality_threshold": config.optimization.quality_threshold
            },
            "monitoring": {
                "enabled": config.optimization.monitoring_enabled,
                "interval": config.optimization.monitoring_interval
            }
        }


# 全域配置載入器實例
_global_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[str] = None, 
                     environment: Optional[str] = None) -> ConfigLoader:
    """取得全域配置載入器實例
    
    Args:
        config_path: 配置檔案路徑
        environment: 環境名稱
    
    Returns:
        配置載入器實例
    """
    global _global_config_loader
    
    if _global_config_loader is None:
        env = environment or os.getenv("PERF_ENVIRONMENT", "production")
        _global_config_loader = ConfigLoader(config_path, env)
    
    return _global_config_loader


def load_performance_config(config_path: Optional[str] = None,
                          environment: Optional[str] = None) -> PerformanceConfig:
    """載入效能配置
    
    Args:
        config_path: 配置檔案路徑
        environment: 環境名稱
    
    Returns:
        效能配置物件
    """
    loader = get_config_loader(config_path, environment)
    return loader.load_config()