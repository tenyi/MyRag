"""
GraphRAG 配置驗證器

提供完整的 GraphRAG 配置檔案驗證功能，確保配置的正確性和完整性
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class ValidationLevel(Enum):
    """驗證等級"""
    ERROR = "error"      # 錯誤，會導致系統無法運行
    WARNING = "warning"  # 警告，可能影響性能或功能
    INFO = "info"       # 資訊，建議優化


@dataclass
class ValidationResult:
    """驗證結果"""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    info: List[Dict[str, Any]]
    
    def add_issue(self, level: ValidationLevel, section: str, key: str, message: str, suggestion: Optional[str] = None):
        """添加驗證問題"""
        issue = {
            "section": section,
            "key": key,
            "message": message,
            "suggestion": suggestion or ""
        }
        
        if level == ValidationLevel.ERROR:
            self.errors.append(issue)
            self.is_valid = False
        elif level == ValidationLevel.WARNING:
            self.warnings.append(issue)
        elif level == ValidationLevel.INFO:
            self.info.append(issue)
    
    def get_summary(self) -> str:
        """獲取驗證結果摘要"""
        status = "✅ 有效" if self.is_valid else "❌ 無效"
        return f"{status} - 錯誤: {len(self.errors)}, 警告: {len(self.warnings)}, 建議: {len(self.info)}"


class GraphRAGConfigValidator:
    """
    GraphRAG 配置驗證器
    
    驗證 GraphRAG 配置檔案的正確性，包括：
    - 必需欄位檢查
    - 資料類型驗證
    - 數值範圍檢查
    - 依賴關係驗證
    - 中文處理特定配置檢查
    """
    
    def __init__(self):
        """初始化驗證器"""
        self.required_sections = {
            "models": ["default_chat_model", "default_embedding_model"],
            "input": ["type", "base_dir"],
            "chunks": ["size", "overlap"],
            "storage": ["type", "base_dir"],
            "cache": ["type", "base_dir"]
        }
        
        self.valid_model_types = {
            "chat": ["openai_chat", "ollama", "azure_openai_chat"],
            "embedding": ["openai_embedding", "ollama", "azure_openai_embedding", "sentence_transformers"]
        }
        
        self.valid_ranges = {
            "chunks.size": (100, 10000),
            "chunks.overlap": (0, 1000),
            "temperature": (0.0, 2.0),
            "max_tokens": (100, 32000)
        }
    
    def validate_settings_yaml(self, config_path: Union[str, Path]) -> ValidationResult:
        """
        驗證 GraphRAG 配置檔案
        
        Args:
            config_path: 配置檔案路徑
            
        Returns:
            ValidationResult: 驗證結果
        """
        config_path = Path(config_path)
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
        
        try:
            # 檢查檔案是否存在
            if not config_path.exists():
                result.add_issue(
                    ValidationLevel.ERROR,
                    "file",
                    "existence",
                    f"配置檔案不存在: {config_path}",
                    "請確認檔案路徑是否正確"
                )
                return result
            
            # 載入 YAML 配置
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                result.add_issue(
                    ValidationLevel.ERROR,
                    "yaml",
                    "syntax",
                    f"YAML 語法錯誤: {e}",
                    "請檢查 YAML 格式是否正確"
                )
                return result
            
            if not isinstance(config, dict):
                result.add_issue(
                    ValidationLevel.ERROR,
                    "yaml",
                    "structure",
                    "配置檔案必須是字典格式",
                    "請確認 YAML 檔案結構正確"
                )
                return result
            
            # 驗證各個區段
            self._validate_models_section(config, result)
            self._validate_input_section(config, result)
            self._validate_chunks_section(config, result)
            self._validate_storage_section(config, result)
            self._validate_cache_section(config, result)
            self._validate_workflows_section(config, result)
            self._validate_chinese_processing_section(config, result)
            self._validate_performance_section(config, result)
            self._validate_environment_variables(config, result)
            
            logger.info(f"配置驗證完成: {result.get_summary()}")
            
        except Exception as e:
            result.add_issue(
                ValidationLevel.ERROR,
                "validation",
                "unexpected_error",
                f"驗證過程中發生未預期錯誤: {e}",
                "請檢查驗證器實現或聯繫技術支援"
            )
            logger.error(f"配置驗證錯誤: {e}")
        
        return result
    
    def _validate_models_section(self, config: Dict[str, Any], result: ValidationResult):
        """驗證模型配置區段"""
        if "models" not in config:
            result.add_issue(
                ValidationLevel.ERROR,
                "models",
                "missing_section",
                "缺少 models 配置區段",
                "請添加 models 區段並配置必需的模型"
            )
            return
        
        models = config["models"]
        if not isinstance(models, dict):
            result.add_issue(
                ValidationLevel.ERROR,
                "models",
                "invalid_type",
                "models 必須是字典格式"
            )
            return
        
        # 檢查必需的模型
        required_models = ["default_chat_model", "default_embedding_model"]
        for model_name in required_models:
            if model_name not in models:
                result.add_issue(
                    ValidationLevel.ERROR,
                    "models",
                    model_name,
                    f"缺少必需的模型配置: {model_name}",
                    f"請添加 {model_name} 配置"
                )
                continue
            
            model_config = models[model_name]
            self._validate_model_config(model_name, model_config, result)
        
        # 檢查其他模型配置
        for model_name, model_config in models.items():
            if model_name not in required_models:
                self._validate_model_config(model_name, model_config, result)
    
    def _validate_model_config(self, model_name: str, model_config: Dict[str, Any], result: ValidationResult):
        """驗證單個模型配置"""
        if not isinstance(model_config, dict):
            result.add_issue(
                ValidationLevel.ERROR,
                "models",
                model_name,
                f"模型配置 {model_name} 必須是字典格式"
            )
            return
        
        # 檢查必需欄位
        if "type" not in model_config:
            result.add_issue(
                ValidationLevel.ERROR,
                "models",
                f"{model_name}.type",
                f"模型 {model_name} 缺少 type 欄位",
                "請指定模型類型，如 openai_chat, ollama 等"
            )
            return
        
        model_type = model_config["type"]
        
        # 驗證模型類型
        all_valid_types = []
        for types in self.valid_model_types.values():
            all_valid_types.extend(types)
        
        if model_type not in all_valid_types:
            result.add_issue(
                ValidationLevel.ERROR,
                "models",
                f"{model_name}.type",
                f"不支援的模型類型: {model_type}",
                f"支援的類型: {', '.join(all_valid_types)}"
            )
        
        # 檢查 API key（對於需要的模型類型）
        if model_type in ["openai_chat", "openai_embedding", "azure_openai_chat", "azure_openai_embedding"]:
            if "api_key" not in model_config:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "models",
                    f"{model_name}.api_key",
                    f"模型 {model_name} 缺少 api_key 配置",
                    "請設置 API key 或使用環境變數"
                )
        
        # 檢查模型名稱
        if "model" not in model_config:
            result.add_issue(
                ValidationLevel.WARNING,
                "models",
                f"{model_name}.model",
                f"模型 {model_name} 缺少 model 欄位",
                "建議指定具體的模型名稱"
            )
        
        # 檢查數值範圍
        if "temperature" in model_config:
            temp = model_config["temperature"]
            if not isinstance(temp, (int, float)) or not (0.0 <= temp <= 2.0):
                result.add_issue(
                    ValidationLevel.WARNING,
                    "models",
                    f"{model_name}.temperature",
                    f"temperature 應該在 0.0-2.0 範圍內，當前值: {temp}"
                )
        
        if "max_tokens" in model_config:
            max_tokens = model_config["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens < 100:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "models",
                    f"{model_name}.max_tokens",
                    f"max_tokens 應該是大於 100 的整數，當前值: {max_tokens}"
                )
    
    def _validate_input_section(self, config: Dict[str, Any], result: ValidationResult):
        """驗證輸入配置區段"""
        if "input" not in config:
            result.add_issue(
                ValidationLevel.ERROR,
                "input",
                "missing_section",
                "缺少 input 配置區段"
            )
            return
        
        input_config = config["input"]
        required_fields = ["type", "base_dir"]
        
        for field in required_fields:
            if field not in input_config:
                result.add_issue(
                    ValidationLevel.ERROR,
                    "input",
                    field,
                    f"input 配置缺少必需欄位: {field}"
                )
        
        # 檢查輸入類型
        if "type" in input_config:
            input_type = input_config["type"]
            if input_type not in ["file", "blob"]:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "input",
                    "type",
                    f"未知的輸入類型: {input_type}",
                    "支援的類型: file, blob"
                )
        
        # 檢查編碼
        if "file_encoding" in input_config:
            encoding = input_config["file_encoding"]
            try:
                "test".encode(encoding)
            except LookupError:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "input",
                    "file_encoding",
                    f"不支援的編碼格式: {encoding}",
                    "建議使用 utf-8"
                )
    
    def _validate_chunks_section(self, config: Dict[str, Any], result: ValidationResult):
        """驗證文本分塊配置區段"""
        if "chunks" not in config:
            result.add_issue(
                ValidationLevel.WARNING,
                "chunks",
                "missing_section",
                "缺少 chunks 配置區段",
                "將使用預設的分塊設定"
            )
            return
        
        chunks_config = config["chunks"]
        
        # 檢查分塊大小
        if "size" in chunks_config:
            size = chunks_config["size"]
            if not isinstance(size, int) or size < 100 or size > 10000:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "chunks",
                    "size",
                    f"分塊大小應該在 100-10000 範圍內，當前值: {size}",
                    "建議使用 800-1200 的值"
                )
        
        # 檢查重疊
        if "overlap" in chunks_config:
            overlap = chunks_config["overlap"]
            if not isinstance(overlap, int) or overlap < 0:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "chunks",
                    "overlap",
                    f"重疊大小應該是非負整數，當前值: {overlap}"
                )
            
            # 檢查重疊與分塊大小的關係
            if "size" in chunks_config and overlap >= chunks_config["size"]:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "chunks",
                    "overlap",
                    "重疊大小不應該大於或等於分塊大小",
                    "建議重疊大小為分塊大小的 10-30%"
                )
    
    def _validate_storage_section(self, config: Dict[str, Any], result: ValidationResult):
        """驗證儲存配置區段"""
        for section_name in ["storage", "cache"]:
            if section_name not in config:
                result.add_issue(
                    ValidationLevel.WARNING,
                    section_name,
                    "missing_section",
                    f"缺少 {section_name} 配置區段"
                )
                continue
            
            section_config = config[section_name]
            if "type" not in section_config:
                result.add_issue(
                    ValidationLevel.ERROR,
                    section_name,
                    "type",
                    f"{section_name} 配置缺少 type 欄位"
                )
            
            if "base_dir" not in section_config:
                result.add_issue(
                    ValidationLevel.ERROR,
                    section_name,
                    "base_dir",
                    f"{section_name} 配置缺少 base_dir 欄位"
                )
    
    def _validate_cache_section(self, config: Dict[str, Any], result: ValidationResult):
        """驗證快取配置區段"""
        # 已在 _validate_storage_section 中處理
        pass
    
    def _validate_workflows_section(self, config: Dict[str, Any], result: ValidationResult):
        """驗證工作流程配置區段"""
        if "workflows" not in config:
            result.add_issue(
                ValidationLevel.INFO,
                "workflows",
                "missing_section",
                "缺少 workflows 配置，將使用預設工作流程"
            )
            return
        
        workflows = config["workflows"]
        if not isinstance(workflows, list):
            result.add_issue(
                ValidationLevel.ERROR,
                "workflows",
                "invalid_type",
                "workflows 必須是列表格式"
            )
            return
        
        # 檢查必需的工作流程
        required_workflows = ["create_base_text_units", "extract_graph"]
        for workflow in required_workflows:
            if workflow not in workflows:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "workflows",
                    workflow,
                    f"建議包含工作流程: {workflow}"
                )
    
    def _validate_chinese_processing_section(self, config: Dict[str, Any], result: ValidationResult):
        """驗證中文處理配置區段"""
        if "chinese_processing" not in config:
            result.add_issue(
                ValidationLevel.INFO,
                "chinese_processing",
                "missing_section",
                "缺少中文處理專用配置",
                "建議添加中文處理優化配置"
            )
            return
        
        chinese_config = config["chinese_processing"]
        
        # 檢查分詞器配置
        if "tokenizer" in chinese_config:
            tokenizer = chinese_config["tokenizer"]
            if tokenizer not in ["jieba", "ltp", "stanza"]:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "chinese_processing",
                    "tokenizer",
                    f"未知的中文分詞器: {tokenizer}",
                    "建議使用 jieba"
                )
        
        # 檢查實體識別配置
        if "entity_recognition" in chinese_config:
            er_config = chinese_config["entity_recognition"]
            if not isinstance(er_config, dict):
                result.add_issue(
                    ValidationLevel.WARNING,
                    "chinese_processing",
                    "entity_recognition",
                    "實體識別配置應該是字典格式"
                )
    
    def _validate_performance_section(self, config: Dict[str, Any], result: ValidationResult):
        """驗證效能配置區段"""
        if "performance" not in config:
            result.add_issue(
                ValidationLevel.INFO,
                "performance",
                "missing_section",
                "缺少效能配置",
                "建議添加效能優化配置"
            )
            return
        
        performance_config = config["performance"]
        
        # 檢查並行設定
        if "max_workers" in performance_config:
            max_workers = performance_config["max_workers"]
            if not isinstance(max_workers, int) or max_workers < 1:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "performance",
                    "max_workers",
                    f"max_workers 應該是正整數，當前值: {max_workers}"
                )
    
    def _validate_environment_variables(self, config: Dict[str, Any], result: ValidationResult):
        """驗證環境變數配置"""
        # 檢查常用的環境變數
        env_vars_to_check = ["GRAPHRAG_API_KEY", "OPENAI_API_KEY"]
        
        for env_var in env_vars_to_check:
            if not os.getenv(env_var):
                # 檢查配置中是否使用了這個環境變數
                config_str = str(config)
                if f"${{{env_var}}}" in config_str or f"${env_var}" in config_str:
                    result.add_issue(
                        ValidationLevel.WARNING,
                        "environment",
                        env_var,
                        f"環境變數 {env_var} 未設置",
                        f"請設置環境變數或在配置中提供實際值"
                    )
    
    def validate_config_compatibility(self, config: Dict[str, Any]) -> ValidationResult:
        """
        驗證配置的相容性
        
        Args:
            config: 配置字典
            
        Returns:
            ValidationResult: 驗證結果
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
        
        # 檢查模型相容性
        self._check_model_compatibility(config, result)
        
        # 檢查存儲相容性
        self._check_storage_compatibility(config, result)
        
        # 檢查中文處理相容性
        self._check_chinese_processing_compatibility(config, result)
        
        return result
    
    def _check_model_compatibility(self, config: Dict[str, Any], result: ValidationResult):
        """檢查模型相容性"""
        if "models" not in config:
            return
        
        models = config["models"]
        
        # 檢查 embedding 模型與 chat 模型的相容性
        chat_model = models.get("default_chat_model", {})
        embedding_model = models.get("default_embedding_model", {})
        
        chat_type = chat_model.get("type", "")
        embedding_type = embedding_model.get("type", "")
        
        # 建議使用相同提供商的模型
        if "openai" in chat_type and "openai" not in embedding_type:
            result.add_issue(
                ValidationLevel.INFO,
                "models",
                "compatibility",
                "建議聊天模型和 embedding 模型使用相同提供商",
                "這可以提供更好的整合體驗"
            )
    
    def _check_storage_compatibility(self, config: Dict[str, Any], result: ValidationResult):
        """檢查存儲相容性"""
        # 檢查存儲路徑是否衝突
        storage_config = config.get("storage", {})
        cache_config = config.get("cache", {})
        
        storage_dir = storage_config.get("base_dir", "")
        cache_dir = cache_config.get("base_dir", "")
        
        if storage_dir and cache_dir and storage_dir == cache_dir:
            result.add_issue(
                ValidationLevel.WARNING,
                "storage",
                "path_conflict",
                "存儲目錄和快取目錄不應該相同",
                "建議使用不同的目錄以避免檔案衝突"
            )
    
    def _check_chinese_processing_compatibility(self, config: Dict[str, Any], result: ValidationResult):
        """檢查中文處理相容性"""
        chinese_config = config.get("chinese_processing", {})
        
        if not chinese_config:
            return
        
        # 檢查分詞器與實體識別的相容性
        tokenizer = chinese_config.get("tokenizer", "")
        er_config = chinese_config.get("entity_recognition", {})
        
        if tokenizer == "ltp" and er_config.get("enabled", False):
            if "ltp" not in er_config.get("engines", []):
                result.add_issue(
                    ValidationLevel.INFO,
                    "chinese_processing",
                    "compatibility",
                    "使用 LTP 分詞器時建議也使用 LTP 進行實體識別",
                    "這可以提供更一致的處理結果"
                )