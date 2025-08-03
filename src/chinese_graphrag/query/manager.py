"""
LLM 管理和適配器

提供統一的 LLM 管理介面，支援多種 LLM 提供者，
包括效能監控、負載均衡、中文 prompt 優化等功能。
"""

import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum
from loguru import logger

from chinese_graphrag.llm import LLM, create_llm


class LLMProvider(str, Enum):
    """LLM 提供者類型"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    MOCK = "mock"


class TaskType(str, Enum):
    """任務類型，用於選擇合適的 LLM"""
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    COMMUNITY_REPORT = "community_report"
    GLOBAL_SEARCH = "global_search"
    LOCAL_SEARCH = "local_search"
    GENERAL_QA = "general_qa"


@dataclass
class LLMMetrics:
    """LLM 效能指標"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """失敗率"""
        return 1.0 - self.success_rate
    
    def update_metrics(self, success: bool, tokens: int, cost: float, response_time: float):
        """更新指標"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_tokens += tokens
        self.total_cost += cost
        self.response_times.append(response_time)
        
        # 計算平均回應時間
        self.avg_response_time = sum(self.response_times) / len(self.response_times)


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: LLMProvider
    model: str
    config: Dict[str, Any]
    max_tokens: int = 4000
    temperature: float = 0.7
    max_retries: int = 3
    timeout: float = 60.0
    cost_per_token: float = 0.0
    weight: float = 1.0  # 負載均衡權重
    task_types: List[TaskType] = field(default_factory=list)  # 支援的任務類型


class LLMAdapter(ABC):
    """LLM 適配器抽象基類"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm: LLM = create_llm(config.provider.value, config.config)
        self.metrics = LLMMetrics()
        self._is_healthy = True
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        task_type: TaskType,
        **kwargs
    ) -> str:
        """生成回應"""
        pass
    
    async def health_check(self) -> bool:
        """健康檢查"""
        try:
            test_prompt = "測試"
            await asyncio.wait_for(
                self.llm.async_generate(test_prompt, max_tokens=10),
                timeout=10.0
            )
            self._is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"LLM {self.config.model} 健康檢查失敗: {e}")
            self._is_healthy = False
            return False
    
    @property
    def is_healthy(self) -> bool:
        """是否健康"""
        return self._is_healthy
    
    def supports_task(self, task_type: TaskType) -> bool:
        """是否支援特定任務類型"""
        return not self.config.task_types or task_type in self.config.task_types


class OpenAIAdapter(LLMAdapter):
    """OpenAI 適配器"""
    
    async def generate(
        self, 
        prompt: str, 
        task_type: TaskType,
        **kwargs
    ) -> str:
        """使用 OpenAI 生成回應"""
        start_time = time.time()
        success = False
        tokens = 0
        cost = 0.0
        
        try:
            # 根據任務類型調整參數
            max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
            temperature = kwargs.pop("temperature", self.config.temperature)
            
            # 針對不同任務類型優化 prompt
            optimized_prompt = self._optimize_prompt_for_task(prompt, task_type)
            
            response = await self.llm.async_generate(
                optimized_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # 估算 token 數量和成本
            tokens = len(response.split()) * 1.3  # 粗略估算
            cost = tokens * self.config.cost_per_token
            success = True
            
            return response
            
        except Exception as e:
            logger.error(f"OpenAI 生成失敗: {e}")
            raise
        finally:
            response_time = time.time() - start_time
            self.metrics.update_metrics(success, int(tokens), cost, response_time)
    
    def _optimize_prompt_for_task(self, prompt: str, task_type: TaskType) -> str:
        """根據任務類型優化 prompt"""
        task_prefixes = {
            TaskType.ENTITY_EXTRACTION: "請仔細分析以下中文文本，提取其中的實體：\n\n",
            TaskType.RELATIONSHIP_EXTRACTION: "請分析以下中文文本中實體之間的關係：\n\n",
            TaskType.COMMUNITY_REPORT: "請為以下社群生成詳細的中文摘要報告：\n\n",
            TaskType.GLOBAL_SEARCH: "基於知識圖譜，請回答以下中文問題：\n\n",
            TaskType.LOCAL_SEARCH: "基於相關實體資訊，請回答以下中文問題：\n\n",
            TaskType.GENERAL_QA: "請回答以下中文問題：\n\n"
        }
        
        prefix = task_prefixes.get(task_type, "")
        return f"{prefix}{prompt}"


class AzureOpenAIAdapter(OpenAIAdapter):
    """Azure OpenAI 適配器"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # Azure OpenAI 特定設定
        self.deployment_name = config.config.get("deployment_name")
        self.api_version = config.config.get("api_version", "2023-12-01-preview")


class OllamaAdapter(LLMAdapter):
    """Ollama 本地模型適配器"""
    
    async def generate(
        self, 
        prompt: str, 
        task_type: TaskType,
        **kwargs
    ) -> str:
        """使用 Ollama 生成回應"""
        start_time = time.time()
        success = False
        tokens = 0
        cost = 0.0  # 本地模型無成本
        
        try:
            # 針對中文任務優化 prompt
            optimized_prompt = self._optimize_chinese_prompt(prompt, task_type)
            
            # 提取並移除已處理的參數，避免重複傳遞
            max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
            temperature = kwargs.pop("temperature", self.config.temperature)
            
            response = await self.llm.async_generate(
                optimized_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            tokens = len(response.split()) * 1.3
            success = True
            
            return response
            
        except Exception as e:
            logger.error(f"Ollama 生成失敗: {e}")
            raise
        finally:
            response_time = time.time() - start_time
            self.metrics.update_metrics(success, int(tokens), cost, response_time)
    
    def _optimize_chinese_prompt(self, prompt: str, task_type: TaskType) -> str:
        """針對中文優化 prompt"""
        chinese_instructions = {
            TaskType.ENTITY_EXTRACTION: "你是一個專業的中文實體識別專家。請仔細分析文本，識別人物、組織、地點、概念等實體。",
            TaskType.RELATIONSHIP_EXTRACTION: "你是一個專業的中文關係提取專家。請分析實體間的語義關係。",
            TaskType.GLOBAL_SEARCH: "你是一個專業的中文問答專家。請基於知識圖譜回答問題。",
            TaskType.LOCAL_SEARCH: "你是一個專業的中文資訊檢索專家。請基於相關資訊回答問題。"
        }
        
        instruction = chinese_instructions.get(task_type, "你是一個專業的中文助理。")
        return f"{instruction}\n\n{prompt}"


class MockAdapter(LLMAdapter):
    """模擬適配器，用於測試"""
    
    async def generate(
        self, 
        prompt: str, 
        task_type: TaskType,
        **kwargs
    ) -> str:
        """返回模擬回應"""
        start_time = time.time()
        
        # 模擬處理時間
        await asyncio.sleep(0.1)
        
        mock_responses = {
            TaskType.ENTITY_EXTRACTION: '{"entities": [{"name": "測試實體", "type": "概念"}]}',
            TaskType.RELATIONSHIP_EXTRACTION: '{"relationships": [{"source": "A", "target": "B", "description": "相關"}]}',
            TaskType.GLOBAL_SEARCH: "這是一個全域搜尋的模擬回答。",
            TaskType.LOCAL_SEARCH: "這是一個本地搜尋的模擬回答。",
            TaskType.GENERAL_QA: "這是一個一般問答的模擬回答。"
        }
        
        response = mock_responses.get(task_type, "模擬回應")
        
        response_time = time.time() - start_time
        self.metrics.update_metrics(True, len(response.split()), 0.0, response_time)
        
        return response


class LLMManager:
    """LLM 管理器
    
    提供 LLM 的統一管理介面，包括：
    - 多 LLM 提供者支援
    - 智慧負載均衡
    - 效能監控
    - 健康檢查
    - 自動重試和降級
    """
    
    def __init__(self, configs: List[LLMConfig]):
        """
        初始化 LLM 管理器
        
        Args:
            configs: LLM 配置列表
        """
        self.adapters: Dict[str, LLMAdapter] = {}
        self._init_adapters(configs)
        self._adapter_classes = {
            LLMProvider.OPENAI: OpenAIAdapter,
            LLMProvider.AZURE_OPENAI: AzureOpenAIAdapter,
            LLMProvider.OLLAMA: OllamaAdapter,
            LLMProvider.MOCK: MockAdapter,
        }
    
    def _init_adapters(self, configs: List[LLMConfig]):
        """初始化適配器"""
        for config in configs:
            adapter_class = self._get_adapter_class(config.provider)
            adapter = adapter_class(config)
            self.adapters[f"{config.provider}_{config.model}"] = adapter
            
            logger.info(f"初始化 LLM 適配器: {config.provider}_{config.model}")
    
    def _get_adapter_class(self, provider: LLMProvider) -> Type[LLMAdapter]:
        """獲取適配器類別"""
        adapter_classes = {
            LLMProvider.OPENAI: OpenAIAdapter,
            LLMProvider.AZURE_OPENAI: AzureOpenAIAdapter,
            LLMProvider.OLLAMA: OllamaAdapter,
            LLMProvider.MOCK: MockAdapter,
        }
        return adapter_classes[provider]
    
    async def generate(
        self,
        prompt: str,
        task_type: TaskType,
        preferred_provider: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        生成回應
        
        Args:
            prompt: 輸入提示詞
            task_type: 任務類型
            preferred_provider: 優先使用的提供者
            **kwargs: 其他參數
            
        Returns:
            生成的回應
        """
        # 選擇最佳適配器
        adapter = self._select_best_adapter(task_type, preferred_provider)
        
        if not adapter:
            raise RuntimeError(f"沒有可用的 LLM 適配器支援任務類型: {task_type}")
        
        # 嘗試生成回應
        max_retries = adapter.config.max_retries
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"使用 {adapter.config.model} 生成回應 (嘗試 {attempt + 1})")
                
                response = await adapter.generate(prompt, task_type, **kwargs)
                
                logger.info(f"成功生成回應，使用模型: {adapter.config.model}")
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM 生成失敗 (嘗試 {attempt + 1}): {e}")
                
                if attempt < max_retries:
                    # 等待後重試
                    await asyncio.sleep(2 ** attempt)
                else:
                    # 嘗試降級到其他適配器
                    fallback_adapter = self._get_fallback_adapter(adapter, task_type)
                    if fallback_adapter:
                        logger.info(f"降級到 {fallback_adapter.config.model}")
                        try:
                            return await fallback_adapter.generate(prompt, task_type, **kwargs)
                        except Exception as fallback_e:
                            logger.error(f"降級適配器也失敗: {fallback_e}")
        
        raise RuntimeError(f"所有 LLM 適配器都失敗: {last_exception}")
    
    def _select_best_adapter(
        self, 
        task_type: TaskType, 
        preferred_provider: Optional[str] = None
    ) -> Optional[LLMAdapter]:
        """選擇最佳適配器"""
        # 過濾支援該任務類型且健康的適配器
        candidates = [
            adapter for adapter in self.adapters.values()
            if adapter.supports_task(task_type) and adapter.is_healthy
        ]
        
        if not candidates:
            return None
        
        # 如果指定了優先提供者，優先選擇
        if preferred_provider:
            for adapter in candidates:
                if adapter.config.provider.value == preferred_provider:
                    return adapter
        
        # 根據權重和效能選擇最佳適配器
        def score_adapter(adapter: LLMAdapter) -> float:
            """計算適配器評分"""
            base_score = adapter.config.weight
            
            # 考慮成功率
            success_rate_bonus = adapter.metrics.success_rate * 0.3
            
            # 考慮回應時間（越快越好）
            if adapter.metrics.avg_response_time > 0:
                response_time_penalty = min(adapter.metrics.avg_response_time / 10.0, 0.5)
            else:
                response_time_penalty = 0
            
            return base_score + success_rate_bonus - response_time_penalty
        
        return max(candidates, key=score_adapter)
    
    def _get_fallback_adapter(
        self, 
        failed_adapter: LLMAdapter, 
        task_type: TaskType
    ) -> Optional[LLMAdapter]:
        """獲取降級適配器"""
        candidates = [
            adapter for adapter in self.adapters.values()
            if (adapter != failed_adapter and 
                adapter.supports_task(task_type) and 
                adapter.is_healthy)
        ]
        
        if not candidates:
            return None
        
        # 優先選擇 Mock 適配器作為最後的降級選項
        for adapter in candidates:
            if adapter.config.provider == LLMProvider.MOCK:
                return adapter
        
        return candidates[0]
    
    async def health_check_all(self) -> Dict[str, bool]:
        """檢查所有適配器的健康狀態"""
        results = {}
        tasks = []
        
        for name, adapter in self.adapters.items():
            tasks.append(self._health_check_single(name, adapter))
        
        health_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (name, _) in enumerate(self.adapters.items()):
            result = health_results[i]
            if isinstance(result, Exception):
                results[name] = False
                logger.error(f"健康檢查異常 {name}: {result}")
            else:
                results[name] = result
        
        return results
    
    async def _health_check_single(self, name: str, adapter: LLMAdapter) -> bool:
        """單個適配器健康檢查"""
        try:
            return await adapter.health_check()
        except Exception as e:
            logger.error(f"適配器 {name} 健康檢查失敗: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, LLMMetrics]:
        """獲取所有適配器的效能指標"""
        return {
            name: adapter.metrics 
            for name, adapter in self.adapters.items()
        }
    
    def get_adapter_info(self) -> Dict[str, Dict[str, Any]]:
        """獲取適配器資訊"""
        info = {}
        for name, adapter in self.adapters.items():
            info[name] = {
                "provider": adapter.config.provider.value,
                "model": adapter.config.model,
                "is_healthy": adapter.is_healthy,
                "supported_tasks": [task.value for task in adapter.config.task_types],
                "metrics": {
                    "total_requests": adapter.metrics.total_requests,
                    "success_rate": adapter.metrics.success_rate,
                    "avg_response_time": adapter.metrics.avg_response_time,
                    "total_cost": adapter.metrics.total_cost,
                }
            }
        return info