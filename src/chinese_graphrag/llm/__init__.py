"""
中文 GraphRAG 系統 LLM 模組

提供統一的 LLM 介面，支援多種語言模型後端
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from loguru import logger


class LLM(ABC):
    """LLM 抽象基類"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 LLM
        
        Args:
            config: LLM 配置字典
        """
        self.config = config
        self.model_name = config.get("model", "unknown")
        
    @abstractmethod
    async def async_generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        異步生成文本
        
        Args:
            prompt: 輸入提示詞
            max_tokens: 最大令牌數
            temperature: 採樣溫度
            **kwargs: 其他參數
            
        Returns:
            生成的文本
        """
        pass
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        同步生成文本
        
        Args:
            prompt: 輸入提示詞
            max_tokens: 最大令牌數
            temperature: 採樣溫度
            **kwargs: 其他參數
            
        Returns:
            生成的文本
        """
        import asyncio
        return asyncio.run(self.async_generate(prompt, max_tokens, temperature, **kwargs))


class MockLLM(LLM):
    """模擬 LLM 實現，用於測試"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mock_response = config.get("mock_response", self._default_mock_response())
        
    def _default_mock_response(self) -> str:
        """預設的模擬回應"""
        return '''{
    "entities": [
        {"name": "測試實體", "type": "概念", "description": "這是一個測試實體"},
        {"name": "範例公司", "type": "組織", "description": "一家範例公司"}
    ],
    "relationships": [
        {"source": "測試實體", "target": "範例公司", "description": "測試實體與範例公司有關聯"}
    ]
}'''
    
    async def async_generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """返回模擬回應"""
        logger.info(f"MockLLM 收到提示詞: {prompt[:100]}...")
        return self.mock_response


class OpenAILLM(LLM):
    """OpenAI LLM 實現"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-3.5-turbo")
        
        if not self.api_key:
            raise ValueError("OpenAI API key 必須提供")
            
    async def async_generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """使用 OpenAI API 生成文本"""
        try:
            import openai
            
            # 設定 OpenAI 客戶端
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # 準備請求參數
            request_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or self.config.get("max_tokens", 4000),
                "temperature": temperature or self.config.get("temperature", 0.7),
            }
            request_params.update(kwargs)
            
            # 發送請求
            response = await client.chat.completions.create(**request_params)
            
            return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("需要安裝 openai 套件: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI API 呼叫失敗: {e}")
            raise


class OllamaLLM(LLM):
    """Ollama LLM 實現"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llama2")
        
    async def async_generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """使用 Ollama API 生成文本"""
        try:
            import aiohttp
            import json
            
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens or self.config.get("max_tokens", 4000),
                    "temperature": temperature or self.config.get("temperature", 0.7),
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API 錯誤 {response.status}: {error_text}")
                        
        except ImportError:
            raise ImportError("需要安裝 aiohttp 套件: pip install aiohttp")
        except Exception as e:
            logger.error(f"Ollama API 呼叫失敗: {e}")
            raise


# LLM 類型映射
LLM_REGISTRY = {
    "openai": OpenAILLM,
    "mock": MockLLM,
    "ollama": OllamaLLM,
}


def create_llm(llm_type: str, config: Dict[str, Any]) -> LLM:
    """
    建立 LLM 實例
    
    Args:
        llm_type: LLM 類型 ("openai", "mock", "ollama")
        config: LLM 配置
        
    Returns:
        LLM 實例
        
    Raises:
        ValueError: 不支援的 LLM 類型
    """
    if llm_type not in LLM_REGISTRY:
        supported_types = list(LLM_REGISTRY.keys())
        raise ValueError(f"不支援的 LLM 類型: {llm_type}. 支援的類型: {supported_types}")
    
    llm_class = LLM_REGISTRY[llm_type]
    return llm_class(config)


# 導出主要類別和函數
__all__ = [
    "LLM",
    "OpenAILLM", 
    "MockLLM",
    "OllamaLLM",
    "create_llm",
    "LLM_REGISTRY",
]