"""
GraphRAG 配置適配器

將中文 GraphRAG 配置轉換為標準 Microsoft GraphRAG 配置格式
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from loguru import logger

from chinese_graphrag.config.models import GraphRAGConfig


class GraphRAGConfigAdapter:
    """
    GraphRAG 配置適配器

    負責將中文 GraphRAG 系統的配置轉換為 Microsoft GraphRAG 標準配置格式，
    使得系統可以正確使用官方的 GraphRAG workflows。
    """

    def __init__(self, chinese_config: GraphRAGConfig):
        """初始化 GraphRAG 配置適配器

        Args:
            chinese_config: 中文 GraphRAG 配置物件
        """
        self.chinese_config = chinese_config
        # 初始化驗證器
        from .graphrag_validator import GraphRAGConfigValidator

        self.validator = GraphRAGConfigValidator()

    def to_graphrag_settings_dict(self) -> Dict[str, Any]:
        """
        轉換為 Microsoft GraphRAG settings.yaml 格式

        Returns:
            Dict: 標準 GraphRAG 配置字典
        """
        try:
            logger.info("開始轉換中文 GraphRAG 配置為標準格式")

            # 基礎配置結構
            settings = {
                "models": self._convert_models_config(),
                "input": self._convert_input_config(),
                "chunks": self._convert_chunks_config(),
                "storage": self._convert_storage_config(),
                "cache": self._convert_cache_config(),
                "reporting": self._convert_reporting_config(),
                "workflows": self._get_default_workflows(),
            }

            # 添加中文處理特定配置
            settings["chinese_processing"] = self._get_chinese_processing_config()

            logger.info("配置轉換完成")
            return settings

        except Exception as e:
            logger.error(f"配置轉換失敗: {e}")
            raise

    def _convert_models_config(self) -> Dict[str, Any]:
        """轉換模型配置"""
        models = {}

        # 預設聊天模型
        if hasattr(self.chinese_config, "models") and self.chinese_config.models:
            for model_name, model_config in self.chinese_config.models.items():
                if hasattr(model_config, "type"):
                    if model_config.type == "openai":
                        models["default_chat_model"] = {
                            "type": "openai_chat",
                            "api_key": "${GRAPHRAG_API_KEY}",
                            "model": getattr(model_config, "model", "gpt-4.1"),
                            "model_supports_json": True,
                            "max_tokens": getattr(model_config, "max_tokens", 4000),
                            "temperature": getattr(model_config, "temperature", 0.1),
                        }
                    elif model_config.type == "ollama":
                        models["ollama_chat_model"] = {
                            "type": "ollama",
                            "model": getattr(model_config, "model", "gemma3"),
                            "api_base": getattr(
                                model_config, "base_url", "http://localhost:11434"
                            ),
                        }

        # 確保有預設聊天模型
        if "default_chat_model" not in models:
            models["default_chat_model"] = {
                "type": "openai_chat",
                "api_key": "${GRAPHRAG_API_KEY}",
                "model": "gpt-4.1",
                "model_supports_json": True,
                "max_tokens": 4000,
                "temperature": 0.1,
            }

        # Embedding 模型
        if hasattr(self.chinese_config, "embeddings"):
            embedding_config = self.chinese_config.embeddings
            if hasattr(embedding_config, "type"):
                if embedding_config.type == "bge_m3":
                    models["default_embedding_model"] = {
                        "type": "sentence_transformers",
                        "model": getattr(embedding_config, "model", "BAAI/bge-m3"),
                        "device": getattr(embedding_config, "device", "auto"),
                    }
                elif embedding_config.type == "openai":
                    models["default_embedding_model"] = {
                        "type": "openai_embedding",
                        "api_key": "${GRAPHRAG_API_KEY}",
                        "model": getattr(
                            embedding_config, "model", "text-embedding-3-small"
                        ),
                    }

        # 確保有預設 embedding 模型
        if "default_embedding_model" not in models:
            models["default_embedding_model"] = {
                "type": "sentence_transformers",
                "model": "BAAI/bge-m3",
                "device": "auto",
            }

        return models

    def _convert_input_config(self) -> Dict[str, Any]:
        """轉換輸入配置"""
        return {
            "type": "file",
            "file_type": "text",
            "base_dir": "input",
            "file_encoding": "utf-8",
        }

    def _convert_chunks_config(self) -> Dict[str, Any]:
        """轉換文本分塊配置"""
        chunks_config = {"size": 1000, "overlap": 200, "group_by_columns": ["id"]}

        if hasattr(self.chinese_config, "chunks"):
            chunk_settings = self.chinese_config.chunks
            chunks_config.update(
                {
                    "size": getattr(chunk_settings, "size", 1000),
                    "overlap": getattr(chunk_settings, "overlap", 200),
                }
            )

        return chunks_config

    def _convert_storage_config(self) -> Dict[str, Any]:
        """轉換儲存配置"""
        storage_config = {"type": "file", "base_dir": "output"}

        if hasattr(self.chinese_config, "storage"):
            storage_settings = self.chinese_config.storage
            base_dir = getattr(storage_settings, "base_dir", "output")
            storage_config["base_dir"] = str(base_dir)

        return storage_config

    def _convert_cache_config(self) -> Dict[str, Any]:
        """轉換快取配置"""
        return {"type": "file", "base_dir": "cache"}

    def _convert_reporting_config(self) -> Dict[str, Any]:
        """轉換報告配置"""
        return {"type": "file", "base_dir": "reporting"}

    def _get_default_workflows(self) -> list:
        """獲取預設工作流程列表"""
        return [
            "create_base_text_units",
            "extract_graph",
            "create_communities",
            "create_community_reports",
            "generate_text_embeddings",
        ]

    def _get_chinese_processing_config(self) -> Dict[str, Any]:
        """獲取中文處理特定配置"""
        return {
            "tokenizer": "jieba",
            "enable_traditional_chinese": True,
            "enable_pos_tagging": True,
            "enable_ner": True,
        }

    def create_graphrag_config_file(self, output_path: Union[str, Path]) -> Path:
        """建立 GraphRAG 標準配置檔案

        Args:
            output_path: 輸出檔案路徑

        Returns:
            Path: 建立的配置檔案路徑

        Raises:
            ValueError: 當配置驗證失敗時
        """
        output_path = Path(output_path)

        # 轉換配置
        graphrag_config = self.to_graphrag_settings_dict()

        # 寫入檔案
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                graphrag_config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
            )

        logger.info(f"已建立 GraphRAG 配置檔案: {output_path}")

        # 驗證建立的配置檔案
        validation_result = self.validator.validate_settings_yaml(output_path)

        if not validation_result.is_valid:
            error_msg = f"建立的配置檔案驗證失敗: {validation_result.get_summary()}"
            logger.error(error_msg)

            # 顯示詳細錯誤
            if validation_result.errors:
                logger.error("配置錯誤:")
                for error in validation_result.errors:
                    logger.error(
                        f"  - {error['section']}.{error['key']}: {error['message']}"
                    )

            raise ValueError(error_msg)

        # 顯示警告和建議
        if validation_result.warnings:
            logger.warning("配置警告:")
            for warning in validation_result.warnings:
                logger.warning(
                    f"  - {warning['section']}.{warning['key']}: {warning['message']}"
                )

        if validation_result.info:
            logger.info("配置建議:")
            for info in validation_result.info:
                logger.info(f"  - {info['section']}.{info['key']}: {info['message']}")

        logger.success(f"配置檔案驗證成功: {validation_result.get_summary()}")

        return output_path

    def validate_and_prepare_environment(self, root_dir: Path) -> bool:
        """
        驗證並準備 GraphRAG 執行環境

        Args:
            root_dir: GraphRAG 項目根目錄

        Returns:
            bool: 環境準備是否成功
        """
        try:
            logger.info(f"準備 GraphRAG 執行環境: {root_dir}")

            # 建立必要目錄
            directories = ["input", "output", "cache", "reporting"]
            for dir_name in directories:
                dir_path = root_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"建立目錄: {dir_path}")

            # 生成配置檔案
            settings_path = root_dir / "settings.yaml"
            self.create_graphrag_config_file(settings_path)

            # 檢查環境變數
            api_key = os.getenv("GRAPHRAG_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("未設置 GRAPHRAG_API_KEY 或 OPENAI_API_KEY 環境變數")
                # 對於測試環境，設置一個假的 API key
                os.environ["GRAPHRAG_API_KEY"] = "test-key-for-demo"

            logger.info("GraphRAG 執行環境準備完成")
            return True

        except Exception as e:
            logger.error(f"準備環境失敗: {e}")
            return False

    def convert_documents_to_graphrag_format(
        self, documents: list, output_dir: Path
    ) -> Path:
        """
        將文檔數據轉換為 GraphRAG 需要的 parquet 格式

        Args:
            documents: 文檔列表
            output_dir: 輸出目錄

        Returns:
            Path: documents.parquet 檔案路徑
        """
        try:
            import pandas as pd

            logger.info("轉換文檔為 GraphRAG 格式")

            # 轉換為 DataFrame
            document_data = []
            for doc in documents:
                if hasattr(doc, "to_dict"):
                    doc_dict = doc.to_dict()
                else:
                    doc_dict = {
                        "id": getattr(doc, "id", ""),
                        "title": getattr(doc, "title", ""),
                        "text": getattr(doc, "content", ""),
                        "metadata": getattr(doc, "metadata", {}),
                    }
                document_data.append(doc_dict)

            documents_df = pd.DataFrame(document_data)

            # 確保輸出目錄存在
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存為 parquet 格式
            output_path = output_dir / "documents.parquet"
            documents_df.to_parquet(output_path, index=False)

            logger.info(f"文檔已轉換並保存至: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"轉換文檔格式失敗: {e}")
            raise

    def validate_chinese_config(self) -> "ValidationResult":
        """驗證中文配置的有效性

        Returns:
            ValidationResult: 驗證結果
        """
        from .graphrag_validator import ValidationLevel, ValidationResult

        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        # 驗證基本模型配置
        if not hasattr(self.chinese_config, "models") or not self.chinese_config.models:
            result.add_issue(
                ValidationLevel.ERROR,
                "models",
                "missing",
                "缺少模型配置",
                "請配置至少一個聊天模型和 embedding 模型",
            )

        # 驗證是否有可用的 API 金鑰
        api_key_found = False
        if hasattr(self.chinese_config, "models") and self.chinese_config.models:
            for model_name, model_config in self.chinese_config.models.items():
                if hasattr(model_config, "api_key") and model_config.api_key:
                    api_key_found = True
                    break

        if not api_key_found:
            result.add_issue(
                ValidationLevel.WARNING,
                "models",
                "api_key",
                "未找到任何 API 金鑰配置",
                "請確保至少有一個模型配置了有效的 API 金鑰",
            )

        # 驗證向量存儲配置
        if (
            hasattr(self.chinese_config, "vector_store")
            and self.chinese_config.vector_store
        ):
            vs_config = self.chinese_config.vector_store
            if hasattr(vs_config, "db_uri") and vs_config.db_uri:
                result.add_issue(
                    ValidationLevel.INFO,
                    "vector_store",
                    "configured",
                    "已配置向量存儲",
                    "將保留現有向量存儲配置",
                )

        return result

    def validate_converted_config(
        self, config_dict: Dict[str, Any]
    ) -> "ValidationResult":
        """驗證轉換後的 GraphRAG 配置

        Args:
            config_dict: 轉換後的配置字典

        Returns:
            ValidationResult: 驗證結果
        """
        return self.validator.validate_config_compatibility(config_dict)
