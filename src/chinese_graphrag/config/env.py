"""環境變數管理系統。

本模組提供環境變數的讀取、驗證和管理功能，包括：
- 環境變數讀取和類型轉換
- 預設值處理
- 環境變數驗證
- .env 檔案載入
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

T = TypeVar("T")


class EnvVarError(Exception):
    """環境變數相關錯誤。"""

    pass


class EnvVarConfig(BaseModel):
    """環境變數配置模型。"""

    name: str = Field(description="環境變數名稱")
    description: str = Field(description="環境變數描述")
    required: bool = Field(default=True, description="是否必需")
    default: Optional[Any] = Field(default=None, description="預設值")
    var_type: str = Field(default="str", description="變數類型")
    allowed_values: Optional[List[str]] = Field(
        default=None, description="允許的值列表"
    )


class EnvironmentManager:
    """環境變數管理器。"""

    def __init__(self, env_file: Optional[Path] = None):
        """初始化環境變數管理器。

        Args:
            env_file: .env 檔案路徑，預設為 .env
        """
        self.env_file = env_file or Path(".env")
        self.loaded_vars: Dict[str, str] = {}
        self._load_env_file()

    def _load_env_file(self) -> None:
        """載入 .env 檔案。"""
        if not self.env_file.exists():
            return

        try:
            with open(self.env_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # 跳過空行和註解
                    if not line or line.startswith("#"):
                        continue

                    # 解析變數定義
                    if "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # 移除引號
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]

                    # 設定環境變數（如果尚未設定）
                    if key not in os.environ:
                        os.environ[key] = value

                    self.loaded_vars[key] = value

        except Exception as e:
            raise EnvVarError(f"載入 .env 檔案失敗: {e}")

    def get_var(
        self,
        name: str,
        default: Optional[T] = None,
        var_type: Type[T] = str,
        required: bool = True,
    ) -> Optional[T]:
        """取得環境變數值。

        Args:
            name: 環境變數名稱
            default: 預設值
            var_type: 變數類型
            required: 是否必需

        Returns:
            環境變數值（已轉換為指定類型）

        Raises:
            EnvVarError: 當必需的環境變數未設定時
        """
        value = os.getenv(name)

        if value is None:
            if required and default is None:
                raise EnvVarError(f"必需的環境變數 '{name}' 未設定")
            return default

        # 類型轉換
        try:
            return self._convert_type(value, var_type)
        except (ValueError, TypeError) as e:
            raise EnvVarError(f"環境變數 '{name}' 類型轉換失敗: {e}")

    def _convert_type(self, value: str, var_type: Type[T]) -> T:
        """轉換環境變數值為指定類型。

        Args:
            value: 環境變數值（字串）
            var_type: 目標類型

        Returns:
            轉換後的值
        """
        if var_type == str:
            return value
        elif var_type == int:
            return int(value)
        elif var_type == float:
            return float(value)
        elif var_type == bool:
            return value.lower() in ("true", "1", "yes", "on", "enabled")
        elif var_type == list:
            # 支援逗號分隔的列表
            return [item.strip() for item in value.split(",") if item.strip()]
        else:
            # 嘗試直接轉換
            return var_type(value)

    def validate_var(self, name: str, config: EnvVarConfig) -> bool:
        """驗證環境變數。

        Args:
            name: 環境變數名稱
            config: 環境變數配置

        Returns:
            是否驗證通過

        Raises:
            EnvVarError: 驗證失敗時
        """
        value = os.getenv(name)

        # 檢查必需性
        if config.required and value is None:
            raise EnvVarError(f"必需的環境變數 '{name}' 未設定")

        if value is None:
            return True

        # 檢查允許的值
        if config.allowed_values and value not in config.allowed_values:
            raise EnvVarError(
                f"環境變數 '{name}' 的值 '{value}' 不在允許的值列表中: "
                f"{config.allowed_values}"
            )

        # 類型驗證
        try:
            type_map = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
            }

            if config.var_type in type_map:
                self._convert_type(value, type_map[config.var_type])

        except (ValueError, TypeError) as e:
            raise EnvVarError(f"環境變數 '{name}' 類型驗證失敗: {e}")

        return True

    def get_all_vars(self) -> Dict[str, str]:
        """取得所有環境變數。

        Returns:
            環境變數字典
        """
        return dict(os.environ)

    def set_var(self, name: str, value: str) -> None:
        """設定環境變數。

        Args:
            name: 環境變數名稱
            value: 環境變數值
        """
        os.environ[name] = value

    def unset_var(self, name: str) -> None:
        """取消設定環境變數。

        Args:
            name: 環境變數名稱
        """
        if name in os.environ:
            del os.environ[name]

    def create_env_template(
        self, configs: Dict[str, EnvVarConfig], output_path: Optional[Path] = None
    ) -> Path:
        """建立 .env 範本檔案。

        Args:
            configs: 環境變數配置字典
            output_path: 輸出路徑，預設為 .env.template

        Returns:
            建立的範本檔案路徑
        """
        output_path = output_path or Path(".env.template")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Chinese GraphRAG 環境變數配置範本\n")
            f.write("# 請複製此檔案為 .env 並填入實際值\n\n")

            for name, config in configs.items():
                f.write(f"# {config.description}\n")
                f.write(f"# 類型: {config.var_type}\n")
                f.write(f"# 必需: {'是' if config.required else '否'}\n")

                if config.allowed_values:
                    f.write(f"# 允許的值: {', '.join(config.allowed_values)}\n")

                if config.default is not None:
                    f.write(f"{name}={config.default}\n")
                else:
                    f.write(f"# {name}=\n")

                f.write("\n")

        return output_path


# 預定義的系統環境變數配置
SYSTEM_ENV_VARS = {
    "GRAPHRAG_API_KEY": EnvVarConfig(
        name="GRAPHRAG_API_KEY",
        description="GraphRAG API 金鑰",
        required=False,
        default=None,
        var_type="str",
    ),
    "GRAPHRAG_API_BASE": EnvVarConfig(
        name="GRAPHRAG_API_BASE",
        description="GraphRAG API 基礎 URL",
        required=False,
        default=None,
        var_type="str",
    ),
    "GRAPHRAG_MODEL": EnvVarConfig(
        name="GRAPHRAG_MODEL",
        description="預設 LLM 模型",
        required=False,
        default="gpt-4.1-mini",
        var_type="str",
    ),
    "GRAPHRAG_EMBEDDING_MODEL": EnvVarConfig(
        name="GRAPHRAG_EMBEDDING_MODEL",
        description="預設 Embedding 模型",
        required=False,
        default="BAAI/bge-m3",
        var_type="str",
    ),
    "GRAPHRAG_TEMPERATURE": EnvVarConfig(
        name="GRAPHRAG_TEMPERATURE",
        description="LLM 溫度參數",
        required=False,
        default="0.7",
        var_type="float",
    ),
    "GRAPHRAG_MAX_TOKENS": EnvVarConfig(
        name="GRAPHRAG_MAX_TOKENS",
        description="LLM 最大標記數",
        required=False,
        default="4000",
        var_type="int",
    ),
    "GRAPHRAG_DEVICE": EnvVarConfig(
        name="GRAPHRAG_DEVICE",
        description="計算設備",
        required=False,
        default="auto",
        var_type="str",
        allowed_values=["auto", "cpu", "cuda", "mps"],
    ),
    "GRAPHRAG_DEBUG": EnvVarConfig(
        name="GRAPHRAG_DEBUG",
        description="除錯模式",
        required=False,
        default="false",
        var_type="bool",
    ),
    "GRAPHRAG_LOG_LEVEL": EnvVarConfig(
        name="GRAPHRAG_LOG_LEVEL",
        description="日誌級別",
        required=False,
        default="INFO",
        var_type="str",
        allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    ),
    "GRAPHRAG_VECTOR_DB_PATH": EnvVarConfig(
        name="GRAPHRAG_VECTOR_DB_PATH",
        description="向量資料庫路徑",
        required=False,
        default="./data/vector_db",
        var_type="str",
    ),
    "GRAPHRAG_OUTPUT_DIR": EnvVarConfig(
        name="GRAPHRAG_OUTPUT_DIR",
        description="輸出目錄",
        required=False,
        default="./output",
        var_type="str",
    ),
    "GRAPHRAG_CACHE_DIR": EnvVarConfig(
        name="GRAPHRAG_CACHE_DIR",
        description="快取目錄",
        required=False,
        default="./cache",
        var_type="str",
    ),
}


# 全域環境變數管理器實例
env_manager = EnvironmentManager()


def get_env_var(
    name: str,
    default: Optional[T] = None,
    var_type: Type[T] = str,
    required: bool = True,
) -> Optional[T]:
    """便利函數：取得環境變數。

    Args:
        name: 環境變數名稱
        default: 預設值
        var_type: 變數類型
        required: 是否必需

    Returns:
        環境變數值
    """
    return env_manager.get_var(name, default, var_type, required)


def validate_system_env_vars() -> bool:
    """驗證系統環境變數。

    Returns:
        是否全部驗證通過

    Raises:
        EnvVarError: 驗證失敗時
    """
    for name, config in SYSTEM_ENV_VARS.items():
        env_manager.validate_var(name, config)

    return True
