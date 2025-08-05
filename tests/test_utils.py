"""
測試工具模組

提供測試過程中常用的工具函數和 helper 類別。
"""

import asyncio
import json
import logging
import random
import string
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock, patch

import numpy as np


class TestDataGenerator:
    """測試資料生成器"""

    @staticmethod
    def generate_random_string(length: int = 10) -> str:
        """生成隨機字串"""
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def generate_chinese_text(length: int = 100) -> str:
        """生成中文測試文本"""
        chinese_words = [
            "人工智慧",
            "機器學習",
            "深度學習",
            "神經網路",
            "自然語言處理",
            "電腦視覺",
            "資料科學",
            "演算法",
            "模型訓練",
            "特徵提取",
            "分類",
            "迴歸",
            "聚類",
            "降維",
            "優化",
            "梯度下降",
            "反向傳播",
            "卷積",
            "池化",
            "全連接",
            "激活函數",
            "損失函數",
            "準確率",
            "召回率",
            "精確率",
            "F1分數",
            "交叉驗證",
            "過擬合",
        ]

        sentences = []
        current_length = 0

        while current_length < length:
            # 隨機選擇2-5個詞組成句子
            sentence_words = random.sample(chinese_words, random.randint(2, 5))
            sentence = "，".join(sentence_words) + "。"
            sentences.append(sentence)
            current_length += len(sentence)

        return "".join(sentences)[:length]

    @staticmethod
    def generate_vector(dimension: int = 768) -> List[float]:
        """生成隨機向量"""
        return np.random.rand(dimension).tolist()

    @staticmethod
    def generate_document(doc_id: Optional[str] = None) -> Dict[str, Any]:
        """生成測試文件"""
        if doc_id is None:
            doc_id = f"doc_{TestDataGenerator.generate_random_string(8)}"

        return {
            "id": doc_id,
            "title": f"測試文件標題 {TestDataGenerator.generate_random_string(5)}",
            "content": TestDataGenerator.generate_chinese_text(200),
            "metadata": {
                "author": f"作者{random.randint(1, 100)}",
                "date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "category": random.choice(["技術", "科學", "教育", "商業"]),
            },
        }

    @staticmethod
    def generate_entity(entity_id: Optional[str] = None) -> Dict[str, Any]:
        """生成測試實體"""
        if entity_id is None:
            entity_id = f"entity_{TestDataGenerator.generate_random_string(8)}"

        entity_types = ["人物", "組織", "概念", "技術", "產品", "地點"]

        return {
            "id": entity_id,
            "name": f"實體{TestDataGenerator.generate_random_string(3)}",
            "type": random.choice(entity_types),
            "description": TestDataGenerator.generate_chinese_text(50),
            "embedding": TestDataGenerator.generate_vector(),
        }

    @staticmethod
    def generate_relationship(
        source_id: Optional[str] = None, target_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成測試關係"""
        if source_id is None:
            source_id = f"entity_{TestDataGenerator.generate_random_string(8)}"
        if target_id is None:
            target_id = f"entity_{TestDataGenerator.generate_random_string(8)}"

        relation_types = ["包含", "屬於", "關聯", "影響", "依賴", "相似"]

        return {
            "id": f"rel_{TestDataGenerator.generate_random_string(8)}",
            "source": source_id,
            "target": target_id,
            "type": random.choice(relation_types),
            "description": TestDataGenerator.generate_chinese_text(30),
            "weight": random.uniform(0.1, 1.0),
        }


class MockFactory:
    """Mock 物件工廠"""

    @staticmethod
    def create_embedding_service(dimension: int = 768) -> Mock:
        """建立模擬 Embedding 服務"""
        mock = Mock()
        mock.encode.return_value = TestDataGenerator.generate_vector(dimension)
        mock.encode_batch.return_value = [
            TestDataGenerator.generate_vector(dimension) for _ in range(3)
        ]
        mock.get_dimension.return_value = dimension
        mock.is_available.return_value = True
        mock.model_name = "test-embedding-model"
        return mock

    @staticmethod
    def create_vector_store() -> Mock:
        """建立模擬向量資料庫"""
        mock = Mock()
        mock.add_documents.return_value = True
        mock.add_document.return_value = True
        mock.search.return_value = [
            {"id": "doc_1", "score": 0.95, "metadata": {"title": "測試文件1"}},
            {"id": "doc_2", "score": 0.87, "metadata": {"title": "測試文件2"}},
        ]
        mock.get_document.return_value = TestDataGenerator.generate_document()
        mock.delete_document.return_value = True
        mock.list_collections.return_value = ["test_collection"]
        mock.create_collection.return_value = True
        mock.drop_collection.return_value = True
        return mock

    @staticmethod
    def create_llm_service() -> Mock:
        """建立模擬 LLM 服務"""
        mock = Mock()
        mock.generate.return_value = "這是一個測試回應。"
        mock.generate_async = AsyncMock(return_value="這是一個異步測試回應。")
        mock.is_available.return_value = True
        mock.model_name = "test-llm-model"
        mock.max_tokens = 4096
        return mock

    @staticmethod
    def create_document_processor() -> Mock:
        """建立模擬文件處理器"""
        mock = Mock()
        mock.process.return_value = TestDataGenerator.generate_document()
        mock.extract_text.return_value = TestDataGenerator.generate_chinese_text()
        mock.supported_formats = [".txt", ".pdf", ".docx", ".md"]
        mock.can_process.return_value = True
        return mock


class AsyncMock(MagicMock):
    """異步 Mock 類別"""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class TestFileManager:
    """測試檔案管理器"""

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.created_files: List[Path] = []

    def create_text_file(self, filename: str, content: str) -> Path:
        """建立文字檔案"""
        file_path = self.temp_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        self.created_files.append(file_path)
        return file_path

    def create_json_file(self, filename: str, data: Dict[str, Any]) -> Path:
        """建立 JSON 檔案"""
        file_path = self.temp_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.created_files.append(file_path)
        return file_path

    def cleanup(self):
        """清理建立的檔案"""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
        self.created_files.clear()


class TestAssertions:
    """測試斷言工具"""

    @staticmethod
    def assert_chinese_text(text: str, min_length: int = 1):
        """斷言文本包含中文字符"""
        assert len(text) >= min_length, f"文本長度不足: {len(text)} < {min_length}"

        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        assert chinese_chars > 0, "文本不包含中文字符"

    @staticmethod
    def assert_vector_format(vector: List[float], expected_dimension: int):
        """斷言向量格式正確"""
        assert isinstance(vector, list), "向量必須是列表格式"
        assert (
            len(vector) == expected_dimension
        ), f"向量維度不匹配: {len(vector)} != {expected_dimension}"
        assert all(isinstance(x, (int, float)) for x in vector), "向量元素必須是數值"

    @staticmethod
    def assert_document_format(document: Dict[str, Any]):
        """斷言文件格式正確"""
        required_fields = ["id", "title", "content"]
        for field in required_fields:
            assert field in document, f"文件缺少必要欄位: {field}"

        assert isinstance(document["id"], str), "文件 ID 必須是字串"
        assert len(document["content"]) > 0, "文件內容不能為空"

    @staticmethod
    def assert_entity_format(entity: Dict[str, Any]):
        """斷言實體格式正確"""
        required_fields = ["id", "name", "type"]
        for field in required_fields:
            assert field in entity, f"實體缺少必要欄位: {field}"

        assert isinstance(entity["id"], str), "實體 ID 必須是字串"
        assert isinstance(entity["name"], str), "實體名稱必須是字串"

    @staticmethod
    def assert_relationship_format(relationship: Dict[str, Any]):
        """斷言關係格式正確"""
        required_fields = ["id", "source", "target", "type"]
        for field in required_fields:
            assert field in relationship, f"關係缺少必要欄位: {field}"

        assert isinstance(relationship["source"], str), "關係來源必須是字串"
        assert isinstance(relationship["target"], str), "關係目標必須是字串"


class PerformanceTimer:
    """效能計時器"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

    def assert_duration_less_than(self, max_seconds: float):
        """斷言執行時間小於指定值"""
        assert self.duration is not None, "計時器尚未完成"
        assert (
            self.duration < max_seconds
        ), f"執行時間過長: {self.duration:.3f}s > {max_seconds}s"


def skip_if_no_gpu():
    """如果沒有 GPU 則跳過測試"""
    try:
        import torch

        if not torch.cuda.is_available():
            import pytest

            pytest.skip("需要 GPU 支援")
    except ImportError:
        import pytest

        pytest.skip("需要 PyTorch 和 GPU 支援")


def skip_if_no_internet():
    """如果沒有網路連接則跳過測試"""
    import socket

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except OSError:
        import pytest

        pytest.skip("需要網路連接")


def capture_logs(logger_name: str = None, level: int = logging.INFO):
    """捕獲日誌輸出的裝飾器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            import logging
            from unittest.mock import patch

            logs = []

            def capture_log(record):
                logs.append(record)

            logger = (
                logging.getLogger(logger_name) if logger_name else logging.getLogger()
            )

            with patch.object(logger, "handle", side_effect=capture_log):
                result = func(*args, **kwargs)

            # 將日誌添加到結果中（如果結果是字典）
            if isinstance(result, dict):
                result["captured_logs"] = logs

            return result

        return wrapper

    return decorator
