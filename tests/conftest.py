"""
測試配置文件

提供測試所需的 fixture 和全域配置。
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock

import pytest

# 設定測試日誌級別
logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="session")
def event_loop():
    """建立事件循環"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """建立臨時目錄"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_data_dir() -> Path:
    """測試資料目錄"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_text() -> str:
    """範例中文文本"""
    return """
    人工智慧（Artificial Intelligence，簡稱AI）是電腦科學的一個分支，
    它企圖了解智慧的實質，並生產出一種新的能以人類智慧相似的方式做出反應的智慧機器。
    該領域的研究包括機器人、語言識別、圖像識別、自然語言處理和專家系統等。
    
    機器學習是人工智慧的一個子領域，專注於演算法的研究，這些演算法能夠透過經驗自動改進。
    深度學習又是機器學習的一個子集，它試圖模擬人腦神經網路的工作方式。
    """


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """範例文件資料"""
    return [
        {
            "id": "doc_1",
            "title": "人工智慧概述",
            "content": "人工智慧是電腦科學的重要分支，致力於開發智慧機器。",
            "metadata": {"author": "張三", "date": "2024-01-01"},
        },
        {
            "id": "doc_2",
            "title": "機器學習基礎",
            "content": "機器學習是AI的核心技術，通過資料訓練模型。",
            "metadata": {"author": "李四", "date": "2024-01-02"},
        },
        {
            "id": "doc_3",
            "title": "深度學習應用",
            "content": "深度學習在圖像識別和自然語言處理中有廣泛應用。",
            "metadata": {"author": "王五", "date": "2024-01-03"},
        },
    ]


@pytest.fixture
def sample_entities() -> List[Dict[str, Any]]:
    """範例實體資料"""
    return [
        {
            "id": "entity_1",
            "name": "人工智慧",
            "type": "概念",
            "description": "電腦科學分支，研究智慧機器",
            "embedding": [0.1] * 768,
        },
        {
            "id": "entity_2",
            "name": "機器學習",
            "type": "技術",
            "description": "AI子領域，專注演算法研究",
            "embedding": [0.2] * 768,
        },
        {
            "id": "entity_3",
            "name": "深度學習",
            "type": "技術",
            "description": "機器學習子集，模擬神經網路",
            "embedding": [0.3] * 768,
        },
    ]


@pytest.fixture
def sample_relationships() -> List[Dict[str, Any]]:
    """範例關係資料"""
    return [
        {
            "id": "rel_1",
            "source": "entity_2",
            "target": "entity_1",
            "type": "子領域",
            "description": "機器學習是人工智慧的子領域",
            "weight": 0.8,
        },
        {
            "id": "rel_2",
            "source": "entity_3",
            "target": "entity_2",
            "type": "子集",
            "description": "深度學習是機器學習的子集",
            "weight": 0.9,
        },
    ]


@pytest.fixture
def mock_embedding_service():
    """模擬 Embedding 服務"""
    mock = Mock()
    mock.encode.return_value = [0.1] * 768
    mock.encode_batch.return_value = [[0.1] * 768, [0.2] * 768]
    mock.get_dimension.return_value = 768
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def mock_vector_store():
    """模擬向量資料庫"""
    mock = Mock()
    mock.add_documents.return_value = True
    mock.search.return_value = [{"id": "doc_1", "score": 0.9}]
    mock.get_document.return_value = {"id": "doc_1", "content": "test"}
    mock.delete_document.return_value = True
    mock.list_collections.return_value = ["test_collection"]
    return mock


@pytest.fixture
def mock_llm_service():
    """模擬 LLM 服務"""
    mock = Mock()
    mock.generate.return_value = "這是一個測試回應"
    mock.generate_async.return_value = asyncio.coroutine(lambda: "異步測試回應")()
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """測試配置"""
    return {
        "embedding": {"model_name": "BAAI/bge-m3", "device": "cpu", "batch_size": 32},
        "vector_store": {
            "type": "lancedb",
            "path": "./test_data/vector_db",
            "collection_name": "test_collection",
        },
        "llm": {"provider": "openai", "model": "gpt-3.5-turbo", "temperature": 0.7},
        "indexing": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "max_chunks_per_document": 100,
        },
        "chinese": {
            "jieba_dict_path": None,
            "stopwords_path": None,
            "enable_parallel": True,
        },
    }


@pytest.fixture
def cleanup_files():
    """清理測試檔案的 fixture"""
    files_to_cleanup = []

    def add_file(file_path: Path):
        files_to_cleanup.append(file_path)

    yield add_file

    # 清理檔案
    for file_path in files_to_cleanup:
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                import shutil

                shutil.rmtree(file_path)


# 測試標記
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.chinese = pytest.mark.chinese


# 跳過條件
def pytest_configure(config):
    """配置 pytest"""
    config.addinivalue_line("markers", "unit: 單元測試標記")
    config.addinivalue_line("markers", "integration: 整合測試標記")
    config.addinivalue_line("markers", "slow: 慢速測試標記")
    config.addinivalue_line("markers", "chinese: 中文處理測試標記")


def pytest_collection_modifyitems(config, items):
    """修改測試項目"""
    # 檢查環境變數，決定是否跳過特定測試
    skip_integration = pytest.mark.skip(
        reason="跳過整合測試（使用 --integration 運行）"
    )
    skip_slow = pytest.mark.skip(reason="跳過慢速測試（使用 --slow 運行）")

    run_integration = (
        config.getoption("--integration", default=False)
        if hasattr(config, "getoption")
        else False
    )
    run_slow = (
        config.getoption("--slow", default=False)
        if hasattr(config, "getoption")
        else False
    )

    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


# 自定義命令列選項
def pytest_addoption(parser):
    """添加命令列選項"""
    parser.addoption(
        "--integration", action="store_true", default=False, help="運行整合測試"
    )
    parser.addoption("--slow", action="store_true", default=False, help="運行慢速測試")
