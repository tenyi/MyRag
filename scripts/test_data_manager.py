#!/usr/bin/env python3
"""
測試資料管理器

負責管理測試過程中所需的各種資料，包括範例文件、測試配置、測試夾具等。
提供資料的建立、驗證、清理和重設功能。
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


class TestDataManager:
    """測試資料管理器類別"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化測試資料管理器
        
        Args:
            base_dir: 測試資料基礎目錄，預設為 tests/test_data
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path(__file__).parent.parent / "tests" / "test_data"
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 定義各種資料目錄
        self.documents_dir = self.base_dir / "documents"
        self.configs_dir = self.base_dir / "configs"
        self.fixtures_dir = self.base_dir / "fixtures"
        self.temp_dir = self.base_dir / "temp"
        
        # 建立目錄結構
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """建立目錄結構"""
        directories = [
            self.documents_dir,
            self.configs_dir,
            self.fixtures_dir,
            self.temp_dir,
            self.documents_dir / "chinese",
            self.documents_dir / "english",
            self.documents_dir / "mixed",
            self.configs_dir / "embedding",
            self.configs_dir / "vector_store",
            self.configs_dir / "llm"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def create_sample_documents(self) -> Dict[str, List[Path]]:
        """建立範例文件"""
        documents = {
            "chinese": [],
            "english": [],
            "mixed": []
        }
        
        # 中文文件
        chinese_docs = [
            {
                "filename": "ai_overview.txt",
                "content": """人工智慧概述

人工智慧（Artificial Intelligence，簡稱AI）是電腦科學的一個重要分支，致力於研究、開發用於模擬、延伸和擴展人的智慧的理論、方法、技術及應用系統。

## 主要領域

### 機器學習
機器學習是人工智慧的核心技術之一，通過演算法讓電腦系統能夠自動學習和改進，而無需明確程式設計。主要包括：

- 監督學習：使用標記資料訓練模型
- 無監督學習：從未標記資料中發現模式
- 強化學習：通過與環境互動學習最佳策略

### 深度學習
深度學習是機器學習的子集，使用多層神經網路來模擬人腦的工作方式。在圖像識別、自然語言處理、語音識別等領域取得了突破性進展。

### 自然語言處理
自然語言處理（NLP）專注於讓電腦理解、解釋和生成人類語言。包括：
- 文本分析
- 語言翻譯
- 情感分析
- 問答系統

## 應用領域

人工智慧技術已廣泛應用於各個領域：

1. **醫療健康**：疾病診斷、藥物研發、個人化治療
2. **金融服務**：風險評估、演算法交易、反欺詐
3. **交通運輸**：自動駕駛、交通優化、物流管理
4. **教育**：個人化學習、智慧輔導、自動評分
5. **娛樂**：推薦系統、遊戲AI、內容生成

## 發展趨勢

未來人工智慧的發展將朝向更加智慧化、人性化的方向：
- 通用人工智慧（AGI）的研究
- 可解釋AI的發展
- AI倫理和安全性的重視
- 人機協作的深化

人工智慧正在改變我們的生活方式和工作模式，為人類社會帶來前所未有的機遇和挑戰。"""
            },
            {
                "filename": "machine_learning_basics.md",
                "content": """# 機器學習基礎

機器學習是人工智慧的重要組成部分，它使電腦能夠在沒有明確程式設計的情況下學習和做出決策。

## 核心概念

### 資料集
機器學習的基礎是資料。資料集通常分為：
- **訓練集**：用於訓練模型的資料
- **驗證集**：用於調整模型參數的資料
- **測試集**：用於評估模型效能的資料

### 特徵工程
特徵工程是機器學習中的關鍵步驟，包括：
- 特徵選擇：選擇最相關的特徵
- 特徵提取：從原始資料中提取有用資訊
- 特徵轉換：將特徵轉換為更適合的形式

### 模型評估
常用的評估指標包括：
- **準確率（Accuracy）**：正確預測的比例
- **精確率（Precision）**：預測為正例中實際為正例的比例
- **召回率（Recall）**：實際正例中被正確預測的比例
- **F1分數**：精確率和召回率的調和平均

## 演算法分類

### 監督學習
- 線性迴歸
- 邏輯迴歸
- 決策樹
- 隨機森林
- 支援向量機（SVM）
- 神經網路

### 無監督學習
- K-means聚類
- 階層聚類
- 主成分分析（PCA）
- 獨立成分分析（ICA）

### 強化學習
- Q-learning
- 策略梯度
- Actor-Critic方法

## 實際應用

機器學習在各個領域都有廣泛應用：
- 推薦系統
- 圖像識別
- 語音識別
- 自然語言處理
- 預測分析
- 異常檢測

學習機器學習需要紮實的數學基礎，包括線性代數、機率統計和微積分。同時，實踐經驗也非常重要。"""
            }
        ]
        
        for doc in chinese_docs:
            file_path = self.documents_dir / "chinese" / doc["filename"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc["content"])
            documents["chinese"].append(file_path)
        
        # 英文文件
        english_docs = [
            {
                "filename": "neural_networks.txt",
                "content": """Neural Networks Overview

Neural networks are computing systems inspired by biological neural networks. They are a key technology in machine learning and artificial intelligence.

## Basic Structure

A neural network consists of:
- Input layer: Receives input data
- Hidden layers: Process the data
- Output layer: Produces the final result

## Types of Neural Networks

### Feedforward Neural Networks
The simplest type where information moves in one direction from input to output.

### Convolutional Neural Networks (CNNs)
Specialized for processing grid-like data such as images.

### Recurrent Neural Networks (RNNs)
Designed for sequential data with memory capabilities.

### Long Short-Term Memory (LSTM)
A type of RNN that can learn long-term dependencies.

## Training Process

1. Forward propagation: Data flows through the network
2. Loss calculation: Compare output with expected result
3. Backpropagation: Adjust weights to minimize loss
4. Iteration: Repeat until convergence

## Applications

- Image recognition
- Natural language processing
- Speech recognition
- Game playing
- Autonomous vehicles
- Medical diagnosis

Neural networks have revolutionized many fields and continue to be an active area of research."""
            }
        ]
        
        for doc in english_docs:
            file_path = self.documents_dir / "english" / doc["filename"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc["content"])
            documents["english"].append(file_path)
        
        return documents
    
    def create_test_configs(self) -> Dict[str, Path]:
        """建立測試配置檔案"""
        configs = {}
        
        # Embedding 配置
        embedding_config = {
            "model_name": "BAAI/bge-m3",
            "device": "cpu",
            "batch_size": 16,
            "max_length": 512,
            "normalize_embeddings": True,
            "cache_dir": "./cache/embeddings",
            "enable_gpu": False
        }
        
        embedding_path = self.configs_dir / "embedding" / "test_embedding.yaml"
        with open(embedding_path, 'w', encoding='utf-8') as f:
            yaml.dump(embedding_config, f, default_flow_style=False, allow_unicode=True)
        configs["embedding"] = embedding_path
        
        # Vector Store 配置
        vector_store_config = {
            "type": "lancedb",
            "connection": {
                "uri": "./test_data/vector_db",
                "table_name": "test_documents"
            },
            "index_config": {
                "metric": "cosine",
                "num_partitions": 4,
                "num_sub_vectors": 8
            }
        }
        
        vector_store_path = self.configs_dir / "vector_store" / "test_vector_store.yaml"
        with open(vector_store_path, 'w', encoding='utf-8') as f:
            yaml.dump(vector_store_config, f, default_flow_style=False, allow_unicode=True)
        configs["vector_store"] = vector_store_path
        
        return configs
    
    def create_test_fixtures(self) -> Dict[str, Path]:
        """建立測試夾具"""
        fixtures = {}
        
        # 範例實體資料
        entities_fixture = {
            "entities": [
                {
                    "id": "entity_ai",
                    "name": "人工智慧",
                    "type": "概念",
                    "description": "模擬人類智慧的電腦系統",
                    "embedding": [0.1] * 768,
                    "metadata": {
                        "category": "技術",
                        "importance": 0.9
                    }
                },
                {
                    "id": "entity_ml",
                    "name": "機器學習",
                    "type": "技術",
                    "description": "讓電腦自動學習的技術",
                    "embedding": [0.2] * 768,
                    "metadata": {
                        "category": "技術",
                        "importance": 0.8
                    }
                }
            ]
        }
        
        entities_path = self.fixtures_dir / "entities.json"
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities_fixture, f, ensure_ascii=False, indent=2)
        fixtures["entities"] = entities_path
        
        return fixtures
    
    def validate_data_integrity(self) -> Dict[str, bool]:
        """驗證資料完整性"""
        results = {}
        
        # 檢查文件
        document_dirs = ["chinese", "english"]
        for dir_name in document_dirs:
            dir_path = self.documents_dir / dir_name
            results[f"documents_{dir_name}"] = dir_path.exists() and len(list(dir_path.glob("*"))) > 0
        
        # 檢查配置檔案
        config_dirs = ["embedding", "vector_store"]
        for config_dir in config_dirs:
            config_path = self.configs_dir / config_dir
            results[f"config_{config_dir}"] = config_path.exists() and len(list(config_path.glob("*.yaml"))) > 0
        
        # 檢查夾具檔案
        fixture_files = ["entities.json"]
        for fixture_file in fixture_files:
            fixture_path = self.fixtures_dir / fixture_file
            results[f"fixture_{fixture_file}"] = fixture_path.exists()
        
        return results
    
    def get_temp_directory(self) -> Path:
        """取得臨時目錄"""
        temp_subdir = self.temp_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_subdir.mkdir(parents=True, exist_ok=True)
        return temp_subdir
    
    def cleanup_temp_data(self):
        """清理臨時資料"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def reset_all_data(self, confirm: bool = False):
        """重設所有測試資料"""
        if not confirm:
            response = input("確定要重設所有測試資料嗎？這將刪除現有的所有測試檔案。(y/N): ")
            if response.lower() != 'y':
                print("取消重設操作")
                return
        
        # 刪除並重建目錄
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._create_directory_structure()
        
        print("測試資料已重設")


if __name__ == "__main__":
    # 簡單的測試
    manager = TestDataManager()
    print("建立範例文件...")
    docs = manager.create_sample_documents()
    print(f"建立了 {sum(len(files) for files in docs.values())} 個文件")
    
    print("建立測試配置...")
    configs = manager.create_test_configs()
    print(f"建立了 {len(configs)} 個配置檔案")
    
    print("建立測試夾具...")
    fixtures = manager.create_test_fixtures()
    print(f"建立了 {len(fixtures)} 個夾具檔案")
    
    print("驗證資料完整性...")
    results = manager.validate_data_integrity()
    valid_count = sum(results.values())
    total_count = len(results)
    print(f"驗證結果: {valid_count}/{total_count} 通過")