"""
測試資料管理器

負責管理測試過程中使用的各種資料，包括測試文件、向量資料、配置檔案等。
提供資料的建立、清理、重設和版本管理功能。
"""

import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dataclasses import dataclass, asdict


@dataclass
class TestDataInfo:
    """測試資料資訊"""
    name: str
    description: str
    data_type: str  # 'document', 'vector', 'config', 'fixture'
    file_path: str
    size_bytes: int
    created_at: str
    checksum: str
    tags: List[str]
    metadata: Dict[str, Any]


class TestDataManager:
    """測試資料管理器"""
    
    def __init__(self, base_dir: Union[str, Path] = None):
        """
        初始化測試資料管理器
        
        Args:
            base_dir: 測試資料基礎目錄，預設為 ./test_data
        """
        self.base_dir = Path(base_dir) if base_dir else Path("test_data")
        self.base_dir.mkdir(exist_ok=True)
        
        # 建立子目錄結構
        self.documents_dir = self.base_dir / "documents"
        self.vectors_dir = self.base_dir / "vectors"
        self.configs_dir = self.base_dir / "configs"
        self.fixtures_dir = self.base_dir / "fixtures"
        self.temp_dir = self.base_dir / "temp"
        
        for dir_path in [self.documents_dir, self.vectors_dir, self.configs_dir, 
                        self.fixtures_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 資料註冊表檔案
        self.registry_file = self.base_dir / "data_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, TestDataInfo]:
        """載入資料註冊表"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        name: TestDataInfo(**info) 
                        for name, info in data.items()
                    }
            except Exception as e:
                print(f"載入資料註冊表失敗: {e}")
        
        return {}
    
    def _save_registry(self):
        """儲存資料註冊表"""
        registry_data = {
            name: asdict(info) 
            for name, info in self.registry.items()
        }
        
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, ensure_ascii=False, indent=2)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """計算檔案校驗和"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def register_data(
        self,
        name: str,
        file_path: Union[str, Path],
        data_type: str,
        description: str = "",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> TestDataInfo:
        """
        註冊測試資料
        
        Args:
            name: 資料名稱
            file_path: 檔案路徑
            data_type: 資料類型
            description: 描述
            tags: 標籤列表
            metadata: 元資料
            
        Returns:
            TestDataInfo: 資料資訊
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"檔案不存在: {file_path}")
        
        data_info = TestDataInfo(
            name=name,
            description=description,
            data_type=data_type,
            file_path=str(file_path),
            size_bytes=file_path.stat().st_size,
            created_at=datetime.now().isoformat(),
            checksum=self._calculate_checksum(file_path),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.registry[name] = data_info
        self._save_registry()
        
        return data_info
    
    def get_data_info(self, name: str) -> Optional[TestDataInfo]:
        """獲取資料資訊"""
        return self.registry.get(name)
    
    def list_data(
        self,
        data_type: str = None,
        tags: List[str] = None
    ) -> List[TestDataInfo]:
        """
        列出測試資料
        
        Args:
            data_type: 篩選資料類型
            tags: 篩選標籤
            
        Returns:
            List[TestDataInfo]: 資料資訊列表
        """
        results = []
        
        for data_info in self.registry.values():
            # 類型篩選
            if data_type and data_info.data_type != data_type:
                continue
            
            # 標籤篩選
            if tags and not any(tag in data_info.tags for tag in tags):
                continue
            
            results.append(data_info)
        
        return results
    
    def create_sample_documents(self) -> Dict[str, Path]:
        """建立範例文件集合"""
        documents = {}
        
        # 中文 AI 技術文件
        ai_doc = self.documents_dir / "ai_technology.md"
        ai_content = """# 人工智慧技術概述

人工智慧（Artificial Intelligence，簡稱AI）是電腦科學的一個重要分支，致力於研究、開發用於模擬、延伸和擴展人的智慧的理論、方法、技術及應用系統。

## 主要技術領域

### 機器學習 (Machine Learning)
機器學習是AI的核心組成部分，讓電腦能夠從資料中學習規律和模式，而無需進行明確程式設計。

#### 監督學習
- 分類問題：如圖像識別、垃圾郵件檢測
- 迴歸問題：如價格預測、風險評估

#### 無監督學習
- 聚類分析：客戶分群、市場細分
- 降維技術：資料視覺化、特徵提取

#### 強化學習
- 遊戲AI：如圍棋、西洋棋
- 自動控制：機器人導航、交通管理

### 深度學習 (Deep Learning)
深度學習是機器學習的子集，使用多層神經網路來模擬人腦的學習過程。

#### 卷積神經網路 (CNN)
- 圖像識別和電腦視覺
- 醫學影像分析
- 自動駕駛視覺系統

#### 循環神經網路 (RNN)
- 自然語言處理
- 語音識別和合成
- 時間序列預測

#### Transformer 架構
- 機器翻譯
- 文本生成和理解
- 多模態學習

## 應用領域

### 醫療健康
- 疾病診斷輔助
- 藥物研發加速
- 個人化醫療方案

### 金融服務
- 風險評估和管理
- 演算法交易
- 反欺詐檢測

### 教育領域
- 個人化學習推薦
- 智慧教學助手
- 學習效果評估

### 製造業
- 品質控制自動化
- 預測性維護
- 供應鏈優化

## 技術挑戰

### 資料品質
- 資料偏見和不平衡
- 隱私保護和安全
- 資料標註成本

### 模型可解釋性
- 黑箱問題
- 決策透明度
- 責任歸屬

### 倫理考量
- 演算法公平性
- 就業影響
- 社會責任

## 未來發展趨勢

1. **AGI (Artificial General Intelligence)**：通用人工智慧的發展
2. **邊緣計算**：將AI能力部署到邊緣設備
3. **量子機器學習**：結合量子計算和機器學習
4. **可持續AI**：減少AI系統的能源消耗
5. **人機協作**：增强人類而非取代人類

AI技術的發展將持續改變我們的生活和工作方式，創造更多可能性。
"""
        
        with open(ai_doc, 'w', encoding='utf-8') as f:
            f.write(ai_content)
        
        documents['ai_technology'] = ai_doc
        self.register_data(
            'ai_technology',
            ai_doc,
            'document',
            '人工智慧技術概述文件',
            ['ai', 'technology', 'chinese'],
            {'language': 'zh-TW', 'word_count': len(ai_content)}
        )
        
        # 機器學習詳細文件
        ml_doc = self.documents_dir / "machine_learning_guide.txt"
        ml_content = """機器學習完整指南

機器學習是人工智慧的一個子領域，專注於開發能夠從資料中學習並做出預測或決策的演算法。

學習類型分類：

1. 監督學習 (Supervised Learning)
監督學習使用標記的訓練資料來學習從輸入到輸出的映射關係。

常見演算法：
- 線性迴歸：用於連續數值預測
- 邏輯迴歸：用於二元分類問題
- 決策樹：易於理解和解釋的分類方法
- 隨機森林：多個決策樹的集成方法
- 支援向量機 (SVM)：尋找最佳分離超平面
- 神經網路：模擬人腦神經元的連接方式

應用實例：
- 電子郵件垃圾信件分類
- 房價預測
- 醫療診斷輔助
- 客戶信用評估

2. 無監督學習 (Unsupervised Learning)
無監督學習處理沒有標籤的資料，目標是發現資料中的隱藏結構。

主要方法：
- K-means 聚類：將資料分成 k 個群組
- 階層聚類：建立資料的樹狀結構
- 主成分分析 (PCA)：降維和特徵提取
- 獨立成分分析 (ICA)：信號分離
- 關聯規則學習：發現項目間的關聯性

應用場景：
- 客戶分群和市場細分
- 異常檢測
- 推薦系統
- 資料壓縮

3. 強化學習 (Reinforcement Learning)
強化學習透過與環境互動，學習如何選擇動作以最大化累積獎勵。

核心概念：
- 智慧體 (Agent)：學習和決策的實體
- 環境 (Environment)：智慧體操作的場景
- 狀態 (State)：環境的當前情況
- 動作 (Action)：智慧體可執行的操作
- 獎勵 (Reward)：動作的即時回饋

經典演算法：
- Q-Learning：學習狀態-動作價值函數
- SARSA：同策略時間差分學習
- Actor-Critic：結合價值函數和策略梯度
- Deep Q-Network (DQN)：深度學習版本的Q-Learning

成功應用：
- 遊戲AI (AlphaGo, OpenAI Five)
- 自動駕駛
- 機器人控制
- 資源分配最佳化

模型評估與選擇：

評估指標：
- 分類問題：準確率、精確率、召回率、F1分數
- 迴歸問題：均方誤差、平均絕對誤差、決定係數
- 聚類問題：輪廓係數、調整蘭德指數

交叉驗證：
- k-fold交叉驗證：將資料分成k份進行驗證
- 留一法：每次留一個樣本做驗證
- 時間序列分割：按時間順序分割資料

過擬合和欠擬合：
- 過擬合：模型過於複雜，記住了訓練資料的雜訊
- 欠擬合：模型過於簡單，無法捕捉資料的模式
- 正則化：L1、L2正則化防止過擬合
- 早停法：監控驗證集性能，適時停止訓練

特徵工程：
- 特徵選擇：選擇最相關的特徵
- 特徵提取：從原始資料中提取有用資訊
- 特徵縮放：標準化、正規化處理
- 特徵組合：創建新的組合特徵

實際應用考量：
- 資料品質：完整性、準確性、一致性
- 計算資源：訓練時間、記憶體需求
- 模型解釋性：業務需求vs模型複雜度
- 部署和維護：模型更新、監控、版本控制

機器學習是一個快速發展的領域，新的演算法和技術不斷涌現，需要持續學習和實踐。
"""
        
        with open(ml_doc, 'w', encoding='utf-8') as f:
            f.write(ml_content)
        
        documents['machine_learning'] = ml_doc
        self.register_data(
            'machine_learning',
            ml_doc,
            'document',
            '機器學習完整指南',
            ['ml', 'guide', 'chinese'],
            {'language': 'zh-TW', 'word_count': len(ml_content)}
        )
        
        # 深度學習技術文件
        dl_doc = self.documents_dir / "deep_learning.md"
        dl_content = """# 深度學習技術詳解

深度學習是機器學習的一個子集，使用多層神經網路來學習資料的複雜表示。

## 神經網路基礎

### 感知機 (Perceptron)
最簡單的神經網路單元，接收多個輸入並產生一個輸出。

### 多層感知機 (MLP)
包含輸入層、隱藏層和輸出層的前饋神經網路。

### 激活函數
- Sigmoid：將輸出壓縮到 (0,1) 區間
- Tanh：將輸出壓縮到 (-1,1) 區間  
- ReLU：修正線性單元，解決梯度消失問題
- Leaky ReLU：允許負值有小幅度輸出
- Swish：自門控激活函數

## 重要架構

### 卷積神經網路 (CNN)
專門處理網格狀資料（如圖像）的神經網路架構。

#### 核心組件
- 卷積層：使用濾波器提取局部特徵
- 池化層：降低資料維度，保留重要資訊
- 全連接層：進行最終的分類或迴歸

#### 經典架構
- LeNet：最早的CNN架構
- AlexNet：深度學習復興的里程碑
- VGG：使用小濾波器的深度網路
- ResNet：引入殘差連接解決梯度消失
- DenseNet：密集連接的特徵重用

### 循環神經網路 (RNN)
處理序列資料的神經網路，具有記憶能力。

#### RNN變體
- Vanilla RNN：基礎循環神經網路
- LSTM：長短期記憶網路，解決長程依賴問題
- GRU：門控循環單元，LSTM的簡化版本
- Bidirectional RNN：雙向處理序列資訊

#### 應用領域
- 自然語言處理
- 語音識別
- 時間序列預測
- 機器翻譯

### Transformer 架構
基於注意力機制的革命性架構，不依賴循環或卷積操作。

#### 核心機制
- 自注意力 (Self-Attention)：計算序列內部的關聯性
- 多頭注意力：並行計算多個注意力表示
- 位置編碼：為序列添加位置資訊
- 前饋網路：點對點的特徵變換

#### 重要模型
- BERT：雙向編碼器表示
- GPT：生成式預訓練Transformer
- T5：文本到文本轉換Transformer
- Vision Transformer (ViT)：視覺Transformer

## 訓練技巧

### 優化演算法
- SGD：隨機梯度下降
- Momentum：動量法加速收斂
- AdaGrad：自適應梯度演算法
- Adam：結合動量和自適應學習率
- AdamW：權重衰減版本的Adam

### 正則化技術
- Dropout：隨機丟棄部分神經元
- Batch Normalization：批次正規化穩定訓練
- Layer Normalization：層正規化
- Weight Decay：權重衰減防止過擬合
- Early Stopping：早停避免過擬合

### 學習率調度
- 固定學習率：整個訓練過程保持不變
- 階梯衰減：按階段降低學習率
- 指數衰減：指數形式衰減
- 餘弦退火：餘弦函數調節學習率
- 循環學習率：在範圍內循環調整

## 生成式模型

### 變分自編碼器 (VAE)
學習資料的潛在表示，能夠生成新的資料樣本。

### 生成對抗網路 (GAN)
通過生成器和判別器的對抗訓練生成逼真資料。

#### GAN變體
- DCGAN：深度卷積GAN
- WGAN：Wasserstein GAN改善訓練穩定性
- StyleGAN：控制生成圖像的風格
- CycleGAN：無配對資料的圖像轉換

### 擴散模型
透過逐步去噪過程生成高品質圖像。

## 實際應用

### 電腦視覺
- 圖像分類和物體檢測
- 語義分割和例項分割
- 人臉識別和表情分析
- 醫學影像診斷

### 自然語言處理
- 機器翻譯
- 文本摘要
- 問答系統
- 對話系統

### 多模態學習
- 圖像描述生成
- 視覺問答
- 視頻理解
- 跨模態檢索

## 發展趨勢

### 模型效率
- 模型壓縮和剪枝
- 知識蒸餾
- 量化技術
- 神經架構搜索

### 可解釋性
- 注意力視覺化
- 梯度分析
- 概念激活向量
- 反事實解釋

### 持續學習
- 避免災難性遺忘
- 元學習
- 少樣本學習
- 零樣本學習

深度學習技術持續快速發展，在各個領域都有重大突破和應用。
"""
        
        with open(dl_doc, 'w', encoding='utf-8') as f:
            f.write(dl_content)
        
        documents['deep_learning'] = dl_doc
        self.register_data(
            'deep_learning',  
            dl_doc,
            'document',
            '深度學習技術詳解',
            ['dl', 'neural_networks', 'chinese'],
            {'language': 'zh-TW', 'word_count': len(dl_content)}
        )
        
        print(f"已建立 {len(documents)} 個範例文件")
        return documents
    
    def create_test_configs(self) -> Dict[str, Path]:
        """建立測試配置檔案"""
        configs = {}
        
        # 基本測試配置
        basic_config = {
            'embedding': {
                'model': 'BAAI/bge-m3',
                'dimension': 768,
                'batch_size': 16,
                'device': 'cpu'
            },
            'vector_store': {
                'type': 'lancedb',
                'path': './test_vectors',
                'table_name': 'test_embeddings'
            },
            'llm': {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'indexing': {
                'chunk_size': 500,
                'chunk_overlap': 50,
                'min_chunk_size': 100
            },
            'query': {
                'top_k': 10,
                'similarity_threshold': 0.7,
                'max_context_length': 4000
            }
        }
        
        basic_config_file = self.configs_dir / "basic_test_config.yaml"
        with open(basic_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(basic_config, f, allow_unicode=True, default_flow_style=False)
        
        configs['basic'] = basic_config_file
        self.register_data(
            'basic_test_config',
            basic_config_file,
            'config',
            '基本測試配置檔案',
            ['config', 'test', 'basic']
        )
        
        # 效能測試配置
        performance_config = {
            'embedding': {
                'model': 'BAAI/bge-m3',
                'dimension': 768,
                'batch_size': 64,  # 更大批次
                'device': 'cpu'
            },
            'vector_store': {
                'type': 'lancedb',
                'path': './perf_test_vectors',
                'table_name': 'perf_embeddings'
            },
            'indexing': {
                'chunk_size': 1000,  # 更大分塊
                'chunk_overlap': 100,
                'min_chunk_size': 200,
                'parallel_workers': 4
            },
            'performance': {
                'max_documents': 1000,
                'max_processing_time': 300,
                'memory_limit_mb': 2048,
                'benchmark_iterations': 10
            }
        }
        
        perf_config_file = self.configs_dir / "performance_test_config.yaml"
        with open(perf_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(performance_config, f, allow_unicode=True, default_flow_style=False)
        
        configs['performance'] = perf_config_file
        self.register_data(
            'performance_test_config',
            perf_config_file,
            'config',
            '效能測試配置檔案',
            ['config', 'test', 'performance']
        )
        
        # 中文特定配置
        chinese_config = {
            'text_processing': {
                'language': 'zh-TW',
                'segmentation': 'jieba',
                'stopwords_file': 'chinese_stopwords.txt',
                'min_word_length': 1,
                'max_word_length': 20
            },
            'embedding': {
                'model': 'BAAI/bge-m3',
                'dimension': 768,
                'batch_size': 32,
                'normalize': True,
                'chinese_optimization': True
            },
            'indexing': {
                'chunk_size': 300,  # 中文字符較短
                'chunk_overlap': 30,
                'sentence_splitter': 'chinese_aware',
                'preserve_formatting': True
            },
            'query': {
                'chinese_query_expansion': True,
                'synonym_matching': True,
                'traditional_simplified_convert': True
            }
        }
        
        chinese_config_file = self.configs_dir / "chinese_test_config.yaml"
        with open(chinese_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(chinese_config, f, allow_unicode=True, default_flow_style=False)
        
        configs['chinese'] = chinese_config_file
        self.register_data(
            'chinese_test_config',
            chinese_config_file,
            'config',
            '中文特定測試配置檔案',
            ['config', 'test', 'chinese']
        )
        
        print(f"已建立 {len(configs)} 個測試配置檔案")
        return configs
    
    def create_test_fixtures(self) -> Dict[str, Path]:
        """建立測試夾具資料"""
        fixtures = {}
        
        # 測試查詢集合
        test_queries = {
            'definition_queries': [
                "什麼是人工智慧？",
                "機器學習的定義是什麼？",
                "請解釋深度學習的概念",
                "神經網路是如何工作的？"
            ],
            'comparison_queries': [
                "機器學習和深度學習有什麼區別？",
                "CNN和RNN的差異在哪裡？",
                "監督學習與無監督學習的對比",
                "Transformer和RNN的優缺點比較"
            ],
            'application_queries': [
                "人工智慧在醫療領域的應用",
                "深度學習在自然語言處理中的使用",
                "機器學習在金融業的實際案例",
                "電腦視覺技術的商業應用"
            ],
            'technical_queries': [
                "如何選擇合適的激活函數？",
                "什麼時候使用CNN而不是RNN？",
                "如何解決梯度消失問題？",
                "Attention機制的工作原理"
            ]
        }
        
        queries_file = self.fixtures_dir / "test_queries.json"
        with open(queries_file, 'w', encoding='utf-8') as f:
            json.dump(test_queries, f, ensure_ascii=False, indent=2)
        
        fixtures['queries'] = queries_file
        self.register_data(
            'test_queries',
            queries_file,
            'fixture',
            '測試查詢集合',
            ['queries', 'test', 'chinese']
        )
        
        # 預期答案範本
        expected_answers = {
            '什麼是人工智慧？': {
                'keywords': ['人工智慧', 'AI', '電腦科學', '模擬智慧'],
                'min_length': 100,
                'should_mention': ['機器學習', '技術', '應用'],
                'confidence_threshold': 0.8
            },
            '機器學習和深度學習有什麼區別？': {
                'keywords': ['機器學習', '深度學習', '神經網路', '差別'],
                'min_length': 150,
                'should_mention': ['多層', '特徵', '演算法'],
                'confidence_threshold': 0.85
            }
        }
        
        answers_file = self.fixtures_dir / "expected_answers.json"
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(expected_answers, f, ensure_ascii=False, indent=2)
        
        fixtures['answers'] = answers_file
        self.register_data(
            'expected_answers',
            answers_file,
            'fixture',
            '預期答案範本',
            ['answers', 'validation', 'test']
        )
        
        # 效能基準資料
        performance_benchmarks = {
            'document_processing': {
                'max_time_per_document': 2.0,
                'max_memory_per_document_mb': 10.0,
                'min_throughput_docs_per_second': 5.0
            },
            'embedding_generation': {
                'max_time_per_batch': 5.0,
                'max_memory_per_batch_mb': 500.0,
                'min_throughput_texts_per_second': 20.0
            },
            'vector_search': {
                'max_search_time_ms': 100.0,
                'max_memory_per_search_mb': 50.0,
                'min_precision_at_k': 0.8
            },
            'end_to_end_query': {
                'max_response_time_seconds': 10.0,
                'max_memory_usage_mb': 1000.0,
                'min_answer_quality_score': 0.7
            }
        }
        
        benchmarks_file = self.fixtures_dir / "performance_benchmarks.json"
        with open(benchmarks_file, 'w', encoding='utf-8') as f:
            json.dump(performance_benchmarks, f, ensure_ascii=False, indent=2)
        
        fixtures['benchmarks'] = benchmarks_file
        self.register_data(
            'performance_benchmarks',
            benchmarks_file,
            'fixture',
            '效能基準資料',
            ['performance', 'benchmarks', 'test']
        )
        
        print(f"已建立 {len(fixtures)} 個測試夾具檔案")
        return fixtures
    
    def create_temporary_workspace(self, prefix: str = "test_") -> Path:
        """建立臨時工作空間"""
        workspace = Path(tempfile.mkdtemp(prefix=prefix, dir=self.temp_dir))
        
        # 建立標準子目錄
        (workspace / "documents").mkdir()
        (workspace / "data").mkdir()
        (workspace / "output").mkdir()
        (workspace / "logs").mkdir()
        
        return workspace
    
    def cleanup_temporary_data(self, max_age_hours: int = 24):
        """清理臨時資料"""
        import time
        current_time = time.time()
        
        cleaned_count = 0
        for item in self.temp_dir.iterdir():
            if item.is_dir():
                # 檢查目錄修改時間
                mod_time = item.stat().st_mtime
                age_hours = (current_time - mod_time) / 3600
                
                if age_hours > max_age_hours:
                    shutil.rmtree(item)
                    cleaned_count += 1
        
        print(f"已清理 {cleaned_count} 個過期的臨時目錄")
        return cleaned_count
    
    def validate_data_integrity(self) -> Dict[str, bool]:
        """驗證資料完整性"""
        results = {}
        
        for name, data_info in self.registry.items():
            file_path = Path(data_info.file_path)
            
            # 檢查檔案是否存在
            if not file_path.exists():
                results[name] = False
                continue
            
            # 檢查檔案大小
            current_size = file_path.stat().st_size
            if current_size != data_info.size_bytes:
                results[name] = False
                continue
            
            # 檢查校驗和
            current_checksum = self._calculate_checksum(file_path)
            if current_checksum != data_info.checksum:
                results[name] = False
                continue
            
            results[name] = True
        
        return results
    
    def export_data_manifest(self, output_file: str = "data_manifest.json") -> Path:
        """匯出資料清單"""
        manifest = {
            'generated_at': datetime.now().isoformat(),
            'total_data_count': len(self.registry),
            'data_by_type': {},
            'data_registry': {name: asdict(info) for name, info in self.registry.items()}
        }
        
        # 按類型統計
        for data_info in self.registry.values():
            data_type = data_info.data_type
            if data_type not in manifest['data_by_type']:
                manifest['data_by_type'][data_type] = 0
            manifest['data_by_type'][data_type] += 1
        
        manifest_file = self.base_dir / output_file
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        print(f"資料清單已匯出至: {manifest_file}")
        return manifest_file
    
    def reset_all_data(self, confirm: bool = False):
        """重設所有測試資料"""
        if not confirm:
            print("警告：此操作將刪除所有測試資料！")
            print("請使用 reset_all_data(confirm=True) 確認執行")
            return
        
        # 清空註冊表
        self.registry.clear()
        self._save_registry()
        
        # 刪除所有資料檔案
        for dir_path in [self.documents_dir, self.vectors_dir, self.configs_dir, 
                        self.fixtures_dir, self.temp_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                dir_path.mkdir()
        
        print("所有測試資料已重設")


def main():
    """測試資料管理器示例用法"""
    # 建立測試資料管理器
    manager = TestDataManager()
    
    print("🗂️ 初始化測試資料管理器")
    print(f"資料目錄: {manager.base_dir}")
    
    # 建立範例資料
    print("\n📄 建立範例文件...")
    documents = manager.create_sample_documents()
    
    print("\n⚙️ 建立測試配置...")
    configs = manager.create_test_configs()
    
    print("\n🧪 建立測試夾具...")
    fixtures = manager.create_test_fixtures()
    
    # 列出所有資料
    print("\n📊 資料總覽:")
    all_data = manager.list_data()
    for data_info in all_data:
        print(f"  {data_info.name} ({data_info.data_type}): {data_info.description}")
    
    # 驗證資料完整性
    print("\n✅ 驗證資料完整性...")
    integrity_results = manager.validate_data_integrity()
    valid_count = sum(integrity_results.values())
    total_count = len(integrity_results)
    print(f"有效資料: {valid_count}/{total_count}")
    
    # 匯出資料清單
    print("\n📋 匯出資料清單...")
    manifest_file = manager.export_data_manifest()
    
    # 清理臨時資料
    print("\n🧹 清理過期臨時資料...")
    manager.cleanup_temporary_data()
    
    print("\n✨ 測試資料管理器設定完成！")


if __name__ == "__main__":
    main()