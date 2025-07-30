# 專案概述

## 專案名稱
chinese-graphrag - 中文 GraphRAG 系統

## 專案目的
基於 Microsoft GraphRAG 框架的知識圖譜檢索增強生成系統，專門針對中文文件處理進行優化。

## 技術堆疊
- **語言**: Python 3.12+
- **套件管理**: uv
- **核心框架**: Microsoft GraphRAG (>=2.4.0)
- **中文處理**: jieba (>=0.42.1)
- **資料驗證**: Pydantic (>=2.0.0)
- **向量資料庫**: LanceDB (>=0.5.0)
- **Embedding**: sentence-transformers (>=5.0.0)
- **測試框架**: pytest + pytest-cov + pytest-asyncio
- **程式碼品質**: black + isort + flake8 + mypy

## 專案架構
```
src/chinese_graphrag/
├── config/              # 配置管理
├── models/              # 資料模型 (Document, TextUnit, Entity, Relationship, Community)
├── processors/          # 文件處理器
├── embeddings/          # Embedding 服務
├── vector_stores/       # 向量資料庫介面
├── indexing/            # 索引引擎
├── query/               # 查詢引擎
└── cli/                 # 命令列介面
```

## 當前開發狀態
- ✅ 基礎架構建立完成
- ✅ 核心資料模型已實作
- ✅ 文件處理介面已建立
- ✅ 中文文本處理器（ChineseTextProcessor）已實作
- 🔄 正在進行任務3：完善中文文本處理模組

## 特色功能
- 🇨🇳 中文優化處理
- 📊 知識圖譜建構
- 💾 向量資料庫整合
- ⚡ 批次處理和並行處理支援