"""
索引引擎模組

提供 GraphRAG 索引功能，包括文件處理、實體提取、關係建立和向量化
"""

from .analyzer import IndexAnalyzer
from .document_processor import DocumentProcessor
from .engine import GraphRAGIndexer

# 為了向後相容性，提供 IndexingEngine 別名
IndexingEngine = GraphRAGIndexer

__all__ = [
    "DocumentProcessor",
    "GraphRAGIndexer",
    "IndexingEngine",
    "IndexAnalyzer",
]
