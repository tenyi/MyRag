"""索引分析器模組。

提供索引結果分析和統計功能。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..monitoring import get_logger

logger = get_logger(__name__)


class IndexAnalyzer:
    """索引分析器，用於分析和統計索引結果。"""

    def __init__(self, index_path: Path):
        """初始化索引分析器。

        Args:
            index_path: 索引結果目錄路徑

        Raises:
            ValueError: 當索引目錄不存在或格式不正確時
        """
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise ValueError(f"索引目錄不存在: {index_path}")

        # 檢查是否有 LanceDB 格式的資料
        self.lancedb_path = self.index_path / "lancedb"
        self.has_lancedb = self.lancedb_path.exists()

        # 常見的 GraphRAG 輸出檔案 (parquet 格式)
        self.artifacts_path = self.index_path / "artifacts"
        self.parquet_files = {
            "entities": self.artifacts_path / "create_final_entities.parquet",
            "relationships": self.artifacts_path / "create_final_relationships.parquet",
            "communities": self.artifacts_path / "create_final_communities.parquet",
            "documents": self.artifacts_path / "create_final_documents.parquet",
            "text_units": self.artifacts_path / "create_final_text_units.parquet",
            "community_reports": self.artifacts_path
            / "create_final_community_reports.parquet",
        }

        # LanceDB 表格名稱
        self.lance_tables = {
            "entities": "entities.lance",
            "relationships": "relationships.lance",
            "communities": "communities.lance",
            "documents": "documents.lance",
            "text_units": "text_units.lance",
            "community_reports": "community_reports.lance",
        }

    def get_index_info(self) -> Dict[str, Any]:
        """獲取索引基本資訊。

        Returns:
            包含索引統計和詳細資訊的字典
        """
        try:
            # 基本資訊
            info = {
                "path": str(self.index_path),
                "created_at": self._get_creation_time(),
                "updated_at": self._get_update_time(),
                "statistics": self._get_basic_statistics(),
                "detailed_stats": self._get_detailed_statistics(),
                "file_info": self._get_file_info(),
            }

            return info

        except Exception as e:
            logger.error(f"獲取索引資訊失敗: {e}")
            raise

    def _get_creation_time(self) -> str:
        """獲取索引建立時間。"""
        try:
            # 嘗試從 artifacts 目錄的建立時間獲取
            if self.artifacts_path.exists():
                stat = self.artifacts_path.stat()
                return datetime.fromtimestamp(stat.st_ctime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            else:
                stat = self.index_path.stat()
                return datetime.fromtimestamp(stat.st_ctime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        except Exception:
            return "未知"

    def _get_update_time(self) -> str:
        """獲取索引最後更新時間。"""
        try:
            # 找到最新修改的檔案
            latest_time = 0
            for file_path in self.parquet_files.values():
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime

            if latest_time > 0:
                return datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d %H:%M:%S")
            else:
                return self._get_creation_time()
        except Exception:
            return "未知"

    def _get_basic_statistics(self) -> Dict[str, int]:
        """獲取基本統計資訊。"""
        stats = {}

        try:
            if self.has_lancedb:
                # 讀取 LanceDB 資料
                stats = self._get_lancedb_statistics()
            else:
                # 讀取各個 parquet 檔案的行數
                for name, file_path in self.parquet_files.items():
                    if file_path.exists():
                        try:
                            df = pd.read_parquet(file_path)
                            stats[name] = len(df)
                        except Exception as e:
                            logger.warning(f"讀取 {name} 檔案失敗: {e}")
                            stats[name] = 0
                    else:
                        stats[name] = 0

            # 計算總計
            stats["總文件數"] = stats.get("documents", 0)
            stats["總實體數"] = stats.get("entities", 0)
            stats["總關係數"] = stats.get("relationships", 0)
            stats["總社群數"] = stats.get("communities", 0)
            stats["總文本塊數"] = stats.get("text_units", 0)
            stats["社群報告數"] = stats.get("community_reports", 0)

        except Exception as e:
            logger.error(f"獲取基本統計失敗: {e}")
            stats["錯誤"] = f"統計失敗: {str(e)}"

        return stats

    def _get_lancedb_statistics(self) -> Dict[str, int]:
        """獲取 LanceDB 格式的統計資訊。"""
        stats = {}

        try:
            import lancedb

            # 連接到 LanceDB
            db = lancedb.connect(str(self.lancedb_path))

            # 獲取所有表格名稱
            table_names = db.table_names()

            for name, table_name in self.lance_tables.items():
                if table_name.replace(".lance", "") in table_names:
                    try:
                        table = db.open_table(table_name.replace(".lance", ""))
                        # 計算表格中的記錄數
                        count = table.count_rows()
                        stats[name] = count
                        logger.debug(f"LanceDB 表格 {table_name} 有 {count} 筆記錄")
                    except Exception as e:
                        logger.warning(f"讀取 LanceDB 表格 {table_name} 失敗: {e}")
                        stats[name] = 0
                else:
                    stats[name] = 0

        except ImportError:
            logger.error("無法導入 lancedb，請確保已安裝 lancedb 套件")
            for name in self.lance_tables.keys():
                stats[name] = 0
        except Exception as e:
            logger.error(f"讀取 LanceDB 統計失敗: {e}")
            for name in self.lance_tables.keys():
                stats[name] = 0

        return stats

    def _get_detailed_statistics(self) -> Dict[str, Dict[str, Any]]:
        """獲取詳細統計資訊。"""
        detailed_stats = {}

        try:
            if self.has_lancedb:
                # 使用 LanceDB 資料
                detailed_stats = self._get_lancedb_detailed_statistics()
            else:
                # 實體統計
                entities_file = self.parquet_files["entities"]
                if entities_file.exists():
                    df = pd.read_parquet(entities_file)
                    detailed_stats["實體"] = self._analyze_entities(df)

                # 關係統計
                relationships_file = self.parquet_files["relationships"]
                if relationships_file.exists():
                    df = pd.read_parquet(relationships_file)
                    detailed_stats["關係"] = self._analyze_relationships(df)

                # 社群統計
                communities_file = self.parquet_files["communities"]
                if communities_file.exists():
                    df = pd.read_parquet(communities_file)
                    detailed_stats["社群"] = self._analyze_communities(df)

        except Exception as e:
            logger.error(f"獲取詳細統計失敗: {e}")
            detailed_stats["錯誤"] = {"訊息": str(e)}

        return detailed_stats

    def _get_lancedb_detailed_statistics(self) -> Dict[str, Dict[str, Any]]:
        """獲取 LanceDB 格式的詳細統計資訊。"""
        detailed_stats = {}

        try:
            import lancedb

            # 連接到 LanceDB
            db = lancedb.connect(str(self.lancedb_path))
            table_names = db.table_names()

            # 分析實體
            if "entities" in table_names:
                try:
                    table = db.open_table("entities")
                    df = table.to_pandas()
                    detailed_stats["實體"] = self._analyze_entities(df)
                except Exception as e:
                    logger.warning(f"分析 LanceDB 實體表格失敗: {e}")
                    detailed_stats["實體"] = {"錯誤": str(e)}

            # 分析關係
            if "relationships" in table_names:
                try:
                    table = db.open_table("relationships")
                    df = table.to_pandas()
                    detailed_stats["關係"] = self._analyze_relationships(df)
                except Exception as e:
                    logger.warning(f"分析 LanceDB 關係表格失敗: {e}")
                    detailed_stats["關係"] = {"錯誤": str(e)}

            # 分析社群
            if "communities" in table_names:
                try:
                    table = db.open_table("communities")
                    df = table.to_pandas()
                    detailed_stats["社群"] = self._analyze_communities(df)
                except Exception as e:
                    logger.warning(f"分析 LanceDB 社群表格失敗: {e}")
                    detailed_stats["社群"] = {"錯誤": str(e)}

        except ImportError:
            logger.error("無法導入 lancedb")
            detailed_stats["錯誤"] = {"訊息": "lancedb 套件未安裝"}
        except Exception as e:
            logger.error(f"獲取 LanceDB 詳細統計失敗: {e}")
            detailed_stats["錯誤"] = {"訊息": str(e)}

        return detailed_stats

    def _analyze_entities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析實體資料。"""
        stats = {}

        try:
            stats["總數"] = len(df)

            # 按類型統計（如果有 type 欄位）
            if "type" in df.columns:
                type_counts = df["type"].value_counts().to_dict()
                stats["按類型"] = type_counts

            # 按級別統計（如果有 level 欄位）
            if "level" in df.columns:
                level_counts = df["level"].value_counts().to_dict()
                stats["按級別"] = level_counts

        except Exception as e:
            logger.warning(f"分析實體失敗: {e}")
            stats["錯誤"] = str(e)

        return stats

    def _analyze_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析關係資料。"""
        stats = {}

        try:
            stats["總數"] = len(df)

            # 權重統計（如果有 weight 欄位）
            if "weight" in df.columns:
                weights = df["weight"]
                stats["權重統計"] = {
                    "平均": float(weights.mean()),
                    "最大": float(weights.max()),
                    "最小": float(weights.min()),
                    "標準差": float(weights.std()),
                }

            # 按級別統計
            if "level" in df.columns:
                level_counts = df["level"].value_counts().to_dict()
                stats["按級別"] = level_counts

        except Exception as e:
            logger.warning(f"分析關係失敗: {e}")
            stats["錯誤"] = str(e)

        return stats

    def _analyze_communities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析社群資料。"""
        stats = {}

        try:
            stats["總數"] = len(df)

            # 按級別統計
            if "level" in df.columns:
                level_counts = df["level"].value_counts().to_dict()
                stats["按級別"] = level_counts

            # 社群大小統計（如果有相關欄位）
            if "size" in df.columns:
                sizes = df["size"]
                stats["大小統計"] = {
                    "平均": float(sizes.mean()),
                    "最大": int(sizes.max()),
                    "最小": int(sizes.min()),
                }

        except Exception as e:
            logger.warning(f"分析社群失敗: {e}")
            stats["錯誤"] = str(e)

        return stats

    def _get_file_info(self) -> Dict[str, Any]:
        """獲取檔案資訊。"""
        file_info = {}

        try:
            # 檢查各個輸出檔案
            for name, file_path in self.parquet_files.items():
                if file_path.exists():
                    stat = file_path.stat()
                    file_info[name] = {
                        "存在": True,
                        "大小": stat.st_size,
                        "大小_可讀": self._format_file_size(stat.st_size),
                        "修改時間": datetime.fromtimestamp(stat.st_mtime).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                else:
                    file_info[name] = {
                        "存在": False,
                        "大小": 0,
                        "大小_可讀": "0 B",
                        "修改時間": "N/A",
                    }

        except Exception as e:
            logger.error(f"獲取檔案資訊失敗: {e}")
            file_info["錯誤"] = str(e)

        return file_info

    def _format_file_size(self, size_bytes: int) -> str:
        """格式化檔案大小。"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math

        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def get_entity_summary(self) -> Dict[str, Any]:
        """獲取實體摘要資訊。"""
        entities_file = self.parquet_files["entities"]
        if not entities_file.exists():
            return {"錯誤": "實體檔案不存在"}

        try:
            df = pd.read_parquet(entities_file)

            summary = {
                "總實體數": len(df),
                "欄位": list(df.columns),
                "樣本資料": df.head(5).to_dict("records") if len(df) > 0 else [],
            }

            return summary

        except Exception as e:
            logger.error(f"獲取實體摘要失敗: {e}")
            return {"錯誤": str(e)}

    def get_relationship_summary(self) -> Dict[str, Any]:
        """獲取關係摘要資訊。"""
        relationships_file = self.parquet_files["relationships"]
        if not relationships_file.exists():
            return {"錯誤": "關係檔案不存在"}

        try:
            df = pd.read_parquet(relationships_file)

            summary = {
                "總關係數": len(df),
                "欄位": list(df.columns),
                "樣本資料": df.head(5).to_dict("records") if len(df) > 0 else [],
            }

            return summary

        except Exception as e:
            logger.error(f"獲取關係摘要失敗: {e}")
            return {"錯誤": str(e)}
