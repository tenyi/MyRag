"""
社群報告生成器

為檢測到的社群生成詳細的分析報告
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from loguru import logger

from chinese_graphrag.models import Community, Entity, Relationship, TextUnit
from chinese_graphrag.config import GraphRAGConfig
from chinese_graphrag.config.strategy import ModelSelector, TaskType


class CommunityReportGenerator:
    """
    社群報告生成器
    
    使用 LLM 為每個社群生成詳細的分析報告，
    包括社群特徵、關鍵實體、重要關係等資訊
    """
    
    def __init__(self, config: GraphRAGConfig):
        """
        初始化社群報告生成器
        
        Args:
            config: GraphRAG 配置
        """
        self.config = config
        self.model_selector = ModelSelector(config)
        
        # 報告模板
        self.report_template = self._load_report_template()
        
        logger.info("初始化社群報告生成器")
    
    async def generate_community_reports(
        self,
        communities: List[Community],
        entities: Dict[str, Entity],
        relationships: Dict[str, Relationship],
        text_units: Optional[Dict[str, TextUnit]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        為所有社群生成報告
        
        Args:
            communities: 社群列表
            entities: 實體字典
            relationships: 關係字典
            text_units: 文本單元字典（可選）
            
        Returns:
            Dict[str, Dict[str, Any]]: 社群 ID 到報告的映射
        """
        logger.info(f"開始生成 {len(communities)} 個社群的報告")
        
        reports = {}
        
        # 選擇 LLM 模型
        llm_name, llm_config = self.model_selector.select_llm_model(
            TaskType.COMMUNITY_REPORT,
            context={"language": "zh", "communities_count": len(communities)}
        )
        
        logger.info(f"使用 LLM 模型生成社群報告: {llm_name}")
        
        # 批次處理社群
        batch_size = self.config.parallelization.batch_size
        
        for i in range(0, len(communities), batch_size):
            batch = communities[i:i + batch_size]
            
            # 並行生成報告
            batch_reports = await self._generate_batch_reports(
                batch, entities, relationships, text_units, llm_config
            )
            
            reports.update(batch_reports)
            
            logger.info(f"已生成 {min(i + batch_size, len(communities))}/{len(communities)} 個社群報告")
        
        logger.info(f"完成所有社群報告生成")
        return reports
    
    async def _generate_batch_reports(
        self,
        communities: List[Community],
        entities: Dict[str, Entity],
        relationships: Dict[str, Relationship],
        text_units: Optional[Dict[str, TextUnit]],
        llm_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """批次生成社群報告"""
        batch_reports = {}
        
        # 並行處理
        tasks = []
        for community in communities:
            task = self._generate_single_report(
                community, entities, relationships, text_units, llm_config
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for community, result in zip(communities, results):
                if isinstance(result, Exception):
                    logger.error(f"生成社群 {community.id} 報告失敗: {result}")
                    # 生成基本報告作為備用
                    batch_reports[community.id] = self._generate_basic_report(
                        community, entities, relationships
                    )
                else:
                    batch_reports[community.id] = result
                    
        except Exception as e:
            logger.error(f"批次生成社群報告失敗: {e}")
            # 回退到順序處理
            for community in communities:
                try:
                    report = await self._generate_single_report(
                        community, entities, relationships, text_units, llm_config
                    )
                    batch_reports[community.id] = report
                except Exception as single_error:
                    logger.error(f"生成社群 {community.id} 報告失敗: {single_error}")
                    batch_reports[community.id] = self._generate_basic_report(
                        community, entities, relationships
                    )
        
        return batch_reports
    
    async def _generate_single_report(
        self,
        community: Community,
        entities: Dict[str, Entity],
        relationships: Dict[str, Relationship],
        text_units: Optional[Dict[str, TextUnit]],
        llm_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成單個社群的報告"""
        try:
            # 收集社群相關資料
            community_data = self._collect_community_data(
                community, entities, relationships, text_units
            )
            
            # 生成 LLM 提示
            prompt = self._create_report_prompt(community, community_data)
            
            # 調用 LLM 生成報告
            if self.config.indexing.enable_llm_reports:
                llm_report = await self._call_llm_for_report(prompt, llm_config)
            else:
                llm_report = None
            
            # 建立完整報告
            report = self._build_complete_report(
                community, community_data, llm_report
            )
            
            return report
            
        except Exception as e:
            logger.error(f"生成社群 {community.id} 報告時發生錯誤: {e}")
            return self._generate_basic_report(community, entities, relationships)
    
    def _collect_community_data(
        self,
        community: Community,
        entities: Dict[str, Entity],
        relationships: Dict[str, Relationship],
        text_units: Optional[Dict[str, TextUnit]]
    ) -> Dict[str, Any]:
        """收集社群相關資料"""
        # 收集社群中的實體
        community_entities = []
        for entity_id in community.entities:
            if entity_id in entities:
                community_entities.append(entities[entity_id])
        
        # 收集社群中的關係
        community_relationships = []
        for rel_id in community.relationships:
            if rel_id in relationships:
                community_relationships.append(relationships[rel_id])
        
        # 收集相關的文本單元
        community_text_units = []
        if text_units:
            # 從實體和關係中找到相關的文本單元
            text_unit_ids = set()
            
            for entity in community_entities:
                text_unit_ids.update(entity.text_units)
            
            for rel in community_relationships:
                text_unit_ids.update(rel.text_units)
            
            for text_id in text_unit_ids:
                if text_id in text_units:
                    community_text_units.append(text_units[text_id])
        
        # 統計資訊
        entity_types = {}
        for entity in community_entities:
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        relationship_types = {}
        for rel in community_relationships:
            relationship_types[rel.relationship_type] = relationship_types.get(rel.relationship_type, 0) + 1
        
        # 找到最重要的實體
        top_entities = sorted(community_entities, key=lambda x: x.rank, reverse=True)[:5]
        
        # 找到最重要的關係
        top_relationships = sorted(community_relationships, key=lambda x: x.weight, reverse=True)[:5]
        
        return {
            "entities": community_entities,
            "relationships": community_relationships,
            "text_units": community_text_units,
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "top_entities": top_entities,
            "top_relationships": top_relationships,
            "entity_count": len(community_entities),
            "relationship_count": len(community_relationships),
            "text_unit_count": len(community_text_units)
        }
    
    def _create_report_prompt(
        self,
        community: Community,
        community_data: Dict[str, Any]
    ) -> str:
        """建立 LLM 報告生成提示"""
        prompt_parts = []
        
        # 基本資訊
        prompt_parts.append(f"請為以下社群生成詳細的分析報告：")
        prompt_parts.append(f"社群標題：{community.title}")
        prompt_parts.append(f"社群層級：{community.level}")
        prompt_parts.append(f"社群摘要：{community.summary}")
        prompt_parts.append("")
        
        # 實體資訊
        prompt_parts.append("## 社群實體")
        prompt_parts.append(f"總計 {community_data['entity_count']} 個實體")
        
        if community_data['entity_types']:
            prompt_parts.append("實體類型分布：")
            for etype, count in community_data['entity_types'].items():
                prompt_parts.append(f"- {etype}: {count}個")
        
        if community_data['top_entities']:
            prompt_parts.append("\n重要實體：")
            for entity in community_data['top_entities']:
                prompt_parts.append(f"- {entity.name} ({entity.type}): {entity.description}")
        
        # 關係資訊
        prompt_parts.append(f"\n## 社群關係")
        prompt_parts.append(f"總計 {community_data['relationship_count']} 個關係")
        
        if community_data['relationship_types']:
            prompt_parts.append("關係類型分布：")
            for rtype, count in community_data['relationship_types'].items():
                prompt_parts.append(f"- {rtype}: {count}個")
        
        if community_data['top_relationships']:
            prompt_parts.append("\n重要關係：")
            entity_dict = {e.id: e for e in community_data['entities']}
            
            for rel in community_data['top_relationships']:
                source_name = entity_dict.get(rel.source_entity_id, {}).name if rel.source_entity_id in entity_dict else "未知"
                target_name = entity_dict.get(rel.target_entity_id, {}).name if rel.target_entity_id in entity_dict else "未知"
                prompt_parts.append(f"- {source_name} → {target_name} ({rel.relationship_type}): {rel.description}")
        
        # 文本內容
        if community_data['text_units']:
            prompt_parts.append(f"\n## 相關文本內容")
            prompt_parts.append(f"來源於 {community_data['text_unit_count']} 個文本單元")
            
            # 選擇一些代表性的文本片段
            sample_texts = community_data['text_units'][:3]
            for i, text_unit in enumerate(sample_texts, 1):
                prompt_parts.append(f"\n文本片段 {i}：")
                prompt_parts.append(f"{text_unit.text[:200]}...")
        
        # 報告要求
        prompt_parts.append("\n## 報告要求")
        prompt_parts.append("請基於以上資訊生成一份詳細的社群分析報告，包括：")
        prompt_parts.append("1. 社群概述和主要特徵")
        prompt_parts.append("2. 關鍵實體分析")
        prompt_parts.append("3. 重要關係分析")
        prompt_parts.append("4. 社群在整體知識圖譜中的作用")
        prompt_parts.append("5. 潛在的應用價值和洞察")
        prompt_parts.append("\n請使用繁體中文撰寫報告，內容要專業且易於理解。")
        
        return "\n".join(prompt_parts)
    
    async def _call_llm_for_report(
        self,
        prompt: str,
        llm_config: Dict[str, Any]
    ) -> Optional[str]:
        """調用 LLM 生成報告"""
        try:
            # 這裡應該整合實際的 LLM 調用邏輯
            # 目前提供一個模擬實作
            
            logger.debug(f"調用 LLM 生成報告，提示長度: {len(prompt)}")
            
            # 模擬 LLM 調用延遲
            await asyncio.sleep(0.1)
            
            # 模擬生成的報告
            mock_report = """
# 社群分析報告

## 社群概述
本社群是一個重要的知識節點集合，包含了多個相關的實體和關係。社群內部的連接密度較高，顯示了強烈的主題相關性。

## 關鍵實體分析
社群中的關鍵實體展現了明確的主題聚焦，這些實體在整體知識圖譜中扮演重要角色。

## 重要關係分析
社群內的關係網絡顯示了實體間的複雜互動模式，這些關係有助於理解主題的內在結構。

## 社群作用
本社群在整體知識圖譜中起到了重要的橋樑作用，連接了不同的知識領域。

## 應用價值
本社群的分析結果可以用於相關領域的知識發現和決策支援。
            """.strip()
            
            return mock_report
            
        except Exception as e:
            logger.error(f"LLM 報告生成失敗: {e}")
            return None
    
    def _build_complete_report(
        self,
        community: Community,
        community_data: Dict[str, Any],
        llm_report: Optional[str]
    ) -> Dict[str, Any]:
        """建立完整的社群報告"""
        report = {
            "community_id": community.id,
            "title": community.title,
            "level": community.level,
            "generated_at": datetime.now().isoformat(),
            "summary": community.summary,
            "rank": community.rank,
            
            # 統計資訊
            "statistics": {
                "entity_count": community_data['entity_count'],
                "relationship_count": community_data['relationship_count'],
                "text_unit_count": community_data['text_unit_count'],
                "entity_types": community_data['entity_types'],
                "relationship_types": community_data['relationship_types']
            },
            
            # 關鍵實體
            "key_entities": [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "rank": entity.rank
                }
                for entity in community_data['top_entities']
            ],
            
            # 重要關係
            "key_relationships": [
                {
                    "id": rel.id,
                    "source_entity_id": rel.source_entity_id,
                    "target_entity_id": rel.target_entity_id,
                    "type": rel.relationship_type,
                    "description": rel.description,
                    "weight": rel.weight
                }
                for rel in community_data['top_relationships']
            ],
            
            # LLM 生成的報告
            "llm_report": llm_report,
            
            # 基本分析報告
            "basic_report": self._generate_basic_analysis(community, community_data),
            
            # 層次資訊
            "hierarchy": {
                "parent_community_id": community.parent_community_id,
                "child_communities": community.child_communities,
                "is_root": community.is_root_community,
                "is_leaf": community.is_leaf_community
            }
        }
        
        return report
    
    def _generate_basic_report(
        self,
        community: Community,
        entities: Dict[str, Entity],
        relationships: Dict[str, Relationship]
    ) -> Dict[str, Any]:
        """生成基本報告（不使用 LLM）"""
        community_data = self._collect_community_data(
            community, entities, relationships, None
        )
        
        return self._build_complete_report(community, community_data, None)
    
    def _generate_basic_analysis(
        self,
        community: Community,
        community_data: Dict[str, Any]
    ) -> str:
        """生成基本分析文本"""
        analysis_parts = []
        
        # 社群概述
        analysis_parts.append(f"## {community.title}")
        analysis_parts.append(f"本社群位於第 {community.level} 層，包含 {community_data['entity_count']} 個實體和 {community_data['relationship_count']} 個關係。")
        
        # 實體分析
        if community_data['entity_types']:
            analysis_parts.append("\n### 實體組成")
            for etype, count in community_data['entity_types'].items():
                percentage = (count / community_data['entity_count']) * 100
                analysis_parts.append(f"- {etype}: {count}個 ({percentage:.1f}%)")
        
        # 關係分析
        if community_data['relationship_types']:
            analysis_parts.append("\n### 關係類型")
            for rtype, count in community_data['relationship_types'].items():
                percentage = (count / community_data['relationship_count']) * 100 if community_data['relationship_count'] > 0 else 0
                analysis_parts.append(f"- {rtype}: {count}個 ({percentage:.1f}%)")
        
        # 重要性分析
        analysis_parts.append(f"\n### 重要性評估")
        analysis_parts.append(f"社群排名: {community.rank:.3f}")
        
        if community.rank >= 0.8:
            importance = "極高"
        elif community.rank >= 0.6:
            importance = "高"
        elif community.rank >= 0.4:
            importance = "中等"
        else:
            importance = "較低"
        
        analysis_parts.append(f"重要性等級: {importance}")
        
        # 層次結構分析
        if community.has_parent or community.has_children:
            analysis_parts.append(f"\n### 層次結構")
            if community.has_parent:
                analysis_parts.append(f"- 隸屬於上層社群")
            if community.has_children:
                analysis_parts.append(f"- 包含 {len(community.child_communities)} 個子社群")
        
        return "\n".join(analysis_parts)
    
    def _load_report_template(self) -> str:
        """載入報告模板"""
        # 這裡可以從檔案載入自訂的報告模板
        # 目前使用內建模板
        return """
# 社群分析報告

## 基本資訊
- 社群ID: {community_id}
- 標題: {title}
- 層級: {level}
- 排名: {rank}

## 統計資訊
{statistics}

## 關鍵實體
{key_entities}

## 重要關係
{key_relationships}

## 詳細分析
{detailed_analysis}
        """.strip()
    
    def export_reports_to_json(
        self,
        reports: Dict[str, Dict[str, Any]],
        output_path: str
    ) -> bool:
        """將報告匯出為 JSON 檔案"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(reports, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"成功匯出 {len(reports)} 個社群報告到 {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"匯出社群報告失敗: {e}")
            return False
    
    def export_reports_to_markdown(
        self,
        reports: Dict[str, Dict[str, Any]],
        output_dir: str
    ) -> bool:
        """將報告匯出為 Markdown 檔案"""
        try:
            from pathlib import Path
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for community_id, report in reports.items():
                # 建立 Markdown 內容
                md_content = self._format_report_as_markdown(report)
                
                # 寫入檔案
                filename = f"community_{community_id}.md"
                file_path = output_path / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
            
            logger.info(f"成功匯出 {len(reports)} 個社群報告到 {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"匯出 Markdown 報告失敗: {e}")
            return False
    
    def _format_report_as_markdown(self, report: Dict[str, Any]) -> str:
        """將報告格式化為 Markdown"""
        md_parts = []
        
        # 標題
        md_parts.append(f"# {report['title']}")
        md_parts.append(f"*生成時間: {report['generated_at']}*")
        md_parts.append("")
        
        # 基本資訊
        md_parts.append("## 基本資訊")
        md_parts.append(f"- **社群ID**: {report['community_id']}")
        md_parts.append(f"- **層級**: {report['level']}")
        md_parts.append(f"- **排名**: {report['rank']:.3f}")
        md_parts.append(f"- **摘要**: {report['summary']}")
        md_parts.append("")
        
        # 統計資訊
        stats = report['statistics']
        md_parts.append("## 統計資訊")
        md_parts.append(f"- **實體數量**: {stats['entity_count']}")
        md_parts.append(f"- **關係數量**: {stats['relationship_count']}")
        md_parts.append(f"- **文本單元數量**: {stats['text_unit_count']}")
        md_parts.append("")
        
        # 實體類型分布
        if stats['entity_types']:
            md_parts.append("### 實體類型分布")
            for etype, count in stats['entity_types'].items():
                md_parts.append(f"- {etype}: {count}個")
            md_parts.append("")
        
        # 關係類型分布
        if stats['relationship_types']:
            md_parts.append("### 關係類型分布")
            for rtype, count in stats['relationship_types'].items():
                md_parts.append(f"- {rtype}: {count}個")
            md_parts.append("")
        
        # 關鍵實體
        if report['key_entities']:
            md_parts.append("## 關鍵實體")
            for entity in report['key_entities']:
                md_parts.append(f"### {entity['name']} ({entity['type']})")
                md_parts.append(f"- **排名**: {entity['rank']:.3f}")
                md_parts.append(f"- **描述**: {entity['description']}")
                md_parts.append("")
        
        # 重要關係
        if report['key_relationships']:
            md_parts.append("## 重要關係")
            for rel in report['key_relationships']:
                md_parts.append(f"- **{rel['type']}** (權重: {rel['weight']:.3f})")
                md_parts.append(f"  - {rel['description']}")
            md_parts.append("")
        
        # LLM 報告
        if report.get('llm_report'):
            md_parts.append("## AI 分析報告")
            md_parts.append(report['llm_report'])
            md_parts.append("")
        
        # 基本分析
        if report.get('basic_report'):
            md_parts.append("## 基本分析")
            md_parts.append(report['basic_report'])
            md_parts.append("")
        
        # 層次資訊
        hierarchy = report['hierarchy']
        if hierarchy['parent_community_id'] or hierarchy['child_communities']:
            md_parts.append("## 層次結構")
            if hierarchy['parent_community_id']:
                md_parts.append(f"- **父社群**: {hierarchy['parent_community_id']}")
            if hierarchy['child_communities']:
                md_parts.append(f"- **子社群**: {', '.join(hierarchy['child_communities'])}")
            md_parts.append("")
        
        return "\n".join(md_parts)