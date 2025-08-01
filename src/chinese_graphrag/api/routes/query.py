#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查詢 API 路由

提供知識圖譜查詢和檢索功能。
"""

import uuid
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse

from ...config.settings import Settings
from ...monitoring.logger import get_logger
from ..models import (
    QueryRequest, QueryResponse, QueryResult, BatchQueryRequest, BatchQueryResponse,
    DataResponse, create_success_response, create_error_response, create_task_response,
    ResponseStatus
)


# 設定日誌
logger = get_logger(__name__)

# 創建路由器
router = APIRouter()

# 查詢任務儲存
query_tasks: Dict[str, Dict[str, Any]] = {}


@router.post("/query", response_model=QueryResponse, summary="執行單一查詢")
async def execute_query(
    request: QueryRequest,
    settings: Settings = Depends()
) -> QueryResponse:
    """執行單一知識圖譜查詢。
    
    支援多種查詢類型：
    - auto: 自動選擇最佳查詢方式
    - semantic: 語義相似性查詢
    - keyword: 關鍵字查詢
    - graph: 圖結構查詢
    
    Args:
        request: 查詢請求參數
        settings: 應用程式設定
        
    Returns:
        查詢結果回應
        
    Raises:
        HTTPException: 當查詢失敗時
    """
    try:
        start_time = time.time()
        
        logger.info(f"執行查詢: {request.query[:100]}...")
        
        # 驗證查詢參數
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查詢內容不能為空"
            )
        
        # 執行查詢
        results = await _execute_single_query(request, settings)
        
        processing_time = time.time() - start_time
        
        # 準備查詢資訊
        query_info = {
            "query": request.query,
            "query_type": request.query_type,
            "processing_time": round(processing_time, 3),
            "result_count": len(results),
            "max_results": request.max_results,
            "include_sources": request.include_sources,
            "filters": request.filters
        }
        
        logger.info(f"查詢完成，找到 {len(results)} 個結果，耗時 {processing_time:.3f}s")
        
        return QueryResponse(
            success=True,
            message=f"查詢完成，找到 {len(results)} 個結果",
            data=results,
            query_info=query_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查詢執行失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查詢執行失敗: {str(e)}"
        )


@router.post("/query/batch", response_model=BatchQueryResponse, summary="執行批次查詢")
async def execute_batch_query(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends()
) -> BatchQueryResponse:
    """執行批次查詢任務。
    
    對於大量查詢，此端點會建立背景任務並立即回傳任務 ID。
    您可以使用任務 ID 來查詢批次處理的進度和結果。
    
    Args:
        request: 批次查詢請求
        background_tasks: FastAPI 背景任務
        settings: 應用程式設定
        
    Returns:
        批次查詢任務回應
    """
    try:
        # 建立任務 ID
        task_id = str(uuid.uuid4())
        
        # 初始化任務狀態
        task_info = {
            "task_id": task_id,
            "status": ResponseStatus.PROCESSING,
            "progress": 0.0,
            "start_time": datetime.now(),
            "request": request.dict(),
            "results": [],
            "error": None,
            "total_queries": len(request.queries),
            "completed_queries": 0,
            "failed_queries": 0
        }
        
        # 儲存任務資訊
        query_tasks[task_id] = task_info
        
        # 啟動背景任務
        background_tasks.add_task(
            run_batch_query_task,
            task_id,
            request,
            settings
        )
        
        logger.info(f"已啟動批次查詢任務: {task_id}, 查詢數量: {len(request.queries)}")
        
        return BatchQueryResponse(
            success=True,
            message="批次查詢任務已啟動",
            task_id=task_id,
            status=ResponseStatus.PROCESSING,
            progress=0.0
        )
        
    except Exception as e:
        logger.error(f"建立批次查詢任務失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"建立批次查詢任務失敗: {str(e)}"
        )


@router.get("/query/batch/{task_id}", response_model=BatchQueryResponse, summary="查詢批次任務狀態")
async def get_batch_query_status(task_id: str) -> BatchQueryResponse:
    """查詢批次查詢任務的狀態和結果。
    
    Args:
        task_id: 批次查詢任務 ID
        
    Returns:
        批次查詢任務狀態和結果
        
    Raises:
        HTTPException: 當任務 ID 不存在時
    """
    try:
        if task_id not in query_tasks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"找不到批次查詢任務: {task_id}"
            )
        
        task_info = query_tasks[task_id]
        
        # 準備回應
        response = BatchQueryResponse(
            success=task_info["status"] != ResponseStatus.ERROR,
            message=_get_query_status_message(task_info),
            task_id=task_id,
            status=task_info["status"],
            progress=task_info["progress"]
        )
        
        # 如果任務完成，包含結果
        if task_info["status"] == ResponseStatus.SUCCESS:
            response.batch_results = task_info["results"]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查詢批次任務狀態失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查詢批次任務狀態失敗: {str(e)}"
        )


@router.get("/query/history", response_model=DataResponse, summary="查詢歷史記錄")
async def get_query_history(
    limit: int = 50,
    offset: int = 0,
    query_type: Optional[str] = None
) -> DataResponse:
    """取得查詢歷史記錄。
    
    Args:
        limit: 回傳結果數量限制
        offset: 結果偏移量
        query_type: 查詢類型過濾
        
    Returns:
        查詢歷史記錄列表
    """
    try:
        # TODO: 實作從資料庫讀取查詢歷史
        # 目前回傳模擬資料
        history = []
        
        # 從批次任務中提取歷史
        tasks = list(query_tasks.values())
        tasks.sort(key=lambda x: x["start_time"], reverse=True)
        
        for task in tasks[offset:offset + limit]:
            if query_type and task["request"].get("query_type") != query_type:
                continue
                
            history.append({
                "task_id": task["task_id"],
                "query_type": task["request"].get("query_type", "auto"),
                "total_queries": task["total_queries"],
                "completed_queries": task["completed_queries"],
                "failed_queries": task["failed_queries"],
                "status": task["status"],
                "start_time": task["start_time"],
                "end_time": task.get("end_time"),
                "processing_time": task.get("processing_time")
            })
        
        return create_success_response(
            data=history,
            message=f"找到 {len(history)} 筆查詢記錄",
            meta={
                "total": len(tasks),
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < len(tasks)
            }
        )
        
    except Exception as e:
        logger.error(f"取得查詢歷史失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取得查詢歷史失敗: {str(e)}"
        )


@router.get("/query/suggestions", response_model=DataResponse, summary="取得查詢建議")
async def get_query_suggestions(
    partial_query: str,
    limit: int = 10,
    settings: Settings = Depends()
) -> DataResponse:
    """根據部分查詢內容提供查詢建議。
    
    Args:
        partial_query: 部分查詢內容
        limit: 建議數量限制
        settings: 應用程式設定
        
    Returns:
        查詢建議列表
    """
    try:
        if len(partial_query.strip()) < 2:
            return create_success_response(
                data=[],
                message="查詢內容太短，無法提供建議"
            )
        
        # TODO: 實作智慧查詢建議
        # 目前回傳模擬資料
        suggestions = [
            f"{partial_query}的定義是什麼？",
            f"{partial_query}有哪些相關概念？",
            f"{partial_query}的應用場景有哪些？",
            f"如何理解{partial_query}？",
            f"{partial_query}與其他概念的關係"
        ]
        
        # 限制建議數量
        suggestions = suggestions[:limit]
        
        logger.info(f"為查詢 '{partial_query}' 提供了 {len(suggestions)} 個建議")
        
        return create_success_response(
            data=suggestions,
            message=f"為您提供了 {len(suggestions)} 個查詢建議"
        )
        
    except Exception as e:
        logger.error(f"取得查詢建議失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取得查詢建議失敗: {str(e)}"
        )


async def run_batch_query_task(task_id: str, request: BatchQueryRequest, settings: Settings):
    """執行批次查詢的背景任務。
    
    Args:
        task_id: 任務 ID
        request: 批次查詢請求
        settings: 應用程式設定
    """
    try:
        logger.info(f"開始執行批次查詢任務: {task_id}")
        
        task_info = query_tasks[task_id]
        task_info["status"] = ResponseStatus.PROCESSING
        
        results = []
        completed_count = 0
        failed_count = 0
        
        for i, query_text in enumerate(request.queries):
            try:
                # 檢查任務是否已取消
                if task_info["status"] == ResponseStatus.ERROR:
                    logger.info(f"批次查詢任務已取消: {task_id}")
                    return
                
                # 建立單一查詢請求
                single_request = QueryRequest(
                    query=query_text,
                    query_type=request.query_type,
                    max_results=request.max_results,
                    include_sources=request.include_sources
                )
                
                # 執行查詢
                query_results = await _execute_single_query(single_request, settings)
                
                # 建立查詢回應
                query_response = QueryResponse(
                    success=True,
                    message=f"查詢完成，找到 {len(query_results)} 個結果",
                    data=query_results,
                    query_info={
                        "query": query_text,
                        "query_type": request.query_type,
                        "result_count": len(query_results)
                    }
                )
                
                results.append(query_response)
                completed_count += 1
                
                # 更新進度
                progress = (completed_count / len(request.queries)) * 100
                task_info["progress"] = progress
                task_info["completed_queries"] = completed_count
                
                logger.debug(f"批次查詢進度: {completed_count}/{len(request.queries)}")
                
            except Exception as e:
                logger.error(f"批次查詢中的單一查詢失敗: {e}")
                failed_count += 1
                task_info["failed_queries"] = failed_count
                
                # 建立錯誤回應
                error_response = QueryResponse(
                    success=False,
                    message=f"查詢失敗: {str(e)}",
                    data=[],
                    query_info={
                        "query": query_text,
                        "query_type": request.query_type,
                        "error": str(e)
                    }
                )
                results.append(error_response)
        
        # 完成任務
        task_info["status"] = ResponseStatus.SUCCESS
        task_info["progress"] = 100.0
        task_info["end_time"] = datetime.now()
        task_info["results"] = results
        task_info["processing_time"] = (task_info["end_time"] - task_info["start_time"]).total_seconds()
        
        logger.info(
            f"批次查詢任務完成: {task_id}, "
            f"成功: {completed_count}, 失敗: {failed_count}"
        )
        
    except Exception as e:
        logger.error(f"批次查詢任務執行錯誤 {task_id}: {e}")
        task_info["status"] = ResponseStatus.ERROR
        task_info["error"] = str(e)
        task_info["end_time"] = datetime.now()


async def _execute_single_query(request: QueryRequest, settings: Settings) -> List[QueryResult]:
    """執行單一查詢。
    
    Args:
        request: 查詢請求
        settings: 應用程式設定
        
    Returns:
        查詢結果列表
    """
    # TODO: 實作實際的查詢邏輯
    # 這裡應該呼叫實際的查詢引擎
    
    # 模擬查詢結果
    results = []
    
    # 根據查詢類型產生不同的模擬結果
    if request.query_type == "semantic":
        results = _generate_semantic_results(request)
    elif request.query_type == "keyword":
        results = _generate_keyword_results(request)
    elif request.query_type == "graph":
        results = _generate_graph_results(request)
    else:  # auto
        results = _generate_auto_results(request)
    
    # 限制結果數量
    if request.max_results:
        results = results[:request.max_results]
    
    return results


def _generate_semantic_results(request: QueryRequest) -> List[QueryResult]:
    """產生語義查詢的模擬結果。"""
    return [
        QueryResult(
            id=f"semantic_{i}",
            content=f"與 '{request.query}' 語義相關的內容 {i}",
            score=0.9 - i * 0.1,
            source={
                "document_id": f"doc_{i}",
                "document_title": f"文件 {i}",
                "page": i + 1
            } if request.include_sources else None,
            metadata={
                "query_type": "semantic",
                "match_type": "semantic_similarity"
            }
        ) for i in range(min(5, request.max_results or 5))
    ]


def _generate_keyword_results(request: QueryRequest) -> List[QueryResult]:
    """產生關鍵字查詢的模擬結果。"""
    return [
        QueryResult(
            id=f"keyword_{i}",
            content=f"包含關鍵字 '{request.query}' 的內容 {i}",
            score=0.8 - i * 0.08,
            source={
                "document_id": f"doc_{i}",
                "document_title": f"文件 {i}",
                "page": i + 1
            } if request.include_sources else None,
            metadata={
                "query_type": "keyword",
                "match_type": "exact_match"
            }
        ) for i in range(min(3, request.max_results or 3))
    ]


def _generate_graph_results(request: QueryRequest) -> List[QueryResult]:
    """產生圖查詢的模擬結果。"""
    return [
        QueryResult(
            id=f"graph_{i}",
            content=f"與 '{request.query}' 在知識圖譜中相關的實體 {i}",
            score=0.85 - i * 0.05,
            source={
                "entity_id": f"entity_{i}",
                "entity_type": "concept",
                "relation": "related_to"
            } if request.include_sources else None,
            metadata={
                "query_type": "graph",
                "match_type": "graph_relation"
            }
        ) for i in range(min(4, request.max_results or 4))
    ]


def _generate_auto_results(request: QueryRequest) -> List[QueryResult]:
    """產生自動查詢的模擬結果。"""
    # 混合不同類型的結果
    results = []
    results.extend(_generate_semantic_results(request)[:2])
    results.extend(_generate_keyword_results(request)[:2])
    results.extend(_generate_graph_results(request)[:2])
    
    # 重新排序和調整分數
    for i, result in enumerate(results):
        result.id = f"auto_{i}"
        result.score = 0.95 - i * 0.05
        if result.metadata:
            result.metadata["query_type"] = "auto"
    
    return results


def _get_query_status_message(task_info: Dict[str, Any]) -> str:
    """取得查詢任務狀態訊息。
    
    Args:
        task_info: 任務資訊
        
    Returns:
        狀態訊息
    """
    status = task_info["status"]
    completed = task_info["completed_queries"]
    total = task_info["total_queries"]
    
    if status == ResponseStatus.PROCESSING:
        return f"正在處理批次查詢 {completed}/{total}"
    elif status == ResponseStatus.SUCCESS:
        failed = task_info["failed_queries"]
        if failed > 0:
            return f"批次查詢完成，成功 {completed} 個，失敗 {failed} 個"
        else:
            return f"批次查詢完成，共處理 {completed} 個查詢"
    elif status == ResponseStatus.ERROR:
        return f"批次查詢失敗：{task_info.get('error', '未知錯誤')}"
    else:
        return "未知狀態"