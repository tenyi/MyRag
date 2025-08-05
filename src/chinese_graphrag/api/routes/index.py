#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
索引 API 路由

提供文件索引建立和管理功能。
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from ...config.settings import Settings
from ...monitoring.logger import get_logger
from ..models import (
    DataResponse,
    IndexRequest,
    IndexResponse,
    IndexStatus,
    ResponseStatus,
    create_error_response,
    create_success_response,
    create_task_response,
)

# 設定日誌
logger = get_logger(__name__)

# 創建路由器
router = APIRouter()

# 全域任務儲存（生產環境中應使用 Redis 或資料庫）
active_tasks: Dict[str, Dict[str, Any]] = {}


@router.post("/index", response_model=IndexResponse, summary="開始文件索引")
async def create_index(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(),
) -> IndexResponse:
    """開始文件索引建立程序。

    此端點接受文件路徑並在背景開始索引程序。
    索引程序是非同步的，您可以使用回傳的任務 ID 來查詢進度。

    Args:
        request: 索引請求參數
        background_tasks: FastAPI 背景任務
        settings: 應用程式設定

    Returns:
        索引任務回應，包含任務 ID 和初始狀態

    Raises:
        HTTPException: 當請求參數無效或系統錯誤時
    """
    try:
        # 驗證輸入路徑
        input_path = Path(request.input_path)
        if not input_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"輸入路徑不存在: {request.input_path}",
            )

        # 建立任務 ID
        task_id = str(uuid.uuid4())

        # 初始化任務狀態
        task_info = {
            "task_id": task_id,
            "status": ResponseStatus.PROCESSING,
            "progress": 0.0,
            "start_time": datetime.now(),
            "request": request.dict(),
            "result": None,
            "error": None,
            "index_status": IndexStatus(
                total_files=0,
                processed_files=0,
                failed_files=0,
                current_file=None,
                estimated_remaining_time=None,
            ),
        }

        # 儲存任務資訊
        active_tasks[task_id] = task_info

        # 啟動背景索引任務
        background_tasks.add_task(run_indexing_task, task_id, request, settings)

        logger.info(f"已啟動索引任務: {task_id}, 輸入路徑: {request.input_path}")

        return IndexResponse(
            success=True,
            message="索引任務已啟動",
            task_id=task_id,
            status=ResponseStatus.PROCESSING,
            progress=0.0,
            index_status=task_info["index_status"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"建立索引任務失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"建立索引任務失敗: {str(e)}",
        )


@router.get("/index/{task_id}", response_model=IndexResponse, summary="查詢索引進度")
async def get_index_status(task_id: str) -> IndexResponse:
    """查詢特定索引任務的進度狀態。

    Args:
        task_id: 索引任務 ID

    Returns:
        索引任務的當前狀態和進度資訊

    Raises:
        HTTPException: 當任務 ID 不存在時
    """
    try:
        if task_id not in active_tasks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"找不到任務 ID: {task_id}",
            )

        task_info = active_tasks[task_id]

        return IndexResponse(
            success=task_info["status"] != ResponseStatus.ERROR,
            message=_get_status_message(task_info["status"]),
            task_id=task_id,
            status=task_info["status"],
            progress=task_info["progress"],
            result=task_info.get("result"),
            index_status=task_info["index_status"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查詢索引狀態失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查詢索引狀態失敗: {str(e)}",
        )


@router.get("/index", response_model=DataResponse, summary="列出所有索引任務")
async def list_index_tasks(
    status_filter: Optional[str] = None, limit: int = 50, offset: int = 0
) -> DataResponse:
    """列出所有索引任務。

    Args:
        status_filter: 狀態過濾（可選）
        limit: 回傳結果數量限制
        offset: 結果偏移量

    Returns:
        索引任務列表
    """
    try:
        tasks = list(active_tasks.values())

        # 應用狀態過濾
        if status_filter:
            tasks = [t for t in tasks if t["status"] == status_filter]

        # 依開始時間排序（最新的在前）
        tasks.sort(key=lambda x: x["start_time"], reverse=True)

        # 應用分頁
        total = len(tasks)
        tasks = tasks[offset : offset + limit]

        # 準備回應資料
        task_summaries = []
        for task in tasks:
            task_summaries.append(
                {
                    "task_id": task["task_id"],
                    "status": task["status"],
                    "progress": task["progress"],
                    "start_time": task["start_time"],
                    "input_path": task["request"]["input_path"],
                    "total_files": task["index_status"].total_files,
                    "processed_files": task["index_status"].processed_files,
                    "failed_files": task["index_status"].failed_files,
                }
            )

        return create_success_response(
            data=task_summaries,
            message=f"找到 {total} 個索引任務",
            meta={
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
            },
        )

    except Exception as e:
        logger.error(f"列出索引任務失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"列出索引任務失敗: {str(e)}",
        )


@router.delete("/index/{task_id}", response_model=DataResponse, summary="取消索引任務")
async def cancel_index_task(task_id: str) -> DataResponse:
    """取消特定的索引任務。

    注意：只有處理中的任務可以被取消。

    Args:
        task_id: 索引任務 ID

    Returns:
        取消操作結果

    Raises:
        HTTPException: 當任務不存在或無法取消時
    """
    try:
        if task_id not in active_tasks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"找不到任務 ID: {task_id}",
            )

        task_info = active_tasks[task_id]

        if task_info["status"] not in [
            ResponseStatus.PROCESSING,
            ResponseStatus.PENDING,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"任務狀態為 {task_info['status']}，無法取消",
            )

        # 標記任務為已取消
        task_info["status"] = ResponseStatus.ERROR
        task_info["error"] = "任務已被用戶取消"
        task_info["progress"] = 0.0

        logger.info(f"索引任務已取消: {task_id}")

        return create_success_response(
            data={"task_id": task_id, "status": "cancelled"}, message="索引任務已取消"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消索引任務失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取消索引任務失敗: {str(e)}",
        )


@router.delete("/index", response_model=DataResponse, summary="清理完成的索引任務")
async def cleanup_index_tasks(
    cleanup_completed: bool = True,
    cleanup_failed: bool = False,
    older_than_hours: int = 24,
) -> DataResponse:
    """清理舊的索引任務記錄。

    Args:
        cleanup_completed: 是否清理已完成的任務
        cleanup_failed: 是否清理失敗的任務
        older_than_hours: 清理多少小時前的任務

    Returns:
        清理操作結果
    """
    try:
        current_time = datetime.now()
        cleanup_count = 0

        # 找出需要清理的任務
        tasks_to_remove = []
        for task_id, task_info in active_tasks.items():
            task_age_hours = (
                current_time - task_info["start_time"]
            ).total_seconds() / 3600

            if task_age_hours < older_than_hours:
                continue

            should_cleanup = False
            if cleanup_completed and task_info["status"] == ResponseStatus.SUCCESS:
                should_cleanup = True
            if cleanup_failed and task_info["status"] == ResponseStatus.ERROR:
                should_cleanup = True

            if should_cleanup:
                tasks_to_remove.append(task_id)

        # 移除任務
        for task_id in tasks_to_remove:
            del active_tasks[task_id]
            cleanup_count += 1

        logger.info(f"已清理 {cleanup_count} 個索引任務")

        return create_success_response(
            data={"cleaned_tasks": cleanup_count},
            message=f"已清理 {cleanup_count} 個索引任務",
        )

    except Exception as e:
        logger.error(f"清理索引任務失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清理索引任務失敗: {str(e)}",
        )


async def run_indexing_task(task_id: str, request: IndexRequest, settings: Settings):
    """執行索引任務的背景函數。

    Args:
        task_id: 任務 ID
        request: 索引請求參數
        settings: 應用程式設定
    """
    try:
        logger.info(f"開始執行索引任務: {task_id}")

        task_info = active_tasks[task_id]

        # 更新任務狀態為處理中
        task_info["status"] = ResponseStatus.PROCESSING
        task_info["progress"] = 0.0

        # 蒐集檔案列表
        await _update_task_status(task_id, "正在蒐集檔案...", 5.0)
        files = await _collect_files(request.input_path, request.file_types)

        if not files:
            await _complete_task_with_error(task_id, "未找到可處理的檔案")
            return

        # 更新檔案總數
        task_info["index_status"].total_files = len(files)

        # 開始處理檔案
        await _update_task_status(task_id, "正在處理檔案...", 10.0)

        processed_count = 0
        failed_count = 0

        for i, file_path in enumerate(files):
            try:
                # 檢查任務是否已被取消
                if task_info["status"] == ResponseStatus.ERROR:
                    logger.info(f"索引任務已取消: {task_id}")
                    return

                # 更新當前檔案
                task_info["index_status"].current_file = str(file_path)

                # 模擬檔案處理（實際實作中會呼叫索引引擎）
                await _process_file(file_path, request, settings)

                processed_count += 1
                task_info["index_status"].processed_files = processed_count

                # 更新進度
                progress = 10.0 + (processed_count / len(files)) * 80.0
                await _update_task_status(
                    task_id,
                    f"正在處理檔案 {processed_count}/{len(files)}: {file_path.name}",
                    progress,
                )

                # 模擬處理時間
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"處理檔案失敗 {file_path}: {e}")
                failed_count += 1
                task_info["index_status"].failed_files = failed_count

        # 完成任務
        if failed_count == 0:
            await _complete_task_successfully(
                task_id, f"索引建立完成，共處理 {processed_count} 個檔案"
            )
        else:
            await _complete_task_successfully(
                task_id,
                f"索引建立完成，處理 {processed_count} 個檔案，{failed_count} 個失敗",
            )

        logger.info(f"索引任務完成: {task_id}")

    except Exception as e:
        logger.error(f"索引任務執行錯誤 {task_id}: {e}")
        await _complete_task_with_error(task_id, str(e))


async def _collect_files(input_path: str, file_types: List[str] = None) -> List[Path]:
    """蒐集需要處理的檔案。

    Args:
        input_path: 輸入路徑
        file_types: 檔案類型過濾

    Returns:
        檔案路徑列表
    """
    path = Path(input_path)
    files = []

    if path.is_file():
        files.append(path)
    elif path.is_dir():
        # 預設支援的檔案類型
        default_types = [".txt", ".md", ".pdf", ".docx", ".doc"]
        allowed_types = file_types or default_types

        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in allowed_types:
                files.append(file_path)

    return files


async def _process_file(file_path: Path, request: IndexRequest, settings: Settings):
    """處理單個檔案。

    Args:
        file_path: 檔案路徑
        request: 索引請求
        settings: 應用程式設定
    """
    # TODO: 實作實際的檔案處理邏輯
    # 這裡應該呼叫實際的索引引擎來處理檔案
    logger.debug(f"處理檔案: {file_path}")

    # 模擬處理時間
    await asyncio.sleep(0.05)


async def _update_task_status(task_id: str, message: str, progress: float):
    """更新任務狀態。

    Args:
        task_id: 任務 ID
        message: 狀態訊息
        progress: 進度百分比
    """
    if task_id in active_tasks:
        active_tasks[task_id]["progress"] = progress
        active_tasks[task_id]["message"] = message


async def _complete_task_successfully(task_id: str, message: str):
    """標記任務為成功完成。

    Args:
        task_id: 任務 ID
        message: 完成訊息
    """
    if task_id in active_tasks:
        task_info = active_tasks[task_id]
        task_info["status"] = ResponseStatus.SUCCESS
        task_info["progress"] = 100.0
        task_info["message"] = message
        task_info["end_time"] = datetime.now()
        task_info["result"] = {
            "total_files": task_info["index_status"].total_files,
            "processed_files": task_info["index_status"].processed_files,
            "failed_files": task_info["index_status"].failed_files,
            "duration": (
                task_info["end_time"] - task_info["start_time"]
            ).total_seconds(),
        }


async def _complete_task_with_error(task_id: str, error_message: str):
    """標記任務為失敗。

    Args:
        task_id: 任務 ID
        error_message: 錯誤訊息
    """
    if task_id in active_tasks:
        task_info = active_tasks[task_id]
        task_info["status"] = ResponseStatus.ERROR
        task_info["error"] = error_message
        task_info["end_time"] = datetime.now()


def _get_status_message(status: ResponseStatus) -> str:
    """取得狀態訊息。

    Args:
        status: 任務狀態

    Returns:
        狀態訊息字串
    """
    status_messages = {
        ResponseStatus.PENDING: "任務等待中",
        ResponseStatus.PROCESSING: "任務處理中",
        ResponseStatus.SUCCESS: "任務完成",
        ResponseStatus.ERROR: "任務失敗",
    }
    return status_messages.get(status, "未知狀態")
