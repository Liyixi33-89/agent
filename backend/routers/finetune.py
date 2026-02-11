"""
微调任务路由
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from schemas.requests import FinetuneRequest
from services.finetune_service import run_finetune_task, task_cancel_flags
from database import get_db
from db_models import TaskStatus
import crud

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/finetune", tags=["微调任务"])


@router.post("")
async def start_finetune(
    req: FinetuneRequest, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    """启动微调任务"""
    # 创建任务记录到数据库
    task = crud.create_finetune_task(
        db=db,
        base_model=req.base_model,
        new_model_name=req.new_model_name,
        dataset_path=req.dataset_path,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
        max_length=req.max_length,
        gradient_accumulation_steps=req.gradient_accumulation_steps,
        text_column=req.text_column,
        label_column=req.label_column,
        use_gpu=req.use_gpu
    )
    
    task_id = str(task.id)
    
    # 在后台运行微调任务
    background_tasks.add_task(run_finetune_task, task_id, req)
    
    return {"task_id": task_id, "status": "started"}


@router.get("")
async def list_finetune_tasks(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取所有微调任务列表"""
    tasks = crud.get_all_finetune_tasks(db, skip=skip, limit=limit, status=status)
    return [task.to_dict() for task in tasks]


@router.get("/{task_id}")
async def get_finetune_status(task_id: str, db: Session = Depends(get_db)):
    """获取微调任务状态"""
    task = crud.get_finetune_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task.to_dict()


@router.delete("/{task_id}")
async def delete_finetune_task(task_id: str, db: Session = Depends(get_db)):
    """删除微调任务"""
    success = crud.delete_finetune_task(db, task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "deleted"}


@router.post("/{task_id}/cancel")
async def cancel_finetune_task(task_id: str, db: Session = Depends(get_db)):
    """取消正在运行的微调任务"""
    task = crud.get_finetune_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status not in [TaskStatus.RUNNING.value, TaskStatus.PENDING.value]:
        raise HTTPException(status_code=400, detail=f"无法取消状态为 {task.status} 的任务")
    
    # 设置取消标志
    if task_id in task_cancel_flags:
        task_cancel_flags[task_id] = True
        logger.info(f"任务 {task_id} 已标记为取消")
        return {"status": "cancelling", "message": "任务正在取消中，请稍候..."}
    else:
        # 任务可能还未开始运行，直接更新状态
        crud.update_finetune_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.CANCELLED.value,
            error_message="任务在启动前被取消"
        )
        return {"status": "cancelled", "message": "任务已取消"}
