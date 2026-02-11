"""
微调任务服务层
处理微调任务的核心逻辑
"""
import os
import logging
import threading
from typing import Dict

from schemas.requests import FinetuneRequest
from db_models import TaskStatus
import crud

logger = logging.getLogger(__name__)

# 存储运行中的任务线程，用于取消功能
running_tasks: Dict[str, threading.Thread] = {}
task_cancel_flags: Dict[str, bool] = {}


def run_finetune_task_sync(task_id: str, req: FinetuneRequest):
    """同步运行微调任务"""
    import torch
    from database import SessionLocal
    from utils_data import load_csv_data, load_json_data, create_data_loader, split_data
    from modeling_bert import load_model, load_tokenizer, save_model
    from trainer import train_model
    
    # 创建新的数据库会话（因为在线程中）
    db = SessionLocal()
    
    # 初始化取消标志
    task_cancel_flags[task_id] = False
    
    # 根据配置和硬件情况决定使用的设备
    if req.use_gpu and torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 使用 GPU 训练: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        if req.use_gpu and not torch.cuda.is_available():
            logger.warning("⚠️ 请求使用 GPU 但 CUDA 不可用，回退到 CPU 训练")
        else:
            logger.info("📌 使用 CPU 训练")
    
    # 进度回调函数
    def progress_callback(current_epoch: int, total_epochs: int, progress: float):
        """更新训练进度到数据库"""
        # 检查取消标志
        if task_cancel_flags.get(task_id, False):
            raise InterruptedError("任务已被用户取消")
        
        try:
            crud.update_finetune_task_status(
                db=db,
                task_id=task_id,
                status=TaskStatus.RUNNING.value,
                progress=progress
            )
            logger.info(f"Task {task_id}: Epoch {current_epoch}/{total_epochs}, Progress: {progress:.1f}%")
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    try:
        # 更新任务状态为运行中
        crud.update_finetune_task_status(db, task_id, TaskStatus.RUNNING.value, progress=0.0)
        
        logger.info(f"Starting finetune task {task_id} for model {req.new_model_name}...")
        
        # 加载数据
        if req.dataset_path.endswith('.csv'):
            texts, labels = load_csv_data(req.dataset_path, req.text_column, req.label_column)
        elif req.dataset_path.endswith('.json'):
            texts, labels = load_json_data(req.dataset_path, req.text_column, req.label_column)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # 划分数据集
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(
            texts, labels, train_ratio=0.8, val_ratio=0.1
        )
        
        # 加载分词器和模型
        tokenizer = load_tokenizer(req.base_model)
        model = load_model(req.base_model, num_labels=len(set(labels)))
        
        # 创建数据加载器
        train_loader = create_data_loader(
            train_texts, train_labels, tokenizer,
            batch_size=req.batch_size, max_length=req.max_length, shuffle=True
        )
        val_loader = create_data_loader(
            val_texts, val_labels, tokenizer,
            batch_size=req.batch_size, max_length=req.max_length
        )
        
        # 训练模型（带进度回调、设备配置和显存优化）
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            progress_callback=progress_callback,
            device=device,
            gradient_accumulation_steps=req.gradient_accumulation_steps,
            use_amp=(device == "cuda")
        )
        
        # 保存模型
        model_path = f"models/{req.new_model_name}.pth"
        os.makedirs("models", exist_ok=True)
        save_model(model, model_path)
        
        # 更新任务状态为完成
        crud.update_finetune_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.COMPLETED.value,
            model_path=model_path,
            training_history=history,
            progress=100.0
        )
        
        # 创建模型记录
        crud.create_model(
            db=db,
            name=req.new_model_name,
            model_type="finetuned",
            base_model=req.base_model,
            path=model_path,
            description=f"从 {req.base_model} 微调得到",
            finetune_task_id=task_id
        )
        
        logger.info(f"Finetune task {task_id} completed. Model saved to {model_path}")
        
    except InterruptedError as e:
        # 任务被取消
        crud.update_finetune_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.CANCELLED.value,
            error_message=str(e)
        )
        logger.warning(f"Finetune task {task_id} cancelled: {str(e)}")
    except Exception as e:
        # 更新任务状态为失败
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        crud.update_finetune_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.FAILED.value,
            error_message=error_detail
        )
        logger.error(f"Finetune task {task_id} failed: {str(e)}")
    finally:
        # 清理取消标志
        if task_id in task_cancel_flags:
            del task_cancel_flags[task_id]
        if task_id in running_tasks:
            del running_tasks[task_id]
        db.close()


async def run_finetune_task(task_id: str, req: FinetuneRequest):
    """异步运行微调任务"""
    # 在线程中运行同步任务
    thread = threading.Thread(target=run_finetune_task_sync, args=(task_id, req))
    thread.start()
    # 记录运行中的任务
    running_tasks[task_id] = thread
