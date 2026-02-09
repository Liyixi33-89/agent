"""
数据库 CRUD 操作模块
封装所有数据库的增删改查操作
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uuid

from db_models import FinetuneTask, ChatHistory, Agent, Model, TaskStatus


# ==================== 微调任务相关操作 ====================

def create_finetune_task(
    db: Session,
    base_model: str,
    new_model_name: str,
    dataset_path: str,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    max_length: int = 512,
    text_column: str = "text",
    label_column: str = "target",
    use_gpu: bool = True
) -> FinetuneTask:
    """创建微调任务"""
    task = FinetuneTask(
        base_model=base_model,
        new_model_name=new_model_name,
        dataset_path=dataset_path,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        text_column=text_column,
        label_column=label_column,
        use_gpu=use_gpu,
        status=TaskStatus.PENDING.value
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_finetune_task(db: Session, task_id: str) -> Optional[FinetuneTask]:
    """根据 ID 获取微调任务"""
    return db.query(FinetuneTask).filter(FinetuneTask.id == task_id).first()


def get_all_finetune_tasks(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None
) -> List[FinetuneTask]:
    """获取所有微调任务"""
    query = db.query(FinetuneTask)
    if status:
        query = query.filter(FinetuneTask.status == status)
    return query.order_by(desc(FinetuneTask.created_at)).offset(skip).limit(limit).all()


def update_finetune_task_status(
    db: Session,
    task_id: str,
    status: str,
    error_message: Optional[str] = None,
    model_path: Optional[str] = None,
    training_history: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    progress: Optional[float] = None
) -> Optional[FinetuneTask]:
    """更新微调任务状态"""
    task = get_finetune_task(db, task_id)
    if not task:
        return None
    
    task.status = status
    
    if error_message is not None:
        task.error_message = error_message
    if model_path is not None:
        task.model_path = model_path
    if training_history is not None:
        task.training_history = training_history
    if metrics is not None:
        task.metrics = metrics
    if progress is not None:
        task.progress = progress
    
    # 更新时间戳
    if status == TaskStatus.RUNNING.value:
        task.started_at = datetime.utcnow()
    elif status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]:
        task.completed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(task)
    return task


def delete_finetune_task(db: Session, task_id: str) -> bool:
    """删除微调任务"""
    task = get_finetune_task(db, task_id)
    if not task:
        return False
    db.delete(task)
    db.commit()
    return True


# ==================== 聊天历史相关操作 ====================

def create_chat_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    model_used: Optional[str] = None,
    tokens_used: Optional[int] = None
) -> ChatHistory:
    """创建聊天消息"""
    message = ChatHistory(
        session_id=session_id,
        role=role,
        content=content,
        model_used=model_used,
        tokens_used=tokens_used
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_chat_history(
    db: Session,
    session_id: str,
    skip: int = 0,
    limit: int = 100
) -> List[ChatHistory]:
    """获取指定会话的聊天历史"""
    return db.query(ChatHistory)\
        .filter(ChatHistory.session_id == session_id)\
        .order_by(ChatHistory.created_at)\
        .offset(skip)\
        .limit(limit)\
        .all()


def get_all_sessions(db: Session) -> List[str]:
    """获取所有会话ID"""
    results = db.query(ChatHistory.session_id).distinct().all()
    return [r[0] for r in results]


def delete_chat_history(db: Session, session_id: str) -> int:
    """删除指定会话的聊天历史，返回删除的条数"""
    count = db.query(ChatHistory)\
        .filter(ChatHistory.session_id == session_id)\
        .delete()
    db.commit()
    return count


# ==================== Agent 相关操作 ====================

def create_agent(
    db: Session,
    name: str,
    role: str,
    system_prompt: str,
    model: str,
    config: Optional[Dict] = None
) -> Agent:
    """创建 Agent"""
    agent = Agent(
        name=name,
        role=role,
        system_prompt=system_prompt,
        model=model,
        config=config
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def get_agent(db: Session, agent_id: str) -> Optional[Agent]:
    """根据 ID 获取 Agent"""
    return db.query(Agent).filter(Agent.id == agent_id).first()


def get_agent_by_name(db: Session, name: str) -> Optional[Agent]:
    """根据名称获取 Agent"""
    return db.query(Agent).filter(Agent.name == name).first()


def get_all_agents(db: Session, active_only: bool = False) -> List[Agent]:
    """获取所有 Agent"""
    query = db.query(Agent)
    if active_only:
        query = query.filter(Agent.is_active == True)
    return query.order_by(desc(Agent.created_at)).all()


def update_agent(
    db: Session,
    agent_id: str,
    name: Optional[str] = None,
    role: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    is_active: Optional[bool] = None,
    config: Optional[Dict] = None
) -> Optional[Agent]:
    """更新 Agent"""
    agent = get_agent(db, agent_id)
    if not agent:
        return None
    
    if name is not None:
        agent.name = name
    if role is not None:
        agent.role = role
    if system_prompt is not None:
        agent.system_prompt = system_prompt
    if model is not None:
        agent.model = model
    if is_active is not None:
        agent.is_active = is_active
    if config is not None:
        agent.config = config
    
    db.commit()
    db.refresh(agent)
    return agent


def delete_agent(db: Session, agent_id: str) -> bool:
    """删除 Agent"""
    agent = get_agent(db, agent_id)
    if not agent:
        return False
    db.delete(agent)
    db.commit()
    return True


# ==================== 模型管理相关操作 ====================

def create_model(
    db: Session,
    name: str,
    model_type: str,
    base_model: Optional[str] = None,
    path: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict] = None,
    finetune_task_id: Optional[str] = None
) -> Model:
    """创建模型记录"""
    model = Model(
        name=name,
        model_type=model_type,
        base_model=base_model,
        path=path,
        description=description,
        parameters=parameters,
        finetune_task_id=finetune_task_id if finetune_task_id else None
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def get_model(db: Session, model_id: str) -> Optional[Model]:
    """根据 ID 获取模型"""
    return db.query(Model).filter(Model.id == model_id).first()


def get_model_by_name(db: Session, name: str) -> Optional[Model]:
    """根据名称获取模型"""
    return db.query(Model).filter(Model.name == name).first()


def get_all_models(
    db: Session,
    model_type: Optional[str] = None,
    status: Optional[str] = None
) -> List[Model]:
    """获取所有模型"""
    query = db.query(Model)
    if model_type:
        query = query.filter(Model.model_type == model_type)
    if status:
        query = query.filter(Model.status == status)
    return query.order_by(desc(Model.created_at)).all()


def update_model(
    db: Session,
    model_id: str,
    status: Optional[str] = None,
    metrics: Optional[Dict] = None,
    description: Optional[str] = None
) -> Optional[Model]:
    """更新模型"""
    model = get_model(db, model_id)
    if not model:
        return None
    
    if status is not None:
        model.status = status
    if metrics is not None:
        model.metrics = metrics
    if description is not None:
        model.description = description
    
    db.commit()
    db.refresh(model)
    return model


def delete_model(db: Session, model_id: str) -> bool:
    """删除模型"""
    model = get_model(db, model_id)
    if not model:
        return False
    db.delete(model)
    db.commit()
    return True
