"""
数据库模型定义
包含微调任务、聊天历史、Agent配置、模型管理等表
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON, Enum, Boolean, CHAR
from database import Base
import enum


def generate_uuid():
    """生成 UUID 字符串"""
    return str(uuid.uuid4())


class TaskStatus(str, enum.Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FinetuneTask(Base):
    """微调任务表"""
    __tablename__ = "finetune_tasks"

    id = Column(CHAR(36), primary_key=True, default=generate_uuid)
    
    # 基本信息
    base_model = Column(String(255), nullable=False, comment="基础模型名称")
    new_model_name = Column(String(255), nullable=False, comment="新模型名称")
    dataset_path = Column(String(512), nullable=False, comment="数据集路径")
    
    # 训练参数
    epochs = Column(Integer, default=3, comment="训练轮数")
    learning_rate = Column(Float, default=2e-5, comment="学习率")
    batch_size = Column(Integer, default=32, comment="批次大小")
    max_length = Column(Integer, default=512, comment="最大序列长度")
    text_column = Column(String(100), default="text", comment="文本列名")
    label_column = Column(String(100), default="target", comment="标签列名")
    use_gpu = Column(Boolean, default=True, comment="是否使用GPU加速")
    
    # 状态信息
    status = Column(String(20), default=TaskStatus.PENDING.value, comment="任务状态")
    progress = Column(Float, default=0.0, comment="任务进度 0-100")
    error_message = Column(Text, nullable=True, comment="错误信息")
    
    # 结果信息
    model_path = Column(String(512), nullable=True, comment="训练后模型路径")
    training_history = Column(JSON, nullable=True, comment="训练历史记录")
    metrics = Column(JSON, nullable=True, comment="评估指标")
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    started_at = Column(DateTime, nullable=True, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": str(self.id),
            "base_model": self.base_model,
            "new_model_name": self.new_model_name,
            "dataset_path": self.dataset_path,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "text_column": self.text_column,
            "label_column": self.label_column,
            "use_gpu": self.use_gpu,
            "status": self.status,
            "progress": self.progress,
            "error_message": self.error_message,
            "model_path": self.model_path,
            "training_history": self.training_history,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ChatHistory(Base):
    """聊天历史表"""
    __tablename__ = "chat_history"

    id = Column(CHAR(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(100), nullable=False, index=True, comment="会话ID")
    
    # 消息内容
    role = Column(String(20), nullable=False, comment="角色: user/assistant/system")
    content = Column(Text, nullable=False, comment="消息内容")
    
    # 元数据
    model_used = Column(String(255), nullable=True, comment="使用的模型")
    tokens_used = Column(Integer, nullable=True, comment="消耗的token数")
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Agent(Base):
    """Agent 配置表"""
    __tablename__ = "agents"

    id = Column(CHAR(36), primary_key=True, default=generate_uuid)
    
    # 基本信息
    name = Column(String(255), nullable=False, unique=True, comment="Agent名称")
    role = Column(String(100), nullable=False, comment="角色描述")
    system_prompt = Column(Text, nullable=False, comment="系统提示词")
    model = Column(String(255), nullable=False, comment="使用的模型")
    
    # 配置
    is_active = Column(Boolean, default=True, comment="是否激活")
    config = Column(JSON, nullable=True, comment="额外配置")
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": str(self.id),
            "name": self.name,
            "role": self.role,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "is_active": self.is_active,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Model(Base):
    """模型管理表"""
    __tablename__ = "models"

    id = Column(CHAR(36), primary_key=True, default=generate_uuid)
    
    # 基本信息
    name = Column(String(255), nullable=False, unique=True, comment="模型名称")
    base_model = Column(String(255), nullable=True, comment="基础模型")
    model_type = Column(String(50), nullable=False, comment="模型类型: ollama/finetuned/huggingface")
    
    # 路径和状态
    path = Column(String(512), nullable=True, comment="模型路径")
    status = Column(String(20), default="available", comment="状态: available/training/disabled")
    
    # 元数据
    description = Column(Text, nullable=True, comment="模型描述")
    parameters = Column(JSON, nullable=True, comment="模型参数")
    metrics = Column(JSON, nullable=True, comment="模型评估指标")
    
    # 关联
    finetune_task_id = Column(CHAR(36), nullable=True, comment="关联的微调任务ID")
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": str(self.id),
            "name": self.name,
            "base_model": self.base_model,
            "model_type": self.model_type,
            "path": self.path,
            "status": self.status,
            "description": self.description,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "finetune_task_id": str(self.finetune_task_id) if self.finetune_task_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
