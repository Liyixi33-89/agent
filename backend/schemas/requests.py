"""
API 请求和响应的 Pydantic 模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """聊天请求"""
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    session_id: Optional[str] = None  # 会话ID用于保存聊天历史


class FinetuneRequest(BaseModel):
    """微调请求"""
    base_model: str
    dataset_path: str
    new_model_name: str
    epochs: int = Field(default=3, ge=1, le=100, description="训练轮数")
    learning_rate: float = Field(default=2e-5, gt=0, description="学习率")
    batch_size: int = Field(default=8, ge=1, le=64, description="批次大小")
    max_length: int = Field(default=128, ge=32, le=512, description="最大序列长度")
    text_column: str = "text"
    label_column: str = "target"
    use_gpu: bool = True  # 是否使用GPU加速
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32, description="梯度累积步数")


class AgentConfig(BaseModel):
    """Agent 配置"""
    name: str
    role: str
    system_prompt: str
    model: str
    config: Optional[Dict] = None


class PredictRequest(BaseModel):
    """模型预测请求"""
    model_path: str
    text: str
    base_model: str = "bert-base-uncased"
