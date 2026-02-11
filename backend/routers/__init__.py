"""
API 路由模块
"""
from .chat import router as chat_router
from .agents import router as agents_router
from .finetune import router as finetune_router
from .models import router as models_router
from .datasets import router as datasets_router
from .gpu import router as gpu_router
from .websocket import router as websocket_router

__all__ = [
    "chat_router",
    "agents_router",
    "finetune_router",
    "models_router",
    "datasets_router",
    "gpu_router",
    "websocket_router",
]
