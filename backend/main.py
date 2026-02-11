"""
Agent 微调平台 API 主入口

模块化结构:
├── main.py              # 主入口（当前文件）
├── routers/             # API 路由模块
│   ├── chat.py          # 聊天相关
│   ├── agents.py        # Agent 管理
│   ├── finetune.py      # 微调任务
│   ├── models.py        # 模型管理
│   ├── datasets.py      # 数据集管理
│   ├── gpu.py           # GPU 状态
│   └── websocket.py     # WebSocket 实时通信
├── services/            # 业务逻辑层
│   ├── finetune_service.py   # 微调任务逻辑
│   └── websocket_manager.py  # WebSocket 连接管理
├── schemas/             # 数据模型定义
│   └── requests.py      # 请求/响应 Pydantic 模型
├── database.py          # 数据库配置
├── db_models.py         # 数据库 ORM 模型
└── crud.py              # 数据库 CRUD 操作
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from services.finetune_service import task_cancel_flags

# 导入所有路由
from routers import (
    chat_router,
    agents_router,
    finetune_router,
    models_router,
    datasets_router,
    gpu_router,
    websocket_router,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    # 启动时初始化
    try:
        init_db()
        logger.info("✅ 数据库初始化成功")
    except Exception as e:
        logger.error(f"⚠️ 数据库初始化失败: {e}")
        logger.error("请确保 MySQL 已启动并创建了数据库 agent_finetune")
    
    yield  # 应用运行中
    
    # 关闭时清理
    logger.info("🔄 应用正在关闭，清理资源...")
    # 取消所有运行中的任务
    for task_id in list(task_cancel_flags.keys()):
        task_cancel_flags[task_id] = True
    logger.info("✅ 应用已关闭")


# 创建 FastAPI 应用实例
app = FastAPI(
    title="Agent 微调平台",
    description="基于 BERT 的文本分类模型微调平台 API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*",  # 开发环境允许所有来源
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
app.include_router(agents_router)
app.include_router(finetune_router)
app.include_router(models_router)
app.include_router(datasets_router)
app.include_router(gpu_router)
app.include_router(websocket_router)


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "Agent Finetune Platform API is running",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
