"""
数据库配置模块
使用 MySQL + SQLAlchemy 进行数据持久化
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# MySQL 连接配置
# 格式: mysql+pymysql://用户名:密码@主机:端口/数据库名?charset=utf8mb4
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:123456@localhost:3306/agent_finetune?charset=utf8mb4"
)

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    pool_size=5,           # 连接池大小
    max_overflow=10,       # 最大溢出连接数
    pool_pre_ping=True,    # 自动检测断开的连接
    pool_recycle=3600,     # 连接回收时间(秒)，防止连接超时
    pool_timeout=30,       # 连接超时时间(秒)
    echo=False             # 设为 True 可打印 SQL 语句，用于调试
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建声明式基类
Base = declarative_base()


def get_db():
    """
    获取数据库会话的依赖函数
    用于 FastAPI 的依赖注入
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    初始化数据库
    创建所有表
    """
    from db_models import FinetuneTask, ChatHistory, Agent, Model  # 避免循环导入
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表创建成功")


def drop_db():
    """
    删除所有表（谨慎使用）
    """
    from db_models import FinetuneTask, ChatHistory, Agent, Model
    Base.metadata.drop_all(bind=engine)
    print("⚠️ 所有数据库表已删除")
