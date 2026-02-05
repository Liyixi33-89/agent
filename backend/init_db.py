"""
数据库初始化脚本
用于创建数据库和表结构
"""

import os
import sys

def check_postgres_connection():
    """检查 PostgreSQL 连接"""
    from database import engine
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print("✅ PostgreSQL 连接成功")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL 连接失败: {e}")
        return False


def create_tables():
    """创建所有表"""
    from database import Base, engine
    from db_models import FinetuneTask, ChatHistory, Agent, Model
    
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ 数据库表创建成功")
        print("\n已创建的表:")
        print("  - finetune_tasks (微调任务表)")
        print("  - chat_history (聊天历史表)")
        print("  - agents (Agent配置表)")
        print("  - models (模型管理表)")
        return True
    except Exception as e:
        print(f"❌ 创建表失败: {e}")
        return False


def drop_tables():
    """删除所有表（危险操作）"""
    from database import Base, engine
    from db_models import FinetuneTask, ChatHistory, Agent, Model
    
    confirm = input("⚠️ 这将删除所有数据！确认删除？(yes/no): ")
    if confirm.lower() != "yes":
        print("操作已取消")
        return False
    
    try:
        Base.metadata.drop_all(bind=engine)
        print("✅ 所有表已删除")
        return True
    except Exception as e:
        print(f"❌ 删除表失败: {e}")
        return False


def show_db_status():
    """显示数据库状态"""
    from database import engine
    from sqlalchemy import inspect
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print("\n📊 数据库状态:")
        print(f"  数据库地址: {engine.url}")
        print(f"  已存在的表: {len(tables)}")
        for table in tables:
            print(f"    - {table}")
        return True
    except Exception as e:
        print(f"❌ 获取状态失败: {e}")
        return False


def main():
    print("=" * 50)
    print("     Agent 微调平台 - 数据库管理工具")
    print("=" * 50)
    print("\n请选择操作:")
    print("  1. 检查数据库连接")
    print("  2. 创建数据库表")
    print("  3. 显示数据库状态")
    print("  4. 删除所有表（危险）")
    print("  5. 退出")
    
    while True:
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            check_postgres_connection()
        elif choice == "2":
            create_tables()
        elif choice == "3":
            show_db_status()
        elif choice == "4":
            drop_tables()
        elif choice == "5":
            print("再见！")
            break
        else:
            print("无效选项，请重新输入")


if __name__ == "__main__":
    main()
