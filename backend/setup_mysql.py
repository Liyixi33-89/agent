"""
MySQL 数据库设置脚本
运行此脚本来检查 MySQL 连接并初始化数据库
"""

import subprocess
import sys

def install_dependencies():
    """安装 MySQL 相关依赖"""
    print("📦 正在安装 MySQL 相关依赖...")
    dependencies = ['pymysql', 'cryptography']
    for dep in dependencies:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
    print("✅ 依赖安装完成")


def check_mysql_connection(host='localhost', port=3306, user='root', password='123456'):
    """检查 MySQL 连接"""
    try:
        import pymysql
        print(f"\n🔍 正在尝试连接 MySQL ({host}:{port})...")
        
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
        print("✅ MySQL 连接成功！")
        connection.close()
        return True
    except ImportError:
        print("❌ pymysql 未安装，正在安装...")
        install_dependencies()
        return check_mysql_connection(host, port, user, password)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


def create_database(host='localhost', port=3306, user='root', password='123456', database='agent_finetune'):
    """创建数据库"""
    try:
        import pymysql
        print(f"\n📝 正在创建数据库 '{database}'...")
        
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
        
        cursor = connection.cursor()
        # 创建数据库（如果不存在）
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        connection.commit()
        
        print(f"✅ 数据库 '{database}' 创建成功！")
        
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        print(f"❌ 创建数据库失败: {e}")
        return False


def init_tables():
    """初始化数据库表"""
    try:
        print("\n📊 正在初始化数据库表...")
        from database import init_db
        init_db()
        print("✅ 数据库表初始化完成！")
        return True
    except Exception as e:
        print(f"❌ 初始化表失败: {e}")
        return False


def main():
    print("=" * 50)
    print("🔧 MySQL 数据库设置工具")
    print("=" * 50)
    
    # 配置信息
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '123456',  # 默认密码，请根据实际情况修改
        'database': 'agent_finetune'
    }
    
    print("\n📋 当前配置:")
    print(f"   主机: {config['host']}")
    print(f"   端口: {config['port']}")
    print(f"   用户: {config['user']}")
    print(f"   数据库: {config['database']}")
    
    # 步骤1: 检查连接
    if not check_mysql_connection(
        config['host'], config['port'], 
        config['user'], config['password']
    ):
        print("\n" + "=" * 50)
        print("⚠️  MySQL 连接失败！请检查以下内容：")
        print("=" * 50)
        print("""
1. 确保 MySQL 已安装并正在运行
   - Windows: 检查 MySQL 服务是否启动
   - 命令: services.msc 查看 MySQL 服务状态

2. 下载安装 MySQL (如果未安装):
   - 下载地址: https://dev.mysql.com/downloads/installer/
   - 选择 MySQL Installer for Windows
   - 安装时设置 root 密码为 123456 (或修改 database.py 中的配置)

3. 或者使用 Docker 快速启动 MySQL:
   docker run -d --name mysql \\
     -p 3306:3306 \\
     -e MYSQL_ROOT_PASSWORD=123456 \\
     mysql:8.0

4. 如果密码不同，请修改 backend/database.py 中的 DATABASE_URL
""")
        return False
    
    # 步骤2: 创建数据库
    if not create_database(
        config['host'], config['port'],
        config['user'], config['password'],
        config['database']
    ):
        return False
    
    # 步骤3: 初始化表
    if not init_tables():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 MySQL 数据库设置完成！")
    print("=" * 50)
    print("\n现在可以启动后端服务了:")
    print("   python main.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
