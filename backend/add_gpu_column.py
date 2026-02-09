"""添加 use_gpu 列到 finetune_tasks 表"""
import pymysql

conn = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    database='agent_finetune'
)

cursor = conn.cursor()

# 检查列是否已存在
cursor.execute("SHOW COLUMNS FROM finetune_tasks LIKE 'use_gpu'")
if cursor.fetchone():
    print("use_gpu 列已存在")
else:
    # 添加列
    cursor.execute("""
        ALTER TABLE finetune_tasks 
        ADD COLUMN use_gpu TINYINT(1) DEFAULT 1 
        COMMENT '是否使用GPU加速'
    """)
    conn.commit()
    print("✅ use_gpu 列添加成功")

cursor.close()
conn.close()
