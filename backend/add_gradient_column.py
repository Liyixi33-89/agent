"""
添加 gradient_accumulation_steps 列到数据库
"""

import pymysql

def main():
    try:
        conn = pymysql.connect(
            host='localhost', 
            user='root', 
            password='123456', 
            database='agent_finetune'
        )
        cursor = conn.cursor()
        
        # 检查列是否存在
        cursor.execute('DESCRIBE finetune_tasks')
        columns = [row[0] for row in cursor.fetchall()]
        
        if 'gradient_accumulation_steps' not in columns:
            sql = """
            ALTER TABLE finetune_tasks 
            ADD COLUMN gradient_accumulation_steps INT DEFAULT 4 
            COMMENT '梯度累积步数'
            """
            cursor.execute(sql)
            conn.commit()
            print('✅ gradient_accumulation_steps 列添加成功')
        else:
            print('ℹ️ gradient_accumulation_steps 列已存在')
        
        # 验证
        cursor.execute('DESCRIBE finetune_tasks')
        columns = [row[0] for row in cursor.fetchall()]
        print('当前所有列:', columns)
        
        conn.close()
        
    except Exception as e:
        print(f'❌ 错误: {e}')

if __name__ == '__main__':
    main()
