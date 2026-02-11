"""
数据集管理路由
"""
import os
import json
import logging
import shutil
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["数据集管理"])

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


@router.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """上传数据集文件"""
    # 检查文件类型
    allowed_extensions = {".csv", ".json"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。请上传 .csv 或 .json 文件"
        )
    
    # 生成安全的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(DATA_DIR, safe_filename)
    
    try:
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 验证文件内容
        if file_ext == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            row_count = len(df)
            columns = list(df.columns)
        elif file_ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                row_count = len(data)
                columns = list(data[0].keys()) if data else []
            else:
                row_count = 1
                columns = list(data.keys())
        
        # 返回相对路径（相对于后端根目录）
        relative_path = f"data/{safe_filename}"
        
        logger.info(f"文件上传成功: {relative_path}, {row_count} 条记录")
        
        return {
            "status": "success",
            "file_path": relative_path,
            "file_name": safe_filename,
            "original_name": file.filename,
            "file_size": os.path.getsize(file_path),
            "row_count": row_count,
            "columns": columns
        }
        
    except Exception as e:
        # 如果处理失败，删除已上传的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@router.get("/datasets")
async def list_datasets():
    """列出所有已上传的数据集"""
    datasets = []
    
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in {".csv", ".json"}:
                stat = os.stat(file_path)
                datasets.append({
                    "name": filename,
                    "path": f"data/{filename}",
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": file_ext[1:]  # csv 或 json
                })
    
    # 按修改时间倒序排列
    datasets.sort(key=lambda x: x["modified"], reverse=True)
    return {"datasets": datasets}


@router.delete("/datasets/{filename}")
async def delete_dataset(filename: str):
    """删除数据集文件"""
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    try:
        os.remove(file_path)
        logger.info(f"数据集已删除: {filename}")
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
