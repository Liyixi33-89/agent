"""
模型管理路由
"""
import os
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from schemas.requests import PredictRequest
from database import get_db
import crud

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["模型管理"])

# 模型目录
MODELS_DIR = "models"


# ==================== Pydantic 模型 ====================

class BatchPredictRequest(BaseModel):
    """批量预测请求"""
    texts: List[str]
    model_path: str
    base_model: str = "bert-base-uncased"


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    path: str
    size_mb: float
    created_at: str
    base_model: Optional[str] = None
    num_labels: Optional[int] = None


# ==================== 路由端点 ====================

@router.get("/finetuned")
async def list_finetuned_models(db: Session = Depends(get_db)):
    """获取所有微调后的模型（数据库记录）"""
    models = crud.get_all_models(db, model_type="finetuned")
    return [model.to_dict() for model in models]


@router.get("/local")
async def list_local_models():
    """扫描本地 models 目录，获取所有模型文件"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        return []
    
    models = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.pth'):
            filepath = os.path.join(MODELS_DIR, filename)
            stat = os.stat(filepath)
            
            # 尝试读取模型元数据
            model_info = {
                "name": filename.replace('.pth', ''),
                "filename": filename,
                "path": filepath,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
            
            # 尝试加载模型元数据（不加载完整权重）
            try:
                import torch
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                if 'config' in checkpoint:
                    model_info['base_model'] = checkpoint['config']._name_or_path
                if 'num_labels' in checkpoint:
                    model_info['num_labels'] = checkpoint['num_labels']
            except Exception as e:
                logger.warning(f"无法读取模型元数据 {filename}: {e}")
            
            models.append(model_info)
    
    # 按修改时间倒序排列
    models.sort(key=lambda x: x['modified_at'], reverse=True)
    return models


@router.get("/local/{model_name}")
async def get_local_model_info(model_name: str):
    """获取单个本地模型的详细信息"""
    filepath = os.path.join(MODELS_DIR, f"{model_name}.pth")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")
    
    import torch
    
    try:
        stat = os.stat(filepath)
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        return {
            "name": model_name,
            "path": filepath,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "base_model": checkpoint.get('config', {})._name_or_path if 'config' in checkpoint else None,
            "num_labels": checkpoint.get('num_labels'),
            "has_state_dict": 'model_state_dict' in checkpoint,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取模型失败: {str(e)}")


@router.delete("/local/{model_name}")
async def delete_local_model(model_name: str, db: Session = Depends(get_db)):
    """删除本地模型文件"""
    filepath = os.path.join(MODELS_DIR, f"{model_name}.pth")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")
    
    try:
        os.remove(filepath)
        
        # 同时删除数据库记录（如果存在）
        model = crud.get_model_by_name(db, model_name)
        if model:
            crud.delete_model(db, model.id)
        
        return {"message": f"模型 {model_name} 已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")


@router.post("/predict")
async def predict_with_model(req: PredictRequest):
    """使用微调后的模型进行预测"""
    import torch
    from modeling_bert import load_saved_model, load_tokenizer
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(req.model_path):
            raise HTTPException(status_code=404, detail=f"模型文件不存在: {req.model_path}")
        
        # 加载分词器
        tokenizer = load_tokenizer(req.base_model)
        
        # 使用 load_saved_model 函数加载模型
        model = load_saved_model(req.model_path, device='cpu')
        model.eval()
        
        # 对输入文本进行编码
        encoding = tokenizer(
            req.text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 进行预测
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            # 模型返回的是字典，包含 'logits' 键
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            "text": req.text,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities[0].tolist()
        }
        
    except Exception as e:
        import traceback
        error_detail = f"预测失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.post("/predict/batch")
async def batch_predict_with_model(req: BatchPredictRequest):
    """批量预测 - 一次预测多个文本"""
    import torch
    from modeling_bert import load_saved_model, load_tokenizer
    
    try:
        if not os.path.exists(req.model_path):
            raise HTTPException(status_code=404, detail=f"模型文件不存在: {req.model_path}")
        
        if len(req.texts) > 100:
            raise HTTPException(status_code=400, detail="单次批量预测最多支持 100 条文本")
        
        tokenizer = load_tokenizer(req.base_model)
        model = load_saved_model(req.model_path, device='cpu')
        model.eval()
        
        results = []
        
        with torch.no_grad():
            for text in req.texts:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                outputs = model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask']
                )
                
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
                
                results.append({
                    "text": text,
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": probabilities[0].tolist()
                })
        
        return {
            "model_path": req.model_path,
            "total": len(results),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"批量预测失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


@router.get("/available-base")
async def list_available_base_models():
    """获取可用的基础模型列表（用于微调）"""
    return [
        {
            "name": "bert-base-uncased",
            "description": "BERT 基础版（无大小写区分）",
            "parameters": "110M",
            "recommended": True
        },
        {
            "name": "bert-base-chinese",
            "description": "BERT 中文版",
            "parameters": "110M",
            "recommended": True
        },
        {
            "name": "bert-large-uncased",
            "description": "BERT 大型版（无大小写区分）",
            "parameters": "340M",
            "recommended": False
        },
        {
            "name": "distilbert-base-uncased",
            "description": "DistilBERT 精简版（速度更快）",
            "parameters": "66M",
            "recommended": True
        },
        {
            "name": "roberta-base",
            "description": "RoBERTa 基础版",
            "parameters": "125M",
            "recommended": False
        }
    ]
