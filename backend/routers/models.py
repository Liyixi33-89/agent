"""
模型管理路由
"""
import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from schemas.requests import PredictRequest
from database import get_db
import crud

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["模型管理"])


@router.get("/finetuned")
async def list_finetuned_models(db: Session = Depends(get_db)):
    """获取所有微调后的模型"""
    models = crud.get_all_models(db, model_type="finetuned")
    return [model.to_dict() for model in models]


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
