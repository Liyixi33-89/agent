from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import os
import json
import asyncio
import threading
import time
from sqlalchemy.orm import Session

from utils_data import load_csv_data, load_json_data, create_data_loader, split_data
from modeling_bert import BertForTextClassification, load_model, load_tokenizer, save_model
from trainer import train_model, Trainer

# 导入数据库相关模块
from database import get_db, init_db, engine
from db_models import FinetuneTask, ChatHistory, Agent as AgentModel, Model as ModelRecord, TaskStatus
import crud

app = FastAPI(title="Agent 微调平台")

# 配置 CORS，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置 Ollama 地址
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 数据模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    session_id: Optional[str] = None  # 新增：会话ID用于保存聊天历史

class FinetuneRequest(BaseModel):
    base_model: str
    dataset_path: str
    new_model_name: str
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 32
    max_length: int = 512
    text_column: str = "text"
    label_column: str = "target"

class AgentConfig(BaseModel):
    name: str
    role: str
    system_prompt: str
    model: str
    config: Optional[Dict] = None


# 启动时初始化数据库
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库"""
    try:
        init_db()
        print("✅ 数据库初始化成功")
    except Exception as e:
        print(f"⚠️ 数据库初始化失败: {e}")
        print("请确保 PostgreSQL 已启动并创建了数据库")


@app.get("/")
async def root():
    return {"message": "Agent Finetune Platform API is running"}


# --- Ollama 代理接口 ---

@app.get("/api/models")
async def list_models():
    """获取 Ollama 中的本地模型"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to Ollama: {str(e)}")


@app.post("/api/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """与模型对话"""
    async with httpx.AsyncClient() as client:
        try:
            # 转发请求给 Ollama
            ollama_req = {
                "model": request.model,
                "messages": [msg.model_dump() for msg in request.messages],
                "stream": request.stream
            }
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=ollama_req, timeout=60.0)
            result = resp.json()
            
            # 如果提供了 session_id，保存聊天历史
            if request.session_id:
                # 保存用户最后一条消息
                if request.messages:
                    last_user_msg = request.messages[-1]
                    crud.create_chat_message(
                        db=db,
                        session_id=request.session_id,
                        role=last_user_msg.role,
                        content=last_user_msg.content,
                        model_used=request.model
                    )
                
                # 保存助手回复
                if "message" in result:
                    crud.create_chat_message(
                        db=db,
                        session_id=request.session_id,
                        role=result["message"].get("role", "assistant"),
                        content=result["message"].get("content", ""),
                        model_used=request.model
                    )
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# --- 聊天历史接口 ---

@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """获取指定会话的聊天历史"""
    messages = crud.get_chat_history(db, session_id)
    return [msg.to_dict() for msg in messages]


@app.get("/api/chat/sessions")
async def get_chat_sessions(db: Session = Depends(get_db)):
    """获取所有聊天会话"""
    sessions = crud.get_all_sessions(db)
    return {"sessions": sessions}


@app.delete("/api/chat/history/{session_id}")
async def delete_chat_history(session_id: str, db: Session = Depends(get_db)):
    """删除指定会话的聊天历史"""
    count = crud.delete_chat_history(db, session_id)
    return {"deleted": count}


# --- Agent 管理接口 ---

@app.post("/api/agents")
async def create_agent(agent: AgentConfig, db: Session = Depends(get_db)):
    """创建新的 Agent 配置"""
    try:
        db_agent = crud.create_agent(
            db=db,
            name=agent.name,
            role=agent.role,
            system_prompt=agent.system_prompt,
            model=agent.model,
            config=agent.config
        )
        return {"status": "success", "agent": db_agent.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"创建 Agent 失败: {str(e)}")


@app.get("/api/agents")
async def get_agents(db: Session = Depends(get_db)):
    """获取所有 Agent"""
    agents = crud.get_all_agents(db)
    return [agent.to_dict() for agent in agents]


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """获取指定 Agent"""
    agent = crud.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.to_dict()


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str, db: Session = Depends(get_db)):
    """删除 Agent"""
    success = crud.delete_agent(db, agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted"}


# --- 微调相关接口 ---

def run_finetune_task_sync(task_id: str, req: FinetuneRequest):
    """同步运行微调任务"""
    # 创建新的数据库会话（因为在线程中）
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        # 更新任务状态为运行中
        crud.update_finetune_task_status(db, task_id, TaskStatus.RUNNING.value)
        
        print(f"Starting finetune task {task_id} for model {req.new_model_name}...")
        
        # 加载数据
        if req.dataset_path.endswith('.csv'):
            texts, labels = load_csv_data(req.dataset_path, req.text_column, req.label_column)
        elif req.dataset_path.endswith('.json'):
            texts, labels = load_json_data(req.dataset_path, req.text_column, req.label_column)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # 划分数据集
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(
            texts, labels, train_ratio=0.8, val_ratio=0.1
        )
        
        # 加载分词器和模型
        tokenizer = load_tokenizer(req.base_model)
        model = load_model(req.base_model, num_labels=len(set(labels)))
        
        # 创建数据加载器
        train_loader = create_data_loader(train_texts, train_labels, tokenizer, 
                                         batch_size=req.batch_size, max_length=req.max_length, shuffle=True)
        val_loader = create_data_loader(val_texts, val_labels, tokenizer,
                                      batch_size=req.batch_size, max_length=req.max_length)
        
        # 训练模型
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=req.epochs,
            learning_rate=req.learning_rate
        )
        
        # 保存模型
        model_path = f"models/{req.new_model_name}.pth"
        os.makedirs("models", exist_ok=True)
        save_model(model, model_path)
        
        # 更新任务状态为完成
        crud.update_finetune_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.COMPLETED.value,
            model_path=model_path,
            training_history=history,
            progress=100.0
        )
        
        # 创建模型记录
        crud.create_model(
            db=db,
            name=req.new_model_name,
            model_type="finetuned",
            base_model=req.base_model,
            path=model_path,
            description=f"从 {req.base_model} 微调得到",
            finetune_task_id=task_id
        )
        
        print(f"Finetune task {task_id} completed. Model saved to {model_path}")
        
    except Exception as e:
        # 更新任务状态为失败
        crud.update_finetune_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.FAILED.value,
            error_message=str(e)
        )
        print(f"Finetune task {task_id} failed: {str(e)}")
    finally:
        db.close()


async def run_finetune_task(task_id: str, req: FinetuneRequest):
    """异步运行微调任务"""
    # 在线程中运行同步任务
    thread = threading.Thread(target=run_finetune_task_sync, args=(task_id, req))
    thread.start()


@app.post("/api/finetune")
async def start_finetune(req: FinetuneRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """启动微调任务"""
    # 创建任务记录到数据库
    task = crud.create_finetune_task(
        db=db,
        base_model=req.base_model,
        new_model_name=req.new_model_name,
        dataset_path=req.dataset_path,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
        max_length=req.max_length,
        text_column=req.text_column,
        label_column=req.label_column
    )
    
    task_id = str(task.id)
    
    # 在后台运行微调任务
    background_tasks.add_task(run_finetune_task, task_id, req)
    
    return {"task_id": task_id, "status": "started"}


@app.get("/api/finetune/{task_id}")
async def get_finetune_status(task_id: str, db: Session = Depends(get_db)):
    """获取微调任务状态"""
    task = crud.get_finetune_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task.to_dict()


@app.get("/api/finetune")
async def list_finetune_tasks(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取所有微调任务列表"""
    tasks = crud.get_all_finetune_tasks(db, skip=skip, limit=limit, status=status)
    return [task.to_dict() for task in tasks]


@app.delete("/api/finetune/{task_id}")
async def delete_finetune_task(task_id: str, db: Session = Depends(get_db)):
    """删除微调任务"""
    success = crud.delete_finetune_task(db, task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "deleted"}


# --- 模型管理接口 ---

@app.get("/api/models/finetuned")
async def list_finetuned_models(db: Session = Depends(get_db)):
    """获取所有微调后的模型"""
    models = crud.get_all_models(db, model_type="finetuned")
    return [model.to_dict() for model in models]


# --- 模型预测接口 ---

class PredictRequest(BaseModel):
    model_path: str
    text: str
    base_model: str = "bert-base-uncased"

@app.post("/api/models/predict")
async def predict_with_model(req: PredictRequest):
    """使用微调后的模型进行预测"""
    import torch
    from modeling_bert import load_saved_model
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(req.model_path):
            raise HTTPException(status_code=404, detail=f"模型文件不存在: {req.model_path}")
        
        # 加载分词器
        tokenizer = load_tokenizer(req.base_model)
        
        # 使用 load_saved_model 函数加载模型（它能正确处理保存的字典格式）
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
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
