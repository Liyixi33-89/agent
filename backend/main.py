from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Set
from contextlib import asynccontextmanager
import httpx
import os
import json
import asyncio
import threading
import time
import logging
import shutil
from datetime import datetime
from sqlalchemy.orm import Session

from utils_data import load_csv_data, load_json_data, create_data_loader, split_data
from modeling_bert import BertForTextClassification, load_model, load_tokenizer, save_model
from trainer import train_model, Trainer

# 导入数据库相关模块
from database import get_db, init_db, engine
from db_models import FinetuneTask, ChatHistory, Agent as AgentModel, Model as ModelRecord, TaskStatus
import crud

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 存储运行中的任务线程，用于取消功能
running_tasks: Dict[str, threading.Thread] = {}
task_cancel_flags: Dict[str, bool] = {}

# WebSocket 连接管理
class ConnectionManager:
    """WebSocket 连接管理器"""
    def __init__(self):
        # task_id -> set of websocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = set()
        self.active_connections[task_id].add(websocket)
        logger.info(f"WebSocket connected for task {task_id}")
    
    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        logger.info(f"WebSocket disconnected for task {task_id}")
    
    async def send_log(self, task_id: str, message: str, level: str = "info"):
        """发送日志消息到所有订阅该任务的客户端"""
        if task_id in self.active_connections:
            log_data = {
                "type": "log",
                "task_id": task_id,
                "level": level,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            dead_connections = set()
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(log_data)
                except Exception:
                    dead_connections.add(connection)
            # 清理断开的连接
            for conn in dead_connections:
                self.active_connections[task_id].discard(conn)
    
    async def send_progress(self, task_id: str, progress: float, epoch: int, total_epochs: int):
        """发送进度更新"""
        if task_id in self.active_connections:
            progress_data = {
                "type": "progress",
                "task_id": task_id,
                "progress": progress,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "timestamp": datetime.utcnow().isoformat()
            }
            dead_connections = set()
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(progress_data)
                except Exception:
                    dead_connections.add(connection)
            for conn in dead_connections:
                self.active_connections[task_id].discard(conn)

# 全局 WebSocket 管理器
ws_manager = ConnectionManager()

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

app = FastAPI(title="Agent 微调平台", lifespan=lifespan)

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
    epochs: int = Field(default=3, ge=1, le=100, description="训练轮数")
    learning_rate: float = Field(default=2e-5, gt=0, description="学习率")
    batch_size: int = Field(default=8, ge=1, le=64, description="批次大小")  # 减小默认值避免GPU显存不足
    max_length: int = Field(default=128, ge=32, le=512, description="最大序列长度")  # 减小默认值避免GPU显存不足
    text_column: str = "text"
    label_column: str = "target"
    use_gpu: bool = True  # 是否使用GPU加速
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32, description="梯度累积步数")  # 梯度累积步数，等效于更大的batch_size

class AgentConfig(BaseModel):
    name: str
    role: str
    system_prompt: str
    model: str
    config: Optional[Dict] = None


@app.get("/")
async def root():
    return {"message": "Agent Finetune Platform API is running"}


# --- GPU 状态接口 ---

@app.get("/api/gpu/status")
async def get_gpu_status():
    """获取 GPU 状态信息"""
    import torch
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "pytorch_version": torch.__version__
    }
    
    if torch.cuda.is_available():
        gpu_info["cuda_version"] = torch.version.cuda
        gpu_info["device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            gpu_info["devices"].append({
                "index": i,
                "name": device_props.name,
                "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
                "major": device_props.major,
                "minor": device_props.minor
            })
    
    return gpu_info


# --- Ollama 代理接口 ---

@app.get("/api/models")
async def list_models():
    """获取 Ollama 中的本地模型"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            return resp.json()
        except Exception as e:
            # Ollama 不可用时返回空列表，而不是报错
            print(f"⚠️ 无法连接 Ollama ({OLLAMA_BASE_URL}): {str(e)}")
            return {"models": [], "error": f"Ollama 服务未运行，请启动 Ollama: ollama serve"}


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


@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: str, agent: AgentConfig, db: Session = Depends(get_db)):
    """更新 Agent"""
    updated_agent = crud.update_agent(
        db=db,
        agent_id=agent_id,
        name=agent.name,
        role=agent.role,
        system_prompt=agent.system_prompt,
        model=agent.model,
        config=agent.config
    )
    if not updated_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "success", "agent": updated_agent.to_dict()}


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
    import torch
    # 创建新的数据库会话（因为在线程中）
    from database import SessionLocal
    db = SessionLocal()
    
    # 初始化取消标志
    task_cancel_flags[task_id] = False
    
    # 根据配置和硬件情况决定使用的设备
    if req.use_gpu and torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 使用 GPU 训练: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        if req.use_gpu and not torch.cuda.is_available():
            logger.warning("⚠️ 请求使用 GPU 但 CUDA 不可用，回退到 CPU 训练")
        else:
            logger.info("📌 使用 CPU 训练")
    
    # 进度回调函数
    def progress_callback(current_epoch: int, total_epochs: int, progress: float):
        """更新训练进度到数据库"""
        # 检查取消标志
        if task_cancel_flags.get(task_id, False):
            raise InterruptedError("任务已被用户取消")
        
        try:
            crud.update_finetune_task_status(
                db=db,
                task_id=task_id,
                status=TaskStatus.RUNNING.value,
                progress=progress
            )
            logger.info(f"Task {task_id}: Epoch {current_epoch}/{total_epochs}, Progress: {progress:.1f}%")
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    try:
        # 更新任务状态为运行中
        crud.update_finetune_task_status(db, task_id, TaskStatus.RUNNING.value, progress=0.0)
        
        logger.info(f"Starting finetune task {task_id} for model {req.new_model_name}...")
        
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
        
        # 训练模型（带进度回调、设备配置和显存优化）
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            progress_callback=progress_callback,
            device=device,  # 使用配置的设备
            gradient_accumulation_steps=req.gradient_accumulation_steps,  # 梯度累积
            use_amp=(device == "cuda")  # GPU时启用混合精度训练
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
        
        logger.info(f"Finetune task {task_id} completed. Model saved to {model_path}")
        
    except InterruptedError as e:
        # 任务被取消
        crud.update_finetune_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.CANCELLED.value,
            error_message=str(e)
        )
        logger.warning(f"Finetune task {task_id} cancelled: {str(e)}")
    except Exception as e:
        # 更新任务状态为失败
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        crud.update_finetune_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.FAILED.value,
            error_message=error_detail
        )
        logger.error(f"Finetune task {task_id} failed: {str(e)}")
    finally:
        # 清理取消标志
        if task_id in task_cancel_flags:
            del task_cancel_flags[task_id]
        if task_id in running_tasks:
            del running_tasks[task_id]
        db.close()


async def run_finetune_task(task_id: str, req: FinetuneRequest):
    """异步运行微调任务"""
    # 在线程中运行同步任务
    thread = threading.Thread(target=run_finetune_task_sync, args=(task_id, req))
    thread.start()
    # 记录运行中的任务
    running_tasks[task_id] = thread


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
        gradient_accumulation_steps=req.gradient_accumulation_steps,
        text_column=req.text_column,
        label_column=req.label_column,
        use_gpu=req.use_gpu
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


@app.post("/api/finetune/{task_id}/cancel")
async def cancel_finetune_task(task_id: str, db: Session = Depends(get_db)):
    """取消正在运行的微调任务"""
    task = crud.get_finetune_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status not in [TaskStatus.RUNNING.value, TaskStatus.PENDING.value]:
        raise HTTPException(status_code=400, detail=f"无法取消状态为 {task.status} 的任务")
    
    # 设置取消标志
    if task_id in task_cancel_flags:
        task_cancel_flags[task_id] = True
        logger.info(f"任务 {task_id} 已标记为取消")
        return {"status": "cancelling", "message": "任务正在取消中，请稍候..."}
    else:
        # 任务可能还未开始运行，直接更新状态
        crud.update_finetune_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.CANCELLED.value,
            error_message="任务在启动前被取消"
        )
        return {"status": "cancelled", "message": "任务已取消"}


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


# --- 文件上传接口 ---

# 创建上传目录
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


@app.post("/api/upload/dataset")
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


@app.get("/api/datasets")
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


@app.delete("/api/datasets/{filename}")
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


# --- WebSocket 训练日志接口 ---

@app.websocket("/ws/finetune/{task_id}")
async def websocket_finetune_logs(websocket: WebSocket, task_id: str):
    """WebSocket 接口：实时推送训练日志"""
    await ws_manager.connect(websocket, task_id)
    
    try:
        # 发送连接成功消息
        await websocket.send_json({
            "type": "connected",
            "task_id": task_id,
            "message": "已连接到训练日志流"
        })
        
        # 保持连接，接收心跳或控制消息
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # 处理心跳
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # 发送心跳检测
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket 断开: task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
    finally:
        ws_manager.disconnect(websocket, task_id)
