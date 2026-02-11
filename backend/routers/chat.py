"""
聊天相关路由
"""
import os
import logging
from fastapi import APIRouter, HTTPException, Depends
import httpx
from sqlalchemy.orm import Session

from schemas.requests import ChatRequest
from database import get_db
import crud

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["聊天"])

# 配置 Ollama 地址
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@router.get("/models")
async def list_models():
    """获取 Ollama 中的本地模型"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            return resp.json()
        except Exception as e:
            # Ollama 不可用时返回空列表，而不是报错
            logger.warning(f"⚠️ 无法连接 Ollama ({OLLAMA_BASE_URL}): {str(e)}")
            return {"models": [], "error": f"Ollama 服务未运行，请启动 Ollama: ollama serve"}


@router.post("/chat")
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


@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """获取指定会话的聊天历史"""
    messages = crud.get_chat_history(db, session_id)
    return [msg.to_dict() for msg in messages]


@router.get("/chat/sessions")
async def get_chat_sessions(db: Session = Depends(get_db)):
    """获取所有聊天会话"""
    sessions = crud.get_all_sessions(db)
    return {"sessions": sessions}


@router.delete("/chat/history/{session_id}")
async def delete_chat_history(session_id: str, db: Session = Depends(get_db)):
    """删除指定会话的聊天历史"""
    count = crud.delete_chat_history(db, session_id)
    return {"deleted": count}
