"""
WebSocket 路由
实时推送训练日志等
"""
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/finetune/{task_id}")
async def websocket_finetune_logs(websocket: WebSocket, task_id: str):
    """WebSocket 接口：实时推送训练日志"""
    logger.info(f"WebSocket 连接请求: task_id={task_id}")
    try:
        await ws_manager.connect(websocket, task_id)
    except Exception as e:
        logger.error(f"WebSocket 连接失败: {e}")
        return
    
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
