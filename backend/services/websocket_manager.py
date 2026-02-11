"""
WebSocket 连接管理器
用于实时推送训练日志等信息
"""
import logging
from typing import Dict, Set
from datetime import datetime
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        # task_id -> set of websocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        """接受 WebSocket 连接"""
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = set()
        self.active_connections[task_id].add(websocket)
        logger.info(f"WebSocket connected for task {task_id}")
    
    def disconnect(self, websocket: WebSocket, task_id: str):
        """断开 WebSocket 连接"""
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


# 全局 WebSocket 管理器实例
ws_manager = ConnectionManager()
