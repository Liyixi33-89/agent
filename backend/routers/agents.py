"""
Agent 管理路由
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from schemas.requests import AgentConfig
from database import get_db
import crud

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["Agent 管理"])


@router.post("")
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


@router.get("")
async def get_agents(db: Session = Depends(get_db)):
    """获取所有 Agent"""
    agents = crud.get_all_agents(db)
    return [agent.to_dict() for agent in agents]


@router.get("/{agent_id}")
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """获取指定 Agent"""
    agent = crud.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.to_dict()


@router.put("/{agent_id}")
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


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, db: Session = Depends(get_db)):
    """删除 Agent"""
    success = crud.delete_agent(db, agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted"}
