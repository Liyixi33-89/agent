"""
GPU 状态路由
"""
import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/gpu", tags=["GPU 状态"])


@router.get("/status")
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
