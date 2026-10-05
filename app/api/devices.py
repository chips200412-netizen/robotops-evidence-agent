"""设备接口（阶段 5 切片 5，A5-4 前端对接）。

- GET    /devices：设备列表（型号/名称/参数/创建时间）
- POST   /devices：新增/编辑设备档案（按型号幂等 upsert，A4-2 设备 CRUD）
- DELETE /devices/{model}：删除设备档案

数据库不可用（MySQL 未启动）返回 503 并给明确提示（A4-4 精神）；
model 为空由 Pydantic 校验拒绝（422），设备不存在删除返回 404。
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.database import repositories
from app.database.connection import DatabaseError

router = APIRouter(tags=["devices"])


class DeviceUpsertRequest(BaseModel):
    """设备档案写入请求（model 必填，name/params 可选，按型号幂等）。"""

    model: str = Field(..., min_length=1, max_length=64, description="设备型号（规范名，唯一）")
    name: str = Field(default="", max_length=128, description="设备名称")
    params: str = Field(default="", description="设备参数")


@router.get("/devices")
def list_devices(
    limit: int = Query(default=200, ge=1, le=500, description="返回条数上限"),
) -> dict:
    """设备列表（最新在前）；count 为本页返回条数。"""
    try:
        devices = repositories.list_devices(limit=limit)
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"devices": devices, "count": len(devices)}


@router.post("/devices")
def upsert_device(req: DeviceUpsertRequest) -> dict:
    """新增/编辑设备档案（按型号幂等 upsert）；连接失败 → 503。"""
    try:
        return repositories.upsert_device(
            model=req.model.strip(),
            name=req.name.strip(),
            params=req.params.strip(),
        )
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.delete("/devices/{model}")
def delete_device(model: str) -> dict:
    """删除设备档案；不存在 → 404，连接失败 → 503。"""
    try:
        deleted = repositories.delete_device(model)
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"设备不存在：{model}")
    return {"model": model, "deleted": True}
