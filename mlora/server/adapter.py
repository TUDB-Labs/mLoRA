import logging
import uuid
import os

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .storage import db_del, db_get_obj, db_get_str, db_it_str, db_put_obj, root_dir_list

class DeleteDataRequest(BaseModel):
    name: str

router = APIRouter()


@router.get("/adapter")
def get_adapter():
    ret = []
    for _, value in db_it_str("__adapter__"):
        ret.append(value)
    return ret


@router.get("/adapter/download/{adapter_name}")
def download_adapter(adapter_name: str):
    adapter = db_get_obj(f'__adapter__{adapter_name}')
    if not adapter or adapter['state'] != 'DONE':
        return {"message": "Adapter is not ready for download."}

    file_path = os.path.join(root_dir_list()['adapter'], adapter['path'], 'adapter_model.bin')
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream')
    return {"message": "File not found."}


@router.post("/adapter")
async def post_adapter(request: Request):
    req = await request.json()

    if db_get_str(f'__adapter__{req["name"]}') is not None:
        return {"message": "already exist"}

    adapter_dir = str(uuid.uuid4()).replace("-", "")[:7]

    req["path"] = adapter_dir
    req["state"] = "UNK"
    req["task"] = "NO"

    logging.info(f"Create new adapter: {req}")

    db_put_obj(f'__adapter__{req["name"]}', req)

    return {"message": "success"}


@router.delete("/adapter")
def delete_adapter(request:DeleteDataRequest):
    adapter = db_get_obj(f"__adapter__{request.name}")

    if adapter is None:
        return {"message": "the adapter not exist"}

    # only adapter no task will train it can be delete
    if adapter["task"] != "NO" and adapter["state"] != "DONE":
        return {"message": "adapter with a task, cannot be delete."}

    db_del(f"__adapter__{request.name}")

    return {"message": "delete the adapter"}
