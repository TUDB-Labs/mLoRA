import uuid
import logging
from fastapi import APIRouter, Request

from .storage import db_it_str, db_get_str, db_put_obj

router = APIRouter()


@router.get("/adapter")
def get_adapter():
    ret = []
    for _, value in db_it_str("__adapter__"):
        ret.append(value)
    return ret


@router.post("/adapter")
async def post_adapter(request: Request):
    req = await request.json()

    if db_get_str(f'__adapter__{req["name"]}') is not None:
        return {"message": "already exist"}

    adapter_dir = str(uuid.uuid4()).replace("-", "")[:7]

    req["path"] = adapter_dir
    req["state"] = "UNK"

    logging.info(f"Create new adapter: {req}")

    db_put_obj(f'__adapter__{req["name"]}', req)

    return {"message": "success"}
