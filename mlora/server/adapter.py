import json
import uuid
import logging
from fastapi import APIRouter, Request

from .storage import db_it, db_get, db_put

router = APIRouter()


@router.get("/adapter")
def get_adapter():
    ret = []
    for _, value in db_it("__adapter__"):
        ret.append(value)
    return ret


@router.post("/adapter")
async def post_adapter(request: Request):
    req = await request.json()

    if db_get(f'__adapter__{req["name"]}') is not None:
        return {"message": "already exist"}

    adapter_dir = str(uuid.uuid4()).replace("-", "")[:7]

    req["path"] = adapter_dir
    req["state"] = "UNK"

    logging.info(f"Create new adapter: {req}")

    db_put(f'__adapter__{req["name"]}', json.dumps(req))

    return {"message": "success"}
