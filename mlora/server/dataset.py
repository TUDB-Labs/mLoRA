import json
import logging
from fastapi import APIRouter, Request

from .storage import db_it, db_get, db_put

router = APIRouter()


@router.get("/dataset")
def get_dataset():
    ret = []
    for _, value in db_it("__dataset__"):
        ret.append(value)
    return ret


@router.post("/dataset")
async def post_dataset(request: Request):
    req = await request.json()

    train_file_name = db_get(f'__train__{req["train"]}')
    prompt_file_name = db_get(f'__prompt__{req["prompt"]}')
    if train_file_name is None or prompt_file_name is None:
        return {"message": "error parameters"}

    if db_get(f'__dataset__{req["name"]}') is not None:
        return {"message": "already exist"}

    dataset = {
        "name": req["name"],
        "train": req["train"],
        "train_path": train_file_name,
        "prompt": req["prompt"],
        "prompt_path": prompt_file_name,
        "preprocess": req["preprocess"]
    }

    logging.info(f'Create new dataset: {req["name"]}')

    db_put(f'__dataset__{req["name"]}', json.dumps(dataset))

    return {"message": "success"}
