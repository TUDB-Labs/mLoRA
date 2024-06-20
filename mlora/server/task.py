from mlora.config import DatasetConfig, TASKCONFIG_CLASS, ADAPTERCONFIG_CLASS

import os
import json
import logging
from fastapi import APIRouter, Request

from .storage import root_dir_list, db_get, db_put, db_it
from .pipe import g_s_create_task

router = APIRouter()


def complete_adapter_path(adapter):
    adapter["path"] = os.path.join(
        root_dir_list()["adapter"], "./" + adapter["path"])
    return adapter


def add_adapter(adapters, name):
    adapter = db_get(f'__adapter__{name}')
    if adapter is None:
        return {"message": "can not found the adapter"}
    adapter = json.loads(adapter)

    # complete the storage path
    adapter = complete_adapter_path(adapter)
    adapters[adapter["name"]] = ADAPTERCONFIG_CLASS[adapter["type"]](adapter)

    return adapters


@router.get("/task")
def get_task():
    ret = []
    for _, value in db_it("__task__"):
        ret.append(value)
    return ret


@router.post("/task")
async def post_task(request: Request):
    req = await request.json()

    if db_get(f'__task__{req["name"]}') is not None:
        return {"message": "already exist"}

    datasets = {}
    adapters = {}

    dataset = db_get(f'__dataset__{req["dataset"]}')
    if dataset is None:
        return {"message": "can not found the dataset"}
    dataset = json.loads(dataset)

    # complete the storage path
    dataset["data"] = os.path.join(
        root_dir_list()["train"], dataset["train_path"])
    dataset["prompt"] = os.path.join(
        root_dir_list()["prompt"], dataset["prompt_path"])
    datasets[dataset["name"]] = DatasetConfig(dataset)

    adapters = add_adapter(adapters, req["adapter"])

    # dpo need add the reference adapter
    if "reference" in req and req["reference"] != "base":
        adapters = add_adapter(adapters, req["reference"])

    task_conf = TASKCONFIG_CLASS[req["type"]](req, adapters, datasets)
    logging.info(f"Create new task: {req["name"]}")

    req["state"] = "UNK"
    db_put(f'__task__{req["name"]}', json.dumps(req))

    g_s_create_task.send(task_conf)

    return {"message": "success"}
