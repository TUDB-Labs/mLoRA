from mlora.config import DatasetConfig, TASKCONFIG_CLASS, ADAPTERCONFIG_CLASS

import os
import logging
from fastapi import APIRouter, Request

from .storage import root_dir_list, db_get_str, db_put_obj, db_it_str, db_get_obj
from .pipe import g_s_create_task

router = APIRouter()


def complete_path(obj, dir_type: str, file_name: str):
    obj[file_name] = os.path.join(
        root_dir_list()[dir_type], "./" + obj[file_name])
    return obj


@router.get("/task")
def get_task():
    ret = []
    for _, value in db_it_str("__task__"):
        ret.append(value)
    return ret


@router.post("/task")
async def post_task(request: Request):
    req = await request.json()

    if db_get_str(f'__task__{req["name"]}') is not None:
        return {"message": "already exist"}

    dataset = db_get_obj(f'__dataset__{req["dataset"]}')
    if dataset is None:
        return {"message": "can not found the dataset"}

    adapter = db_get_obj(f'__adapter__{req["adapter"]}')
    if adapter is None:
        return {"message": "can not found the adapter"}

    datasets = {}
    adapters = {}
    # complete the storage path
    dataset = complete_path(dataset, "data", "data")
    dataset = complete_path(dataset, "prompt", "prompt")
    datasets[dataset["name"]] = DatasetConfig(dataset)

    adapter = complete_path(adapter, "adapter", "path")
    adapters[adapter["name"]] = ADAPTERCONFIG_CLASS[adapter["type"]](adapter)

    # dpo need add the reference adapter
    if "reference" in req and req["reference"] != "base":
        adapter = db_get_obj(f'__adapter__{req["reference"]}')
        if adapter is None:
            return {"message": "can not found the reference adapter"}
        adapter = complete_path(adapter, "adapter", "path")
        adapters[adapter["name"]
                 ] = ADAPTERCONFIG_CLASS[adapter["type"]](adapter)

    task_conf = TASKCONFIG_CLASS[req["type"]](req, adapters, datasets)

    logging.info(f"Create new task: {req["name"]}")

    req["state"] = "UNK"
    db_put_obj(f'__task__{req["name"]}', req)

    g_s_create_task.send(task_conf)

    return {"message": "success"}
