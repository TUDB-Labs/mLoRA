import logging
import os

from fastapi import APIRouter, Request

from mlora.config import ADAPTERCONFIG_CLASS, TASKCONFIG_CLASS, DatasetConfig

from .pipe import g_s_create_task, g_s_notify_terminate_task
from .storage import (
    db_del,
    db_get_obj,
    db_get_str,
    db_it_str,
    db_put_obj,
    root_dir_list,
)

router = APIRouter()


def complete_path(obj, dir_type: str, file_name: str):
    obj[file_name] = os.path.join(root_dir_list()[dir_type], "./" + obj[file_name])
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

    # create the task config for executor
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
        ref_adapter = db_get_obj(f'__adapter__{req["reference"]}')
        if ref_adapter is None:
            return {"message": "can not found the reference adapter"}
        ref_adapter = complete_path(ref_adapter, "adapter", "path")
        adapters[ref_adapter["name"]] = ADAPTERCONFIG_CLASS[ref_adapter["type"]](
            ref_adapter
        )

    task_conf = TASKCONFIG_CLASS[req["type"]](req, adapters, datasets)

    logging.info(f"Create new task: {req["name"]} with adapter")

    # set the task's state
    req["state"] = "UNK"
    db_put_obj(f'__task__{req["name"]}', req)

    # set the adapter's state
    adapter = db_get_obj(f'__adapter__{req["adapter"]}')
    adapter["task"] = req["name"]
    db_put_obj(f'__adapter__{req["adapter"]}', adapter)

    g_s_create_task.send(task_conf)

    return {"message": "success"}


@router.delete("/task")
def terminate_task(name: str):
    task = db_get_obj(f"__task__{name}")

    if task is None:
        return {"message": "the task not exist"}

    if task["state"] == "DONE":
        db_del(f"__task__{name}")
        return {"message": "delete the done task."}

    g_s_notify_terminate_task.send(name)

    task["state"] = "TERMINATING"
    db_put_obj(f"__task__{name}", task)

    return {"message": f"to terminate the task {name}, wait."}
