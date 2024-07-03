import logging
import os

from datasets import load_dataset
from fastapi import APIRouter, Request

from mlora.config import DatasetConfig
from mlora.prompter import PrompterFactory

from .storage import (
    db_del,
    db_get_obj,
    db_get_str,
    db_it_str,
    db_put_obj,
    root_dir_list,
)

router = APIRouter()


@router.get("/dataset")
def get_dataset():
    ret = []
    for _, value in db_it_str("__dataset__"):
        ret.append(value)
    return ret


@router.get("/showcase")
def showcase_dataset(name: str):
    dataset = db_get_obj(f"__dataset__{name}")

    if dataset is None:
        return {"message": "the dataset not exist"}

    dataset_config = DatasetConfig(dataset)

    dataset_config.data_path_ = os.path.join(
        root_dir_list()["data"], dataset_config.data_path_
    )
    dataset_config.prompt_path_ = os.path.join(
        root_dir_list()["prompt"], dataset_config.prompt_path_
    )

    prompter = PrompterFactory.create(dataset_config)

    # just read one item
    data_points = load_dataset(
        "json", data_files=dataset_config.data_path_, split="train[:1]"
    )

    ret = prompter.generate_prompt(data_points)

    return {"example": ret}


@router.post("/dataset")
async def post_dataset(request: Request):
    req = await request.json()

    data_file = db_get_str(f'__data__{req["data_name"]}')
    prompt_file = db_get_str(f'__prompt__{req["prompt_name"]}')

    if data_file is None or prompt_file is None:
        return {"message": "error parameters"}

    if db_get_str(f'__dataset__{req["name"]}') is not None:
        return {"message": "dataset already exist"}

    dataset = {
        "name": req["name"],
        "data_name": req["data_name"],
        "prompt_name": req["prompt_name"],
        "data": data_file,
        "prompt": prompt_file,
        "prompt_type": req["prompt_type"],
        "preprocess": req["preprocess"],
    }

    logging.info(f'Create new dataset: {req["name"]}')

    db_put_obj(f'__dataset__{req["name"]}', dataset)

    return {"message": "success"}


@router.delete("/dataset")
def delete_dataset(name: str):
    dataset = db_get_obj(f"__dataset__{name}")

    if dataset is None:
        return {"message": "the dataset not exist"}

    db_del(f"__dataset__{name}")

    return {"message": "delete the dataset"}
