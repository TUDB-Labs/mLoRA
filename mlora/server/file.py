import os
import uuid
import logging
from fastapi import APIRouter, UploadFile

from .storage import root_dir_list, db_put, db_it

router = APIRouter()


def get_local_file(file_type: str):
    ret = []
    for key, value in db_it(file_type):
        ret.append({"name": key[len(file_type):], "file": value})

    return ret


def save_local_file(file_type: str, name: str, data_file: UploadFile):
    if data_file.filename.split(".")[-1] != "json":
        return {"message": "unsupport file type"}

    filename = str(uuid.uuid4()).replace("-", "")[:7] + ".json"

    logging.info(f"Recv and save data file: {filename}")

    file_path = os.path.join(root_dir_list()[file_type], filename)
    with open(file_path, "wb+") as file_object:
        file_object.write(data_file.file.read())

    db_put(f"__{file_type}__{name}", filename)

    return {"message": "success"}


@router.get("/train")
def get_data():
    return get_local_file("__train__")


@router.post("/train")
def post_data(name: str, data_file: UploadFile):
    return save_local_file("train", name, data_file)


@router.get("/prompt")
def get_prompt():
    return get_local_file("__prompt__")


@router.post("/prompt")
def post_prompt(name: str, data_file: UploadFile):
    return save_local_file("prompt", name, data_file)
