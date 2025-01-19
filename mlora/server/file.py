import logging
import os
import uuid

from fastapi import APIRouter, Form, UploadFile, File
from pydantic import BaseModel

from .storage import db_del, db_get_str, db_it_str, db_put_str, root_dir_list

class DeleteDataRequest(BaseModel):
    name: str

router = APIRouter()


def get_local_file(file_type: str):
    ret = []
    for key, value in db_it_str(file_type):
        ret.append({"name": key[len(file_type) :], "file": value})

    return ret


def save_local_file(file_type: str, name: str, data_file: UploadFile):
    if data_file.filename is None:
        return {"message": "error file name"}

    file_postfix = data_file.filename.split(".")[-1]
    check_postfix = ["json", "yaml"]
    if file_postfix not in check_postfix:
        return {"message": "unsupport file type"}

    if db_get_str(f"__{file_type}__{name}") is not None:
        return {"message": "file already exist"}

    file_uuid = str(uuid.uuid4()).replace("-", "")[:7]
    file_name = file_uuid + "." + file_postfix

    logging.info(f"Recv and save data file: {file_name}")

    file_path = os.path.join(root_dir_list()[file_type], file_name)
    with open(file_path, "wb+") as file_object:
        file_object.write(data_file.file.read())

    return file_name


@router.get("/data")
def get_data():
    return get_local_file("__data__")


@router.post("/data")
def post_data(name: str = Form(...), data_file: UploadFile = File(...)):
    file_name = save_local_file("data", name, data_file)
    db_put_str(f"__data__{name}", file_name)
    return {"message": "success"}


@router.delete("/data")
def delete_data(request: DeleteDataRequest):
    file_name = db_get_str(f"__data__{request.name}")

    if file_name is None:
        return {"message": "file not exist"}

    db_del(f"__data__{request.name}")

    return {"message": "delete success"}


@router.get("/prompt")
def get_prompt():
    return get_local_file("__prompt__")


@router.post("/prompt")
def post_prompt(name: str = Form(...), data_file: UploadFile = File(...)):
    file_name = save_local_file("prompt", name, data_file)

    db_put_str(f"__prompt__{name}", file_name)

    return {"message": "success"}


@router.delete("/prompt")
def delete_prompt(request: DeleteDataRequest):
    file_name = db_get_str(f"__prompt__{request.name}")

    if file_name is None:
        return {"message": "file not exist"}

    db_del(f"__prompt__{request.name}")

    return {"message": "delete success"}
