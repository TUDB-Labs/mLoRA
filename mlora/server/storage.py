import json
from typing import Dict

import plyvel

# define the root_dir
__g_db: plyvel.DB = None
__g_root_dir: str = ""
__g_root_dir_list = {
    "data": "./datas",
    "prompt": "./prompts",
    "adapter": "./adapters",
    "db": "./db",
}


def db() -> plyvel.DB:
    global __g_db
    return __g_db


def db_get_str(key: str) -> str:
    value = db().get(key.encode())
    if value is not None:
        value = value.decode()
        assert isinstance(value, str)
    return value


def db_put_str(key: str, value: str):
    assert isinstance(key, str)
    assert isinstance(value, str)
    db().put(key.encode(), value.encode())


def db_get_obj(key: str) -> Dict[str, str]:
    ret_str = db_get_str(key)
    if ret_str is None:
        return None
    return json.loads(ret_str)


def db_put_obj(key: str, value: Dict[str, str]):
    assert isinstance(key, str)
    assert isinstance(value, Dict)
    db_put_str(key, json.dumps(value))


def db_del(key: str):
    db().delete(key.encode())


def db_it_str(prefix: str):
    for key, value in db().iterator(prefix=prefix.encode()):
        yield key.decode(), value.decode()


def db_it_obj(prefix: str):
    for key, value in db().iterator(prefix=prefix.encode()):
        yield key.decode(), json.loads(value.decode())


def root_dir() -> str:
    global __g_root_dir
    return __g_root_dir


def root_dir_list() -> Dict[str, str]:
    global __g_root_dir_list
    return __g_root_dir_list


def set_db(db: plyvel):
    global __g_db
    __g_db = db


def set_root_dir(dir: str):
    global __g_root_dir
    __g_root_dir = dir


def set_root_dir_list(dir_list: Dict[str, str]):
    global __g_root_dir_list
    __g_root_dir_list = dir_list
