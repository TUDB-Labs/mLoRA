import plyvel
from typing import Dict

# define the root_dir
__g_db: plyvel.DB = None
__g_root_dir: str = ""
__g_root_dir_list = {"train": "./trains",
                     "prompt": "./prompts",
                     "adapter": "./adapters",
                     "db": "./db"}


def db() -> plyvel.DB:
    global __g_db
    return __g_db


def db_get(key: str) -> str:
    value = db().get(key.encode())
    if value is not None:
        value = value.decode()
        assert isinstance(value, str)
    return value


def db_put(key: str, value: str):
    assert isinstance(key, str)
    assert isinstance(value, str)
    db().put(key.encode(), value.encode())


def db_it(prefix: str):
    for key, value in db().iterator(prefix=prefix.encode()):
        yield key.decode(), value.decode()


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
