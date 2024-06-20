from fastapi import APIRouter

from .pipe import g_s_dispatcher

router = APIRouter()


@router.get("/dispatcher")
def get_dispatcher():
    g_s_dispatcher.send([])
    return g_s_dispatcher.recv()
