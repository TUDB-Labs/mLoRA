from typing import Dict, Type

from .backend_dispatcher import BackendDispatcher
from .dispatcher import Dispatcher
from .pipe_dispatcher import PipeDispatcher

DISPATCHER_CLASS: Dict[str, Type[Dispatcher]] = {
    "default": Dispatcher,
    "backend": BackendDispatcher,
    "pipe": PipeDispatcher,
}

__all__ = ["Dispatcher", "BackendDispatcher", "PipeDispatcher", "DISPATCHER_CLASS"]
