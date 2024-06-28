from .backend_dispatcher import BackendDispatcher
from .dispatcher import Dispatcher

DISPATCHER_CLASS = {"default": Dispatcher, "backend": BackendDispatcher}

__all__ = ["Dispatcher", "BackendDispatcher", "DISPATCHER_CLASS"]
