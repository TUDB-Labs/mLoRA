from .dispatcher import Dispatcher
from .backend_dispatcher import BackendDispatcher

DISPATCHER_CLASS = {
    "default": Dispatcher,
    "backend": BackendDispatcher
}

__all__ = [
    "Dispatcher",
    "BackendDispatcher",
    "DISPATCHER_CLASS"
]
