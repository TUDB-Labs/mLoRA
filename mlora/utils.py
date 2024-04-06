import importlib.metadata
import importlib.util
import logging

from typing import Tuple, Union


def setup_logging(log_level: str = "WARN", log_file: str = None):
    # set the logger
    log_handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(format="[%(asctime)s] m-LoRA: %(message)s",
                        level=log_level,
                        handlers=log_handlers,
                        force=True)


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logging.debug(f"Detected {pkg_name} version {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


class Unsubscribable:
    def __init__(self) -> None:
        raise RuntimeError(f"Instant unsubscribable class {__class__}")


# Class Placeholder for Bitsandbytes
class Linear8bitLt(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class Linear4bit(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class BitsAndBytesConfig:
    def __init__(self, **kwargs) -> None:
        raise RuntimeError("Quantization not supported.")


class NoneContexts(object):
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
