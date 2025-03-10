from .operator import run

def __call__(config_path):
    return run(config_path)

__all__ = ["run"]
