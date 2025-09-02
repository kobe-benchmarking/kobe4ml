from fastapi import FastAPI
import uvicorn
import importlib
import subprocess
import sys

from . import utils

app = FastAPI(title="Dynamic Executor API")

def install_and_import(package: str, index_url: str = None):
    """
    Ensure a package is installed and importable.
    - If not installed, installs from PyPI (or custom index).
    - Returns the imported module.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]

    if index_url:
        cmd += ["--extra-index-url", index_url]

    subprocess.check_call(cmd)
    importlib.import_module(package)

@app.get("/run")
def execute(package: str, method: str, data_id: str, model: str, options: str):
    """
    Dynamically import a package and run its 'main' function with parameters loaded from pickle files.

    :param package: Name of the package to import.
    :param method: Name of the method to execute.
    :param data_id: Pickle file containing the dataset.
    :param model: Pickle file containing the model parameters.
    :param options: Pickle file containing process parameters and metrics.
    :return: Result of the function execution.
    """
    args = {}
    arg_files = {"data_id": data_id, "model": model, "options": options}

    install_and_import(package='torch_loader', index_url='https://kobe-benchmarking.github.io/kobe4ml/')

    for arg_name, file_name in arg_files.items():
        args[arg_name] = utils.load_pickle(file_name)

    full_module_name = f"{package}.{method}"
    module = importlib.import_module(full_module_name)
    function= getattr(module, "main")
    
    results = function(**args)

    return {"status": "success", "results": results}

def setup():
    """
    Entry point to run the FastAPI server.
    """
    uvicorn.run("predictor.api:app", host="0.0.0.0", port=48035, reload=True)