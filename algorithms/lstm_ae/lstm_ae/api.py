from fastapi import FastAPI
import pickle
import uvicorn
import importlib

app = FastAPI(title="Dynamic Executor API")

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

    for arg_name, file_name in arg_files.items():
        with open(file_name, "rb") as f:
            args[arg_name] = pickle.load(f)

    full_module_name = f"{package}.{method}"
    module = importlib.import_module(full_module_name)
    function= getattr(module, "main")
    
    results = function(**args)

    return {"status": "success", "results": results}

def setup():
    """
    Entry point to run the FastAPI server.
    """
    uvicorn.run("lstm_ae.api:app", host="0.0.0.0", port=48031, reload=True)