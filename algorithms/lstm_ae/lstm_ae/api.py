from fastapi import FastAPI
import pickle
import importlib
import uvicorn

app = FastAPI(title="Dynamic Executor API")

@app.get("/run")
def execute(package: str, method: str, params: str):
    """
    Dynamically import a package and run a function with parameters from a pickle file.

    :param package: Name of the package to import.
    :param method): Name of the method) to execute.
    :param params: Path to the pickle file containing function parameters.
    :return: Result of the function execution.
    """
    with open(params, "rb") as f:
        params = pickle.load(f)

    full_module_name = f"{package}.{method}"
    module = importlib.import_module(full_module_name)

    function= getattr(module, "main")
    result = function(**params)

    return {"status": "success", "result": result}

def setup():
    """
    Entry point to run the FastAPI server.
    """
    uvicorn.run("lstm_ae.api:app", host="0.0.0.0", port=48031, reload=True)