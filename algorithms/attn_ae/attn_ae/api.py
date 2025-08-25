from fastapi import FastAPI
import pickle
import importlib
import uvicorn

app = FastAPI(title="Dynamic Executor API")

@app.get("/run")
def execute(package, func, params):
    """
    Dynamically import a package and run a function with parameters from a pickle file.

    :param package: Name of the package to import.
    :param func: Name of the function to execute.
    :param params: Path to the pickle file containing function parameters.
    :return: Result of the function execution.
    """
    with open(params, "rb") as f:
        params = pickle.load(f)

    module = importlib.import_module(package)
    func = getattr(module, func)

    result = func(**params)

    return {"status": "success", "result": result}

def setup():
    """
    Entry point to run the FastAPI server.
    """
    uvicorn.run("attn_ae.api:app", host="0.0.0.0", port=48033, reload=True)

##############################################################################
# import requests

# response = requests.get(
#     "http://16.171.214.46:48033/run",
#     params={
#         "package": "attn_ae",
#         "func": "train",
#         "params": "models/attn_ae_train_params.pkl"
#     }
# )
# print(response.json())
