import torch
import zipfile
import os

from . import utils
from .model import *

logger = utils.get_logger(level='CRITICAL')

device = utils.detect_device()
logger.info(f'Device is {device}')

def mae(X, X_dec):
    """
    Compute Mean Absolute Error (MAE) manually.

    :param X: Original input tensor.
    :param X_dec: Reconstructed output tensor.
    :return: MAE value.
    """
    return torch.mean(torch.abs(X - X_dec)).item()

def mse(X, X_dec):
    """
    Compute Mean Squared Error (MSE) manually.

    :param X: Original input tensor.
    :param X_dec: Reconstructed output tensor.
    :return: MSE value.
    """
    return torch.mean((X - X_dec) ** 2).item()

def unzip_model(path):
    """
    Extract a model zip file into the model weights and parameters for inference.

    :param path: Path to the zip file created by zip_model, e.g., models/attn_ae.zip
    :return: Tuple (model_params_dict, model_pth_path)
    """
    extract_dir = os.path.dirname(path)

    with zipfile.ZipFile(path, 'r') as zipf:
        zipf.extractall(extract_dir)

    pth_path = path.replace('.zip', '.pth')
    json_path = path.replace('.zip', '_params.json')

    model_params = utils.load_json(path=json_path)

    return model_params, pth_path

def infer(data, model, model_pth, criterion, metrics):
    """
    Test the model on the provided data and calculate the test loss, MAE, and MSE.

    :param data: DataLoader for inference.
    :param model: The model to be tested.
    :param model_pth: Path to the pth file where the model weights are saved, e.g., models/attn_ae.pth.
    :param criterion: Loss function to be used during inference.
    :param metrics: List of metric names to calculate (e.g., ['mae', 'mse']).
    :return: Dictionary containing metrics as defined in the input metrics list.
    """
    state_dict = utils.load_pth(path=model_pth)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_infer_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0

    batches = len(data)

    with torch.no_grad():
        for _, (X, Xn, _) in enumerate(data):
            X, Xn = X.to(device), Xn.to(device)

            X_dec, _ = model(X)

            X_dec, _ = utils.separate(src=X_dec, c=[0,1], t=[2])
            Xn, _ = utils.separate(src=Xn, c=[0,1], t=[2])

            infer_loss = criterion(X_dec, Xn)
            total_infer_loss += infer_loss.item()

            total_mae += mae(Xn, X_dec)
            total_mse += mse(Xn, X_dec)

    avg_infer_loss = total_infer_loss / batches
    avg_mae = total_mae / batches
    avg_mse = total_mse / batches

    all_metrics = {
        'infer_loss': avg_infer_loss,
        'mae': avg_mae,
        'mse': avg_mse
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(data_id, model, options):
    """
    Main function to execute the testing workflow.

    :param data_id: Dictionary containing dataset and its parameters.
    :param model: Path to the model zip file containing model weights and parameters.
    :param options: Dictionary containing process parameters and metrics to calculate.
    :return: Dictionary containing calculated metrics.
    """
    data, _ = data_id.values()
    model_params, model_pth = unzip_model(path=model)
    process_params, metrics = options.values()
    batch_size, loss = process_params.values()

    model = Predictor(**model_params)
 
    if hasattr(utils, loss):
        criterion = getattr(utils, loss)()
    else:
        raise ValueError(f"Loss function '{loss}' not found in utils")

    dl = utils.create_dataloader(ds=data[0],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=False)
 
    results = infer(data=dl,
                    model=model,
                    model_pth=model_pth,
                    criterion=criterion,
                    metrics=metrics)
    
    return results
