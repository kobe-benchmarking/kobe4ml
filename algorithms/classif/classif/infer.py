import torch
import zipfile
import os
from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1

from . import utils
from .model import *

logger = utils.get_logger(level='CRITICAL')

device = utils.detect_device()
logger.info(f'Device is {device}')

def precision(y, y_pred):
    """
    Compute precision using scikit-learn.
    y: ground-truth tensor (shape: N,)
    y_pred: logits or probabilities (shape: N x num_classes)
    """
    y_true = y.detach().cpu().numpy()
    y_hat = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

    return precision_score(y_true, y_hat, average="macro", zero_division=0)

def recall(y, y_pred):
    """
    Compute recall using scikit-learn.
    """
    y_true = y.detach().cpu().numpy()
    y_hat = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

    return recall_score(y_true, y_hat, average="macro", zero_division=0)

def f1_score(y, y_pred):
    """
    Compute F1 score using scikit-learn.
    """
    y_true = y.detach().cpu().numpy()
    y_hat = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

    return sk_f1(y_true, y_hat, average="macro", zero_division=0)

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
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    batches = len(data)

    with torch.no_grad():
        for _, (X, _, y) in enumerate(data):
            X, y = X.to(device), y.to(device)

            y_pred, _ = model(X)

            batch_size, seq_len, num_classes = y_pred.size()
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y = y.reshape(batch_size * seq_len)

            infer_loss = criterion(y_pred, y)

            total_infer_loss += infer_loss.item()
            total_precision += precision(y, y_pred)
            total_recall += recall(y, y_pred)
            total_f1 += f1_score(y, y_pred)

    avg_infer_loss = total_infer_loss / batches
    avg_precision = total_precision / batches
    avg_recall = total_recall / batches
    avg_f1 = total_f1 / batches

    all_metrics = {
        'infer_loss': avg_infer_loss,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1
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
    data, weights = data_id.values()
    model_params, model_pth = unzip_model(path=model)
    process_params, metrics = options.values()
    batch_size, loss = process_params.values()

    model = Classifier(**model_params)
 
    if hasattr(utils, loss):
        criterion = getattr(utils, loss)(weights)
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
