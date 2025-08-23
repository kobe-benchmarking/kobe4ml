import torch
import zipfile
import os
from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1

from . import utils
from .model import *

logger = utils.get_logger(level='CRITICAL')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

def precision(y, y_pred):
    """
    Compute precision using scikit-learn.
    y: ground-truth tensor (shape: N,)
    y_pred: logits or probabilities (shape: N x num_classes)
    """
    y_true = y.detach().cpu().numpy()
    y_hat = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

    return precision_score(y_true, y_hat, average="macro")

def recall(y, y_pred):
    """
    Compute recall using scikit-learn.
    """
    y_true = y.detach().cpu().numpy()
    y_hat = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

    return recall_score(y_true, y_hat, average="macro")

def f1_score(y, y_pred):
    """
    Compute F1 score using scikit-learn.
    """
    y_true = y.detach().cpu().numpy()
    y_hat = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

    return sk_f1(y_true, y_hat, average="macro")

def unzip_model(path):
    """
    Extract a model zip file into the model weights and parameters for inference.

    :param path: Path to the zip file created by zip_model, e.g., models/attn_ae.zip
    :return: Tuple (model_pth_path, model_params_dict)
    """
    extract_dir = os.path.dirname(path)

    with zipfile.ZipFile(path, 'r') as zipf:
        zipf.extractall(extract_dir)

    model_pth = path.replace('.zip', '.pth')
    json_path = path.replace('.zip', '_params.json')

    model_params = utils.load_json(path=json_path)

    return model_pth, model_params

def infer(dls, model, model_pth, metrics):
    """
    Test the model on the provided data and calculate the inference metrics.

    :param dls: Tuple containing (loaders, weights).
    :param model: The model to be evaluated.
    :param model_pth: Path to the pth file where the model weights are saved, e.g., models/classifier.pth.
    :param metrics: List of metric names to calculate (e.g., WeightedCrossEntropyLoss).
    :return: Dictionary containing metrics as defined in the input metrics list.
    """
    state_dict = utils.load_pth(path=model_pth)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    data, weights = dls
    data = data[0]
    batches = len(data)

    total_infer_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    criterion = utils.WeightedCrossEntropyLoss(weights)

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
        'WeightedCrossEntropyLoss': avg_infer_loss,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(params):
    """
    Main function to execute the testing workflow, including data preparation and model evaluation.
    """
    model_url, dls, metrics = params.values()

    model_pth, model_params = unzip_model(path=model_url)

    model = Classifier(**model_params)
 
    metrics = infer(dls=dls,
                    model=model,
                    model_pth=model_pth,
                    metrics=metrics)
    
    return metrics
