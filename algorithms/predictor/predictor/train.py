import torch
import time
import zipfile
import os

from . import utils
from .model import *

logger = utils.get_logger(level='CRITICAL')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

def zip_model(model, save_url, model_params):
    """
    Package the trained model weights and parameters into a zip file for inference.

    :param model: Trained PyTorch model.
    :param save_url: Path to the pth file where the model weights are saved, e.g., models/classifier.pth.
    :param model_params: Dictionary of model configuration parameters.
    """
    zip_path = save_url.replace('.pth', '.zip')
    json_url = save_url.replace('.pth', '_params.json')

    utils.save_pth(model=model, path=save_url)
    utils.save_json(data=model_params, path=json_url)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(save_url, os.path.basename(save_url))
        zipf.write(json_url, os.path.basename(json_url))

    os.remove(save_url)
    os.remove(json_url)

    logger.info(f"Packaged {save_url} and {json_url} into {zip_path}.")

def train(data, model, save_url, model_params, process_params, metrics):
    """
    Train the model on the provided data and calculate the training metrics.

    :param data: Tuple containing (train_data, val_data), where each is a DataLoader.
    :param model: The model to be trained.
    :param save_url: Path to save the trained model.
    :param model_params: Dictionary containing model configuration parameters.
    :param process_params: Dictionary containing process parameters.
    :param metrics: List of metric names to calculate (e.g., train_loss).
    :return: Dictionary containing metrics as defined in the input metrics list.
    """
    loss, epochs, patience, lr, optimizer, scheduler = process_params.values()

    if hasattr(utils, loss):
        criterion = getattr(utils, loss)()
    else:
        raise ValueError(f"Loss function '{loss}' not found in utils")

    model.to(device)

    train_data, val_data = data
    batches = len(train_data)
    
    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(optimizer, scheduler['name'], **scheduler['params'])

    train_time = 0.0
    best_val_loss = float('inf')
    stationary = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        start = time.time()
        total_train_loss = 0.0

        model.train()

        for _, (X, _, y) in enumerate(train_data):
            X, y = X.to(device), y.to(device)

            y_pred, _ = model(X)

            batch_size, seq_len, num_classes = y_pred.size()
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y = y.reshape(batch_size * seq_len)

            train_loss = criterion(y_pred, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

        avg_train_loss = total_train_loss / batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for _, (X, _, y) in enumerate(val_data):
                X, y = X.to(device), y.to(device)

                y_pred, _ = model(X)

                batch_size, seq_len, num_classes = y_pred.size()
                y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
                y = y.reshape(batch_size * seq_len)

                val_loss = criterion(y_pred, y)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        end = time.time()
        duration = end - start
        train_time += duration

        if avg_val_loss < best_val_loss:
            stationary = 0
            
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss

            zip_model(model, save_url, model_params)
        else:
            stationary += 1

        if stationary >= patience:
            break

        scheduler.step(avg_val_loss)

    all_metrics = {
        'epochs': epoch + 1,
        'train_time': train_time,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(params):
    """
    Main function to execute the training workflow, including data preparation and model evaluation.
    """
    model_params, dls, metrics, process_params, save_url = params.values()

    model = Classifier(**model_params)
 
    results = train(data=dls,
                    model=model,
                    save_url=save_url,
                    model_params=model_params,
                    process_params=process_params,
                    metrics=metrics)
    
    return results