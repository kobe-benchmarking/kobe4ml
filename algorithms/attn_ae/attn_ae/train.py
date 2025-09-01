import torch
import time
import zipfile
import os

from . import utils
from .model import *

logger = utils.get_logger(level='CRITICAL')

device = utils.detect_device()
logger.info(f'Device is {device}')

def zip_model(model, model_pth, model_params):
    """
    Package the trained model weights and parameters into a zip file for inference.

    :param model: Trained PyTorch model.
    :param model_pth: Path to the pth file where the model weights are saved, e.g., models/attn_ae.pth.
    :param model_params: Dictionary of model configuration parameters.
    """
    zip_path = model_pth.replace('.pth', '.zip')
    json_path = model_pth.replace('.pth', '_params.json')

    utils.save_pth(model=model, path=model_pth)
    utils.save_json(data=model_params, path=json_path)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(model_pth, os.path.basename(model_pth))
        zipf.write(json_path, os.path.basename(json_path))

    os.remove(model_pth)
    os.remove(json_path)

    logger.info(f"Packaged {model_pth} and {json_path} into {zip_path}.")

def train(data, model, model_params, model_pth, criterion, epochs, patience, optimizer, scheduler, metrics):
    """
    Train the model on the provided data and calculate the train loss, MAE, and MSE.

    :param data: List containing training and validation DataLoaders.
    :param model: The model to be trained.
    :param model_params: Dictionary containing model configuration parameters.
    :param model_pth: Path to save the trained model.
    :param criterion: Loss function to be used during training.
    :param epochs: Maximum number of training epochs.
    :param patience: Number of epochs with no improvement after which training will be stopped.
    :param optimizer: Optimizer for training.
    :param scheduler: Learning rate scheduler.
    :param metrics: List of metric names to calculate (e.g., ['mae', 'mse']).
    :return: Dictionary containing metrics as defined in the input metrics list.
    """
    model.to(device)

    train_time = 0.0
    best_val_loss = float('inf')
    stationary = 0
    train_losses, val_losses = [], []

    train_data, val_data = data
    batches = len(train_data)

    for epoch in range(epochs):
        start = time.time()
        total_train_loss = 0.0

        model.train()

        for _, (X, _, _) in enumerate(train_data):
            X = X.to(device)

            X_dec, _, _ = model(X)
            
            X_dec, _ = utils.separate(src=X_dec, c=[0,1], t=[2])
            X, _ = utils.separate(src=X, c=[0,1], t=[2])

            train_loss = criterion(X_dec, X)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

        avg_train_loss = total_train_loss / batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for _, (X, _, _) in enumerate(val_data):
                X = X.to(device)

                X_dec, _, _ = model(X)

                X_dec, _ = utils.separate(src=X_dec, c=[0,1], t=[2])
                X, _ = utils.separate(src=X, c=[0,1], t=[2])

                val_loss = criterion(X_dec, X)
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
            
            zip_model(model, model_pth, model_params)
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

def main(data_id, model, options):
    """
    Main function to execute the training workflow.

    :param data_id: Dictionary containing dataset and its parameters.
    :param model: Dictionary containing model parameters and save URL.
    :param options: Dictionary containing process parameters and metrics to calculate.
    :return: Dictionary containing calculated metrics.
    """
    data, _ = data_id.values()
    model_params, model_pth = model.values()
    process_params, metrics = options.values()
    batch_size, loss, epochs, patience, lr, optimizer, scheduler = process_params.values()

    model = Attn_Autoencoder(**model_params)

    if hasattr(utils, loss):
        criterion = getattr(utils, loss)()
    else:
        raise ValueError(f"Loss function '{loss}' not found in utils")

    dls = []
    for ds, shuffle in zip(data, [True, False]):
        dl = utils.create_dataloader(ds=ds,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=0,
                                     drop_last=False)
        dls.append(dl)

    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(optimizer, scheduler['name'], **scheduler['params'])

    results = train(data=dls,
                    model=model,
                    model_params=model_params,
                    model_pth=model_pth,
                    criterion=criterion,
                    epochs=epochs,
                    patience=patience,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=metrics)
    
    return results