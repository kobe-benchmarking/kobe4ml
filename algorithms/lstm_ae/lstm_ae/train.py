import torch
import time

from . import utils
from .loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def train(data, model_url, criterion, model, epochs, patience, lr, optimizer, scheduler, metrics):
    """
    Train the model on the provided data and calculate the test loss, MAE, and MSE.

    :param data: Tuple containing (train_data, val_data), where each is a DataLoader.
    :param model_url: URL reference for saving the best model (e.g., S3 storage).
    :param criterion: Loss function used to compute training and validation loss.
    :param model: The model to be trained.
    :param epochs: Maximum number of training epochs.
    :param patience: Number of epochs to wait for validation loss improvement before early stopping.
    :param lr: Learning rate for optimization.
    :param optimizer: Optimizer type or configuration for training.
    :param scheduler: Learning rate scheduler configuration.
    :param metrics: List of metric names to calculate (e.g., ['mae', 'mse']).
    :return: Dictionary containing metrics as defined in the input metrics list.
    """
    model.to(device)

    train_data, val_data = data
    batches = len(train_data)
    
    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(optimizer, scheduler['name'], **scheduler['params'])

    train_time = 0.0
    best_val_loss = float('inf')
    stationary = 0
    train_losses, val_losses, maes, mses = [], [], [], []

    for epoch in range(epochs):
        start = time.time()

        total_train_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0

        model.train()

        for _, (X, _) in enumerate(train_data):
            X = X.to(device)

            X, t = separate(src=X, c=[0,1], t=[2])
            X_dec, _ = model(X)

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
            for _, (X, _) in enumerate(val_data):
                X = X.to(device)

                X, t = separate(src=X, c=[0,1], t=[2])

                X_dec, _ = model(X)

                val_loss = criterion(X_dec, X)
                total_val_loss += val_loss.item()

                total_mae += mae(X, X_dec)
                total_mse += mse(X, X_dec)

        avg_val_loss = total_val_loss / batches
        avg_mae = total_mae / batches
        avg_mse = total_mse / batches

        val_losses.append(avg_val_loss)
        maes.append(avg_mae)
        mses.append(avg_mse)

        end = time.time()
        duration = end - start
        train_time += duration

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss

            stationary = 0

            # save model to s3 bucket using model_url reference
        else:
            stationary += 1

        if stationary >= patience:
            break

        scheduler.step(avg_val_loss)

    all_metrics = {
        'epochs': epoch + 1,
        'train_time': train_time,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'mae': avg_mae,
        'mse': avg_mse
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(params):
    """
    Main function to execute the testing workflow, including data preparation and model evaluation.
    """
    model_url, dls, num_feats, latent_seq_len, latent_num_feats, hidden_size, num_layers, dropout, seq_len, loss, epochs, patience, lr, optimizer, scheduler, metrics = params.values()

    samples, chunks = 7680, 32
    seq_len = samples // chunks

    model = LSTM_Autoencoder(seq_len=seq_len, 
                             num_feats=num_feats, 
                             latent_seq_len=latent_seq_len,
                             latent_num_feats=latent_num_feats,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout)

    if hasattr(utils, loss):
        criterion = getattr(utils, loss)()
    else:
        raise ValueError(f"Loss function '{loss}' not found in utils")
 
    metrics = train(data=dls,
                    model_url=model_url,
                    criterion=criterion,
                    model=model,
                    epochs=epochs,
                    patience=patience,
                    lr=lr,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=metrics)
    
    return metrics