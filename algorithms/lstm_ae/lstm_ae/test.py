import torch

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

def test(data, pth, criterion, model, metrics):
    """
    Test the model on the provided data and calculate the test loss, MAE, and MSE.

    :param data: Data to test the model on.
    :param criterion: Loss function used to compute the test loss.
    :param model: The model to be evaluated.
    :param metrics: List of metric names to calculate (e.g., ['mae', 'mse']).
    :return: Dictionary containing metrics as defined in the input metrics list.
    """
    model.load_state_dict(pth)
    model.to(device)
    model.eval()

    batches = len(data)
    total_test_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0

    with torch.no_grad():
        for _, (X, _) in enumerate(data):
            X = X.to(device)

            X, t = separate(src=X, c=[0,1], t=[2])
            X_dec, _ = model(X)

            test_loss = criterion(X_dec, X)
            total_test_loss += test_loss.item()

            total_mae += mae(X, X_dec)
            total_mse += mse(X, X_dec)

    avg_test_loss = total_test_loss / batches
    avg_mae = total_mae / batches
    avg_mse = total_mse / batches

    all_metrics = {
        'test_loss': avg_test_loss,
        'mae': avg_mae,
        'mse': avg_mse
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(params):
    """
    Main function to execute the testing workflow, including data preparation and model evaluation.
    """
    model_url, dls, num_feats, latent_seq_len, latent_num_feats, hidden_size, num_layers, dropout, seq_len, loss, metrics = params.values()

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
    
    pth = utils.load_model_from_s3(model_url)
 
    metrics = test(data=dls[0],
                   pth=pth,
                   criterion=criterion,
                   model=model,
                   metrics=metrics)
    
    return metrics
