import torch
from tqdm import tqdm
import warnings

from . import utils
from .loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

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

def test(data, pth, criterion, model):
    """
    Test the model on the provided data and calculate the test loss, MAE, and MSE.

    :param data: Data to test the model on.
    :param criterion: Loss function used to compute the test loss.
    :param model: The model to be evaluated.
    """
    model.load_state_dict(pth)
    model.to(device)
    model.eval()

    batches = len(data)
    total_test_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0

    progress_bar = tqdm(enumerate(data), total=batches, desc=f'Evaluation', leave=True)

    with torch.no_grad():
        for _, (X, _) in progress_bar:
            X = X.to(device)

            X, t = separate(src=X, c=[0,1], t=[2])
            X_dec, _ = model(X)

            test_loss = criterion(X_dec, X)
            total_test_loss += test_loss.item()

            total_mae += mae(X, X_dec)
            total_mse += mse(X, X_dec)

            progress_bar.set_postfix(Loss=test_loss.item())

    avg_test_loss = total_test_loss / batches
    avg_mae = total_mae / batches
    avg_mse = total_mse / batches

    logger.info(f'\nTesting complete!\n'
                f'Testing Loss: {avg_test_loss:.6f}\n'
                f'MAE: {avg_mae:.6f}\n'
                f'MSE: {avg_mse:.6f}\n')

    return {
        'mae': avg_mae,
        'mse': avg_mse
    }

def main(params):
    """
    Main function to execute the testing workflow, including data preparation and model evaluation.
    """
    name, model_url, dls, id, num_feats, latent_seq_len, latent_num_feats, hidden_size, num_layers, dropout, batch_size, seq_len, loss = params.values()

    samples, chunks = 7680, 32
    seq_len = samples // chunks

    model_class = globals()[name]
    model = model_class(seq_len=seq_len, 
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
                   model=model)