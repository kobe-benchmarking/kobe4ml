# id = 'lstm_ae'
# name = 'LSTM_Autoencoder'
# num_feats = 2
# latent_seq_len = 1 
# latent_num_feats = 8 
# hidden_size = 4
# num_layers = 1
# dropout = 0.05

import torch
from tqdm import tqdm
import warnings

from . import utils
from .loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

def test(data, pth, criterion, model):
    """
    Test the model on the provided data and calculate the test loss.

    :param data: Data to test the model on.
    :param criterion: Loss function used to compute the test loss.
    :param model: The model to be evaluated.
    :param visualize: Whether to visualize the model's predictions.
    :param estimate: Whether to estimate the quality of the predictions.
    """
    model.load_state_dict(pth)
    model.to(device)
    model.eval()

    batches = len(data)
    total_test_loss = 0.0

    progress_bar = tqdm(enumerate(data), total=batches, desc=f'Evaluation', leave=True)

    with torch.no_grad():
        for _, (X, _) in progress_bar:
            X = X.to(device)

            X, t = separate(src=X, c=[0,1], t=[2])
            X_dec, _ = model(X)

            test_loss = criterion(X_dec, X)

            total_test_loss += test_loss.item()
            progress_bar.set_postfix(Loss=test_loss.item())

        avg_test_loss = total_test_loss / batches

    logger.info(f'\nTesting complete!\nTesting Loss: {avg_test_loss:.6f}\n')

def main(**params):
    """
    Main function to execute the testing workflow, including data preparation and model evaluation.
    """
    name, model_url, ds_url, _, num_feats, latent_seq_len, latent_num_feats, hidden_size, num_layers, dropout, batch_size, seq_len, loss = params

    samples, chunks = 7680, 32
    seq_len = samples // chunks

    bitbrain_dir = utils.get_dir(ds_url, 'bitbrain')
    raw_dir = utils.get_dir(ds_url, 'raw')

    get_boas_data(base_path=bitbrain_dir, output_path=raw_dir)
    
    datapaths = split_data(dir=raw_dir, train_size=43, val_size=3, test_size=10)
    
    _, _, test_df = get_dataframes(datapaths, seq_len=seq_len, exist=True)

    datasets = create_datasets(dataframes=(test_df,), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=batch_size, drop_last=False)

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
 
    test(data=dataloaders[0],
         pth=pth,
         criterion=criterion,
         model=model)