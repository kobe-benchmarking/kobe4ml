import time
import warnings
import yaml

from . import utils
from .loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

def train(data, epochs, patience, lr, criterion, model, optimizer, scheduler):
    """
    Trains the model using the provided training data.

    :param data: Tuple containing training and validation data loaders.
    :param epochs: Number of epochs to train the model.
    :param patience: Number of epochs with no improvement after which training will be stopped.
    :param lr: Learning rate for the optimizer.
    :param criterion: Loss function used for training.
    :param model: The model to be trained.
    :param optimizer: Optimizer to use for training.
    :param scheduler: Learning rate scheduler configuration.
    :param visualize: Flag to indicate whether to visualize training progress.
    """
    model.to(device)

    train_data, val_data = data
    batches = len(train_data)

    logger.info(f"Number of training iterations per epoch: {batches}")

    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(optimizer, scheduler['name'], **scheduler['params'])

    train_time = 0.0
    best_val_loss = float('inf')
    stationary = 0
    train_losses, val_losses = [], []

    checkpoints = {'epochs': 0, 
                   'best_epoch': 0, 
                   'best_train_loss': float('inf'), 
                   'best_val_loss': float('inf'),
                   'train_time': 0.0 }

    for epoch in range(epochs):
        start = time.time()
        total_train_loss = 0.0

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

        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        end = time.time()
        duration = end - start
        train_time += duration

        logger.info(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, Duration: {duration:.2f}s')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0

            logger.info(f'New best val found! ~ Epoch [{epoch + 1}/{epochs}], Val Loss {avg_val_loss}')

            path = utils.get_path('..', '..', 'models', filename=f'{config['id']}.pth')
            torch.save(model.state_dict(), path)

            checkpoints.update({
                'best_epoch': epoch + 1, 
                'best_train_loss': avg_train_loss, 
                'best_val_loss': best_val_loss
            })
                
        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f'Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.')
            break

        scheduler.step(avg_val_loss)

    checkpoints.update({
        'epochs': epoch + 1,
        'train_time': train_time})
    
    cfn = utils.get_path('..', '..', 'static', config['id'], filename='train_checkpoints.json')
    utils.save_json(data=checkpoints, filename=cfn)

    logger.info(f'\nTraining complete!\nFinal Training Loss: {avg_train_loss:.6f} & Validation Loss: {best_val_loss:.6f}\n')

def main():
    """
    Main function to execute the training process. Prepares data, initializes the model, and starts training.
    """
    samples, chunks = 7680, 32
    seq_len = samples // chunks

    bitbrain_dir = utils.get_dir('..', '..', 'data', 'bitbrain')
    raw_dir = utils.get_dir('..', '..', 'data', 'raw')

    get_boas_data(base_path=bitbrain_dir, output_path=raw_dir)

    datapaths = split_data(dir=raw_dir, train_size=43, val_size=3, test_size=10)
    
    train_df, val_df, _ = get_dataframes(datapaths, seq_len=seq_len, exist=True)

    datasets = create_datasets(dataframes=(train_df, val_df), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=512, drop_last=False)

    model_class = globals()[config['name']]
    model = model_class(seq_len=seq_len, **config['params'])
        
    train(data=dataloaders,
          epochs=1000,
          patience=30,
          lr=1e-4,
          criterion=utils.BlendedLoss(p=1.0, blend=0.8),
          model=model,
          optimizer='Adam',
          scheduler={"name": 'ReduceLROnPlateau',"params": {'factor': 0.99, 'patience': 3}})