import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import multiprocessing
from torch.utils.data import DataLoader
import pickle

def get_logger(level='DEBUG'):
    """
    Create and configure a logger object with the specified logging level.

    :param level: Logging level to set for the logger. Default is 'DEBUG'.
    :return: Logger object configured with the specified logging level.
    """
    logger = logging.getLogger(__name__)

    level_name = logging.getLevelName(level)
    logger.setLevel(level_name)
    
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def get_dir(*sub_dirs):
    """
    Retrieve or create a directory path based on the script's location and the specified subdirectories.

    :param sub_dirs: List of subdirectories to append to the script's directory.
    :return: Full path to the directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(script_dir, *sub_dirs)

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def get_path(*dirs, filename):
    """
    Construct a full file path by combining directory paths and a filename.

    :param dirs: List of directory paths.
    :param filename: Name of the file.
    :return: Full path to the file.
    """
    dir_path = get_dir(*dirs)
    path = os.path.join(dir_path, filename)

    return path

def get_optim(name, model, lr):
    """
    Get optimizer object based on name, model, and learning rate.

    :param name: Name of the optimizer class.
    :param model: Model to optimize.
    :param lr: Learning rate for the optimizer.
    :return: Optimizer object.
    """
    optim_class = getattr(optim, name)
    optimizer = optim_class(model.parameters(), lr=lr)

    return optimizer

def get_sched(optimizer, name, **params):
    """
    Get scheduler object based on optimizer and additional parameters.

    :param optimizer: Optimizer object to schedule.
    :param name: Name of the scheduler.
    :param params: Additional parameters for the scheduler.
    :return: Scheduler object.
    """
    sched_class = getattr(sched, name)
    scheduler = sched_class(optimizer, **params)

    return scheduler
    
class BlendedLoss(nn.Module):
    def __init__(self, p=1.0, epsilon=1e-6, blend=0.8):
        """
        Initialize the BlendedLoss module.

        :param p: Power to which the differences are raised.
        :param epsilon: Small value added for numerical stability.
        :param blend: Blend factor between median and mean.
        """
        super(BlendedLoss, self).__init__()
        self.p = p
        self.epsilon = epsilon
        self.blend = blend

    def forward(self, input, target):
        """
        Compute the blended loss between the input and target.

        :param input: Tensor containing the predicted values.
        :param target: Tensor containing the target values.
        :return: Computed blended loss.
        """
        diff = torch.abs(input - target) + self.epsilon

        powered_diff = diff ** self.p
        median_diff = (1 - self.blend) * torch.median(powered_diff)
        mean_diff = self.blend * torch.mean(powered_diff)
        
        loss = median_diff + mean_diff
        
        return loss

def load_pth(path):
    """
    Load a PyTorch model's state_dict from a local file.

    :param path: Local path of the model (e.g., 'models/attn_ae.pth')
    :return: Loaded state_dict
    """
    state_dict = torch.load(path, map_location='cpu')

    return state_dict

def save_pth(model, path):
    """
    Save a PyTorch model's state_dict locally.

    :param model: PyTorch model to save.
    :param path: Local path where the model will be saved (e.g., 'models/attn_ae.pth').
    """
    torch.save(model.state_dict(), path)

def load_json(path):
    """
    Load a JSON file from the given path.

    :param path: Full path to the .json file.
    :return: Parsed JSON as a Python dict.
    """
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    """
    Save a Python dict to a JSON file at the specified path.

    :param data: Data to save as JSON.
    :param path: Full path to the .json file.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def create_dataloader(ds, batch_size, shuffle=False, num_workers=None, drop_last=False):
    """
    Create DataLoader object for the specified dataset.

    :param ds: Dataset object.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data at every epoch.
    :param num_workers: Number of subprocesses to use for data loading (default is all available CPU cores).
    :param drop_last: Whether to drop the last incomplete batch.
    :return: DataLoader object.
    """
    cpu_cores = multiprocessing.cpu_count()

    if num_workers is None:
        num_workers = cpu_cores

    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return dl

def detect_device():
    """
    Detects the best available device for PyTorch. Works for CPU, CUDA, and Apple MPS (Metal).

    :return: torch.device object.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def save_pickle(obj, path):
    """
    Save a Python object to a pickle file.

    :param obj: Python object to save.
    :param path: File path where to save the pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """
    Load a Python object from a pickle file.

    :param path: File path of the pickle.
    :return: Python object loaded from pickle.
    """
    with open(path, "rb") as f:
        return pickle.load(f)