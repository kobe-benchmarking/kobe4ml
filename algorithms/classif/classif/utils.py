import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn.functional as F

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
    
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        """
        Initialize the WeightedCrossEntropyLoss module.

        :param weights: dictionary
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = self.get_weights(weights)

    def get_weights(self, weights):
        """
        Extract weights from the given dictionary and convert them to a tensor.

        :param weights: dictionary
        :return: tensor
        """
        weights = [weights[i] for i in range(len(weights))]

        return torch.tensor(weights, dtype=torch.float)

    def forward(self, pred, true):
        """
        Compute the weighted cross-entropy loss.

        :param pred: tensor (batch_size * seq_len, num_classes)
        :param true: tensor (batch_size * seq_len)
        :return: tensor
        """
        if true.size(0) == 0 or pred.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        loss = F.cross_entropy(pred, true, weight=self.weights.to(pred.device))

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