import os
import boto3
from io import BytesIO
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched

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
    
def load_model_from_s3(url):
    """
    Load the model directly from S3 into memory.
    
    :param url: S3 URL of the model (e.g., 's3://bucket-name/path/to/model.pth')
    :return: Loaded model
    """
    s3 = boto3.client('s3')
    s3_parts = url.replace('s3://', '').split('/', 1)
    bucket_name = s3_parts[0]
    key = s3_parts[1]

    model_byte_stream = BytesIO()
    s3.download_fileobj(bucket_name, key, model_byte_stream)
    model_byte_stream.seek(0)

    model_state_dict = torch.load(model_byte_stream, map_location='cpu')

    return model_state_dict