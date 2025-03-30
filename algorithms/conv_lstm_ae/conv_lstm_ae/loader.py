import torch

from . import utils

logger = utils.get_logger(level='CRITICAL')

def separate(src, c, t):
    """
    Separates channels and time features from the source tensor.

    :param src: Tensor of shape (batch_size, seq_len, num_feats).
    :param c: Range of channel features.
    :param t: Range of time features.
    :return: Tuple of (channels, time) tensors.
    """
    channels = src[:, :, c]
    time = src[:, :, t]

    return channels, time

def aggregate_seqs(data):
    """
    Aggregates the tensor by reducing the sequence length to a single time step.

    :param data: Tensor of shape (batch_size, seq_len, num_feats).
    :return: Tensor of shape (batch_size, 1, num_feats).
    """
    return data[:, 0:1, :]

def merge(c, t):
    """
    Concatenates channel and time feature tensors along the feature dimension.

    :param c: Tensor of shape (batch_size, seq_len, num_channels_feats).
    :param t: Tensor of shape (batch_size, seq_len, num_time_feats).
    :return: Tensor of shape (batch_size, seq_len, num_channels_feats + num_time_feats).
    """
    return torch.cat((c, t), dim=2)