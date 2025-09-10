import numpy as np
import torch
from torch.utils.data import Dataset

from . import utils

logger = utils.get_logger(level='DEBUG')

def get_stats(dir, name, process, done=False, stats_from='train'):
    """
    Load structured .npz data and metadata, compute stats (mean, std, median, IQR) per column,
    and save the stats as a JSON file.

    :param dir: Directory containing {name}-{process}.npz and {name}.json.
    :param name: Dataset name prefix (e.g., 'bitbrain').
    :param process: Process type (e.g., 'train', 'val', 'infer').
    :param done: If True, skip the stats calculation.
    :param stats_from: Specifies which dataset split to use as reference.
    :return: Dict of stats keyed by column name.
    """
    data_path = utils.get_path(dir, filename=f"{name}-{process}.npz")
    meta_path = utils.get_path(dir, filename=f"{name}.json")
    stats_path = utils.get_path(dir, filename=f"{name}-stats.json")

    if done or stats_from == process:
        logger.info(f"Skipping stats calculation for {data_path}.")
        return utils.load_json(stats_path)

    data = utils.load_npz(data_path)
    metadata = utils.load_json(meta_path)

    stats = {}

    for key in data:
        for idx, col in enumerate(metadata[key]):
            if col in stats:
                continue

            col_values = data[key][:, idx]

            mean = np.mean(col_values)
            std = np.std(col_values)
            median = np.median(col_values)
            q75, q25 = np.percentile(col_values, [75, 25])
            iqr = q75 - q25

            stats[col] = {
                'mean': float(mean),
                'std': float(std),
                'median': float(median),
                'iqr': float(iqr)
            }

    utils.save_json(data=stats, path=stats_path)

    logger.info(f"Saved statistics JSON to {stats_path}.")

    return stats

def robust_normalize(dir, name, process, include, stats, done=False):
    """
    Normalize structured .npz dataset using robust scaling (median and IQR) from precomputed stats. Applies normalization only to specified columns across any sub-array.

    :param dir: Directory containing the dataset.
    :param name: Dataset base name (e.g., 'bitbrain').
    :param process: Process type (e.g., 'train', 'val', 'infer').
    :param include: List of column names to include in normalization.
    :param stats: Dict of precomputed stats (median, iqr) keyed by column name.
    :param done: If True, skip the normalization process.
    """
    data_path = utils.get_path(dir, filename=f"{name}-{process}.npz")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    if done:
        logger.info(f"Skipping robust normalization for {data_path}.")
        return

    data = utils.load_npz(data_path)
    metadata = utils.load_json(meta_path)

    data_norm = {k: v.copy().astype(np.float32) for k, v in data.items()}

    for key in data_norm:
        for idx, col in enumerate(metadata[key]):
            if col not in include:
               continue
            
            median = stats[col]["median"]
            iqr = stats[col]['iqr'] if stats[col]['iqr'] > 0 else 1.0

            data_norm[key][:, idx] = (data_norm[key][:, idx] - median) / iqr

    norm_path = utils.get_path(dir, filename=f"{name}-{process}-rbst-norm.npz")
    utils.save_npz(data_norm, norm_path)

    logger.info(f"Robust normalized data saved to {norm_path}.")

def standard_normalize(dir, name, process, include, stats, done=False):
    """
    Normalize structured .npz dataset using standard scaling (mean and std) from precomputed stats. Applies normalization only to specified columns across any sub-array.

    :param dir: Directory containing the dataset.
    :param name: Dataset base name (e.g., 'bitbrain').
    :param process: Process type (e.g., 'train', 'val', 'infer').
    :param include: List of column names to include in normalization.
    :param stats: Dict of precomputed stats (mean, std) keyed by column name.
    :param done: If True, skip the normalization process.
    """
    data_path = utils.get_path(dir, filename=f"{name}-{process}.npz")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    if done:
        logger.info(f"Skipping standard normalization for {data_path}.")
        return

    data = utils.load_npz(data_path)
    metadata = utils.load_json(meta_path)

    data_norm = {k: v.copy().astype(np.float32) for k, v in data.items()}

    for key in data_norm:
        for idx, col in enumerate(metadata[key]):
            if col not in include:
               continue

            mean = stats[col]["mean"]
            std = stats[col]["std"] if stats[col]["std"] > 0 else 1.0

            data_norm[key][:, idx] = (data_norm[key][:, idx] - mean) / std

    norm_path = utils.get_path(dir, filename=f"{name}-{process}-std-norm.npz")
    utils.save_npz(data_norm, norm_path)

    logger.info(f"Normalized data saved to {norm_path}.")

class TSDataset(Dataset):
    def __init__(self, dir, name, seq_len, full_epoch=7680, per_epoch=True, time_include=False):
        """
        Initializes a time series dataset. Returns for each sample:

            - X: current sequence
            - Xn: next sequence (t+1)
            - y: target labels for the current sequence

        :param dir: Directory containing the NumPy array dataset.
        :param name: Name of the dataset (e.g. bitbrain-train-std-norm.npy).
        :param seq_len: Length of the input sequence (number of time steps).
        :param full_epoch: Length of a full epoch in samples (default is 7680).
        :param per_epoch: Whether to create sequences in non-overlapping (True) or overlapping (False) epochs.
        :param time_include: Whether to include the time features in the input data.
        """
        self.seq_len = seq_len
        self.full_epoch = full_epoch
        self.per_epoch = per_epoch

        self.data_path = utils.get_path(dir, filename=f"{name}.npz")
        self.data = utils.load_npz(self.data_path)
        
        if time_include:
            self.X = np.concatenate([self.data["features"], self.data["time"]], axis=1)
        else:
            self.X = self.data["features"]

        self.y = self.data["labels"]

        logger.debug(f'Initialized dataset with: samples={self.num_samples}, seq_len={seq_len}, num_seqs={self.num_seqs}, full_epochs={self.num_epochs}.')

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        :return: Length of the dataset.
        """
        return self.num_seqs

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: Index of the sample.
        :return: Tuple of features and target tensors.
        """
        if self.per_epoch:
            start_idx = idx * self.seq_len
        else:
            start_idx = idx

        end_idx = start_idx + self.seq_len
        next_start_idx = end_idx
        next_end_idx = end_idx + self.seq_len

        if next_end_idx > len(self.X):
            next_start_idx = start_idx
            next_end_idx = end_idx

        X = self.X[start_idx:end_idx]
        Xn = self.X[next_start_idx:next_end_idx]
        y = self.y[start_idx:end_idx]

        X, Xn, y = torch.FloatTensor(X), torch.FloatTensor(Xn), torch.LongTensor(y)

        return X, Xn, y
    
    @property
    def num_samples(self):
        """
        Returns the total number of samples in the dataset.
        
        :return: Total number of samples.
        """
        return self.X.shape[0]
    
    @property
    def num_epochs(self):
        """
        Returns the number of full epochs available based on the dataset size.

        :return: Number of epochs.
        """
        return self.num_samples // self.full_epoch

    @property
    def max_seq_id(self):
        """
        Returns the maximum index for a sequence.

        :return: Maximum index for a sequence.
        """
        return self.num_samples - self.seq_len
    
    @property
    def num_seqs(self):
        """
        Returns the number of sequences that can be created from the dataset.

        :return: Number of sequences.
        """
        if self.per_epoch:
            return self.num_samples // self.seq_len
        else:
            return self.max_seq_id + 1