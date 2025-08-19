import random
import multiprocessing
import numpy as np
import os
from torch.utils.data import DataLoader

from . import utils

logger = utils.get_logger(level='DEBUG')

def shift_labels(dir, name):
    """
    Load the structured .npz dataset and metadata, shift label values to start from 0,
    and save the updated .npz back to the same path.

    :param dir: Directory containing the dataset and metadata files.
    :param name: Dataset name prefix (e.g., 'bitbrain').
    """
    data_path = utils.get_path(dir, filename=f"{name}.npz")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    data = utils.load_npz(data_path)
    metadata = utils.load_json(meta_path)

    label_cols = metadata["labels"]
    weight_cols = metadata["weights"]

    labels_values = data["labels"]
    weights_values = data["weights"]

    for i, label in enumerate(label_cols):
        label_col_values = labels_values[:, i]
        unique_vals = np.unique(label_col_values)

        mapping = {val: k for k, val in enumerate(sorted(unique_vals))}

        mapping_func = np.vectorize(mapping.get)
        labels_values[:, i] = mapping_func(label_col_values)

        if label in weight_cols:
            j = weight_cols.index(label)
            weights_values[:, j] = mapping_func(label_col_values)

    utils.save_npz(data=data, path=data_path)
    logger.info(f"Shifted labels {label_cols} in {data_path} so values start at 0.")

def split_data(dir, name, train_size=0.75, val_size=0.25, test_size=0):
    """
    Split structured .npz dataset into training, validation, and testing sets based on unique values in the split column.

    :param dir: Directory containing the dataset.
    :param name: Name of the dataset (e.g., 'bitbrain').
    :param train_size: Proportion of nights to use for training.
    :param val_size: Proportion of nights to use for validation.
    :param test_size: Proportion of nights to use for testing.
    """
    data_path = utils.get_path(dir, filename=f'{name}.npz')
    meta_path = utils.get_path(dir, filename=f'{name}.json')

    train_path = utils.get_path(dir, filename=f'{name}-train.npz')
    val_path = utils.get_path(dir, filename=f'{name}-val.npz')
    test_path = utils.get_path(dir, filename=f'{name}-test.npz')

    data = utils.load_npz(data_path)
    metadata = utils.load_json(meta_path)

    logger.info(f"Loaded data from {data_path} and metadata from {meta_path}.")

    split_col = metadata["split"][0]

    if split_col is None:
        total = len(next(iter(data.values())))
        indices = list(range(total))

        random.seed(42)
        random.shuffle(indices)

        train_end = int(total * train_size)
        val_end = train_end + int(total * val_size)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        def subset_data(idxs):
            return {k: v[idxs] for k, v in data.items()}

        train_data = subset_data(train_idx)
        val_data = subset_data(val_idx)
        test_data = subset_data(test_idx)
    
    else:
        split_values = data["split"].flatten() 
        unique_values = list(np.unique(split_values))

        logger.info(f"Unique values for split column '{split_col}': {unique_values}.")

        random.seed(42)
        random.shuffle(unique_values)

        n = len(unique_values)

        if train_size + val_size + test_size == 0:
            raise ValueError("All sets have size 0, which is invalid.")
        if train_size + val_size + test_size > 1:
            raise ValueError("Sum of train, val, and test sizes must not exceed 1.")
        
        raw = {'train': round(n * train_size), 'val': round(n * val_size), 'test': round(n * test_size)}
        total = sum(raw.values())

        while total > n:
            max_key = max(raw, key=raw.get)
            raw[max_key] -= 1
            total -= 1
            
        while total < n:
            min_key = min(raw, key=raw.get)
            raw[min_key] += 1
            total += 1
        
        train_end = raw['train']
        val_end = train_end + raw['val']

        train_vals = set(unique_values[:train_end])
        val_vals = set(unique_values[train_end:val_end])
        test_vals = set(unique_values[val_end:])

        def filter_data(values):
            filtered = {}
            mask = np.isin(split_values, list(values))

            for k, v in data.items():
                filtered[k] = v[mask]

            return filtered

        train_data = filter_data(train_vals)
        val_data = filter_data(val_vals)
        test_data = filter_data(test_vals)

        logger.info(f"Train values: {sorted(train_vals)}")
        logger.info(f"Validation values: {sorted(val_vals)}")
        logger.info(f"Test values: {sorted(test_vals)}")

        assert train_vals.isdisjoint(val_vals), "Overlap in train and val nights!"
        assert train_vals.isdisjoint(test_vals), "Overlap in train and test nights!"
        assert val_vals.isdisjoint(test_vals), "Overlap in val and test nights!"   

    utils.save_npz(train_data, train_path)
    utils.save_npz(val_data, val_path)
    utils.save_npz(test_data, test_path)

    logger.info(f"Data split into train ({len(next(iter(train_data.values())))} samples), "
        f"val ({len(next(iter(val_data.values())))} samples), "
        f"test ({len(next(iter(test_data.values())))} samples).")

def extract_weights(dir, name):
    """
    Calculate class weights from the training structured .npz dataset to handle class imbalance, and save them to a JSON file. Supports multiple weight columns.

    :param dir: Directory to save the weights file.
    :param name: Name of the dataset (e.g., 'bitbrain').
    :return: Dictionary of class weights.
    """
    data_path = utils.get_path(dir, filename=f'{name}-train.npz')
    meta_path = utils.get_path(dir, filename=f'{name}.json')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}. Cannot extract weights.")

    data = utils.load_npz(data_path)
    metadata = utils.load_json(meta_path)

    weights_values = data["weights"]
    weights_cols = metadata["weights"]

    weights = {}

    for idx, col in enumerate(weights_cols):
        col_values = weights_values[:, idx]

        unique_labels, counts = np.unique(col_values, return_counts=True)
        occs = dict(zip(unique_labels, counts))

        inverse_occs = {int(k): 1 / (v + 1e-10) for k, v in occs.items()}
        total = sum(inverse_occs.values())

        col_weights = {int(k): v / total for k, v in inverse_occs.items()}
        weights[col] = dict(sorted(col_weights.items()))

    weights_path = utils.get_path(dir, filename=f'{name}-weights.json')
    utils.save_json(data=weights, path=weights_path)
    logger.info(f"Saved class weights to {weights_path}: {weights}")

    return weights

def create_dataloader(ds, batch_size, shuffle=[True, False, False], num_workers=None, drop_last=False):
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

    logger.info(f'System has {cpu_cores} CPU cores. Using {num_workers}/{cpu_cores} workers for data loading.')

    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return dl