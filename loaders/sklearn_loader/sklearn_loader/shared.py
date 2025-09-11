import numpy as np
import os
from sklearn.model_selection import train_test_split

from . import utils

logger = utils.get_logger(level='DEBUG')

def shift_labels(dir, name, done=False):
    """
    Load the structured .npz dataset and metadata, shift label values to start from 0,
    and save the updated .npz back to the same path.

    :param dir: Directory containing the dataset and metadata files.
    :param name: Dataset name prefix (e.g., 'bitbrain').
    :param done: If True, skip the shifting process.
    """
    data_path = utils.get_path(dir, filename=f"{name}.npz")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    if done:
        logger.info(f"Skipping label shifting for {data_path}.")
        return

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

def split_data(dir, name, train_size=0.8, infer_size=0.2, done=False):
    """
    Split structured .npz dataset into training and inference sets based on unique values in the split column.
    Uses scikit-learn's train_test_split for randomization.

    :param dir: Directory containing the dataset.
    :param name: Name of the dataset (e.g., 'bitbrain').
    :param train_size: Proportion of data (or groups) for training.
    :param infer_size: Proportion for inference.
    :param done: If True, skip the splitting process.
    """
    data_path = utils.get_path(dir, filename=f'{name}.npz')
    meta_path = utils.get_path(dir, filename=f'{name}.json')

    if done:
        logger.info(f"Skipping data splitting for {data_path}.")
        return   

    train_path = utils.get_path(dir, filename=f'{name}-train.npz')
    infer_path = utils.get_path(dir, filename=f'{name}-infer.npz')

    data = utils.load_npz(data_path)
    metadata = utils.load_json(meta_path)

    logger.info(f"Loaded data from {data_path} and metadata from {meta_path}.")

    split_col = metadata["split"][0]

    if split_col is None:
        total = len(next(iter(data.values())))
        indices = list(range(total))

        train_idx, infer_idx = train_test_split(
            indices, train_size=train_size, test_size=infer_size, random_state=42, shuffle=True
        )

        def subset_data(idxs):
            return {k: v[idxs] for k, v in data.items()}

        train_data = subset_data(train_idx)
        infer_data = subset_data(infer_idx)

    else:
        split_values = data["split"].flatten()
        unique_values = np.unique(split_values)

        train_vals, infer_vals = train_test_split(
            unique_values, train_size=train_size, test_size=infer_size, random_state=42, shuffle=True
        )

        def filter_data(values):
            mask = np.isin(split_values, values)
            return {k: v[mask] for k, v in data.items()}

        train_data = filter_data(train_vals)
        infer_data = filter_data(infer_vals)

        logger.info(f"Train values: {sorted(train_vals)}")
        logger.info(f"Infer values: {sorted(infer_vals)}")

        assert train_vals.size + infer_vals.size == unique_values.size, "Mismatch in group splitting!"

    utils.save_npz(train_data, train_path)
    utils.save_npz(infer_data, infer_path)

    logger.info(f"Data split into train ({len(next(iter(train_data.values())))} samples), "
                f"infer ({len(next(iter(infer_data.values())))} samples).")

def extract_weights(dir, name, process, done=False, weights_from='train'):
    """
    Calculate class weights from the training structured .npz dataset to handle class imbalance, and save them to a JSON file. Supports multiple weight columns.

    :param dir: Directory to save the weights file.
    :param name: Name of the dataset (e.g., 'bitbrain').
    :param process: Process type (e.g., 'train', 'val', 'infer').
    :param done: If True, skip the weight extraction process.
    :param weights_from: Specifies which dataset split to use as reference.
    """
    data_path = utils.get_path(dir, filename=f'{name}-train.npz')
    meta_path = utils.get_path(dir, filename=f'{name}.json')
    weights_path = utils.get_path(dir, filename=f'{name}-weights.json')

    if done or weights_from != process:
        logger.info(f"Skipping weight extraction for {data_path}.")
        return utils.load_json(weights_path)

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

    utils.save_json(data=weights, path=weights_path)
    logger.info(f"Saved class weights to {weights_path}: {weights}")

def create_dataset(dir, name, time_include):
    """
    Load a structured .npz dataset and return (X, y) for sklearn models.

    :param dir: Directory containing the dataset.
    :param name: Dataset base name without .npz extension (e.g., 'bitbrain-train-std-norm').
    :param time_include: Whether to include the time features in the input data.
    :return: Tuple (X, y) as numpy arrays.
    """
    data_path = utils.get_path(dir, filename=f"{name}.npz")
    data = utils.load_npz(data_path)

    X = data["features"]
    y = data["labels"]
    t = data["time"]

    if time_include:
        X = np.concatenate([X, t], axis=1)

    logger.debug(f"Created sklearn dataset from {name}: X={X.shape}, y={y.shape}")

    return X, y