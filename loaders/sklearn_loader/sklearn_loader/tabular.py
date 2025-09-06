import numpy as np

from . import utils

logger = utils.get_logger(level='DEBUG')

def get_stats(dir, name, process, done=False):
    """
    Load structured .npz train data and metadata, compute stats (mean, std, median, IQR) per column,
    and save the stats as a JSON file.

    :param dir: Directory containing {name}-{process}.npz and {name}.json.
    :param name: Dataset name prefix (e.g., 'bitbrain').
    :param process: Process type (e.g., 'train', 'val', 'infer').
    :param done: If True, skip the stats calculation.
    :return: Dict of stats keyed by column name.
    """
    data_path = utils.get_path(dir, filename=f"{name}-{process}.npz")
    meta_path = utils.get_path(dir, filename=f"{name}.json")
    stats_path = utils.get_path(dir, filename=f"{name}-stats.json")

    data = utils.load_npz(data_path)
    metadata = utils.load_json(meta_path)

    if done:
        logger.info(f"Skipping stats calculation for {data_path}.")
        return utils.load_json(stats_path)

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