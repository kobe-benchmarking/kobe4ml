import numpy as np

def filter_error(x, x_f):
    """
    Compute per-sample, per-feature absolute error between original and filtered data.

    :param x: Original data, shape (n_samples, n_features).
    :param x_f: Filtered data, same shape as x.
    :return: Numpy array of errors, shape (n_samples, n_features).
    """
    return np.abs(x - x_f)