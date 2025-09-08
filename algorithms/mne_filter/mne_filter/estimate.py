import numpy as np

def filter_error(X, X_flt):
    """
    Compute per-sample, per-feature absolute error between original and filtered data.

    :param X: Original data, shape (n_samples, n_features).
    :param X_flt: Filtered data, same shape as X.
    :return: Numpy array of errors, shape (n_samples, n_features).
    """
    return np.abs(X - X_flt)