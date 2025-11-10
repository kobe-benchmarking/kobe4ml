import numpy as np

def rec_error(X, X_dec):
    """
    Compute per-sample, per-feature absolute error between original and decoded data.

    :param X: Original data, shape (n_samples, seq_len, num_feats).
    :param X_dec: Decoded data, same shape as X.
    :return: Numpy array of errors, shape (n_samples, seq_len, num_feats).
    """
    return np.abs(X - X_dec)