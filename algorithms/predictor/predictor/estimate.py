import numpy as np

def pred_error(X, X_pred):
    """
    Compute per-sample, per-feature absolute error between original and predicted data.

    :param X: Original data, shape (batch_size, seq_len, num_feats).
    :param X_pred: Predicted data, same shape as X.
    :return: Numpy array of errors, shape (batch_size, seq_len, num_feats).
    """
    return np.abs(X - X_pred)