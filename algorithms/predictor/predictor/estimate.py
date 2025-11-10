import numpy as np

def pred_error(X, X_pred):
    """
    Compute per-sample, per-feature absolute error between original and predicted data.

    :param X: Original data, shape (n_samples, seq_len, num_feats).
    :param X_pred: Predicted data, same shape as X.
    :return: Numpy array of errors, shape (n_samples, seq_len, num_feats).
    """
    return np.abs(X - X_pred)

def attn_error(attn_matrix, absolute=False):
    """
    Compute per-sample error or magnitude using the attention matrix.

    :param attn_matrix: Attention matrix output from the model, shape (n_samples, seq_len, num_feats).
    :param absolute: If True, returns absolute values; if False, keeps original signed values.
    :return: Numpy array of errors, same shape as attn_matrix.
    """
    if absolute:
        return np.abs(attn_matrix)
    else:
        return attn_matrix