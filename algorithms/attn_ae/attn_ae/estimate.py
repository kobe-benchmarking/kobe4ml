import numpy as np

def rec_error(X, X_dec):
    """
    Compute per-sample, per-feature absolute error between original and decoded data.

    :param X: Original data, shape (n_samples, seq_len, num_feats).
    :param X_dec: Decoded data, same shape as X.
    :return: Numpy array of errors, shape (n_samples, seq_len, num_feats).
    """
    return np.abs(X - X_dec)

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