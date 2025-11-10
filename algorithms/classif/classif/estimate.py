import numpy as np

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