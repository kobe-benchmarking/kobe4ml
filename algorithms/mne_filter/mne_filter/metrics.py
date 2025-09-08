import numpy as np

def mae(X, X_dec):
    """
    Compute Mean Absolute Error (MAE) manually.
    
    :param X: Original data, ndarray of shape (n_samples, n_features)
    :param X_dec: Reconstructed/filtered data, same shape as X
    :return: Scalar MAE value
    """
    return np.mean(np.abs(X - X_dec))

def mse(X, X_dec):
    """
    Compute Mean Squared Error (MSE) manually.
    
    :param X: Original data, ndarray of shape (n_samples, n_features)
    :param X_dec: Reconstructed/filtered data, same shape as X
    :return: Scalar MSE value
    """
    return np.mean((X - X_dec)**2)