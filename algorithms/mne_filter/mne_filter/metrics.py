import numpy as np

def mae(X, X_flt):
    """
    Compute Mean Absolute Error (MAE) manually.
    
    :param X: Original data, ndarray of shape (n_samples, n_features).
    :param X_flt: Filtered data, same shape as X.
    :return: Scalar MAE value.
    """
    return np.mean(np.abs(X - X_flt))

def mse(X, X_flt):
    """
    Compute Mean Squared Error (MSE) manually.
    
    :param X: Original data, ndarray of shape (n_samples, n_features).
    :param X_flt: Filtered data, same shape as X.
    :return: Scalar MSE value.
    """
    return np.mean((X - X_flt)**2)