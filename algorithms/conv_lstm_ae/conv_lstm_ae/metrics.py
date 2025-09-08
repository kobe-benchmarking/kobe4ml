import torch

def mae(X, X_dec):
    """
    Compute Mean Absolute Error (MAE) manually.

    :param X: Original input tensor.
    :param X_dec: Reconstructed output tensor.
    :return: MAE value.
    """
    return torch.mean(torch.abs(X - X_dec)).item()

def mse(X, X_dec):
    """
    Compute Mean Squared Error (MSE) manually.

    :param X: Original input tensor.
    :param X_dec: Reconstructed output tensor.
    :return: MSE value.
    """
    return torch.mean((X - X_dec) ** 2).item()