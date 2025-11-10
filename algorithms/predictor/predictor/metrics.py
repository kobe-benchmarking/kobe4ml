import torch

def mae(X, X_pred):
    """
    Compute Mean Absolute Error (MAE) manually.

    :param X: Original input tensor.
    :param X_pred: Predicted output tensor.
    :return: MAE value.
    """
    return torch.mean(torch.abs(X - X_pred)).item()

def mse(X, X_pred):
    """
    Compute Mean Squared Error (MSE) manually.

    :param X: Original input tensor.
    :param X_pred: Predicted output tensor.
    :return: MSE value.
    """
    return torch.mean((X - X_pred) ** 2).item()