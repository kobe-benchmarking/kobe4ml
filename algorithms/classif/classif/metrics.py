from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def precision(y_true, y_pred_probs):
    """
    Compute precision using scikit-learn.
    y_true: numpy array of ground-truth labels (shape: N,)
    y_pred_probs: numpy array of logits/probabilities (shape: N x num_classes)
    """
    y_hat = np.argmax(y_pred_probs, axis=1)
    return precision_score(y_true, y_hat, average="macro", zero_division=0)

def recall(y_true, y_pred_probs):
    """
    Compute recall using scikit-learn.
    """
    y_hat = np.argmax(y_pred_probs, axis=1)
    return recall_score(y_true, y_hat, average="macro", zero_division=0)

def f1(y_true, y_pred_probs):
    """
    Compute F1 score using scikit-learn.
    """
    y_hat = np.argmax(y_pred_probs, axis=1)
    return f1_score(y_true, y_hat, average="macro", zero_division=0)
