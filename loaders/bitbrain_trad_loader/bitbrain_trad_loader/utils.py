import numpy as np

def one_hot_encode(labels, num_classes=5):
    """
    Manually encode labels into one-hot format.
    
    :param labels: Array of integer labels.
    :param num_classes: Number of unique classes for one-hot encoding.
    :return: One-hot encoded labels.
    """
    one_hot = np.zeros((len(labels), num_classes), dtype=int)
    
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
        
    return one_hot