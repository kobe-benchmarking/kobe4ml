import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from . import utils

def train(data, model_url, model, metrics):
    """
    Train the model on the provided data and calculate the accuracy, MAE, and MSE.

    :param data: Tuple containing (features, labels), where each is a NumPy array.
    :param model_url: URL reference for saving the model (e.g., S3 storage).
    :param model: The model to be trained (Random Forest model).
    :param metrics: List of metric names to calculate (e.g., ['accuracy', 'mae', 'mse']).
    :return: Dictionary containing accuracy, MAE, MSE, and training time.
    """
    start = time.time()
    
    X, y = data
    model.fit(X, y)

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    end = time.time()
    train_time = end - start

    utils.save_model_to_s3(model, model_url)

    all_metrics = {
        'accuracy': accuracy,
        'mae': mae,
        'mse': mse,
        'train_time': train_time,
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(params):
    """
    Main function to execute the training workflow, including data preparation and model training.
    """
    model_url, data, n_estimators, criterion, max_depth, metrics = params.values()

    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)

    metrics = train(data, model_url, model, metrics)
    
    return metrics