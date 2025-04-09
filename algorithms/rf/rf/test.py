import time
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from . import utils

def test(data, model, metrics):
    """
    Evaluate the trained model on the provided test data and calculate the accuracy, MAE, and MSE.

    :param data: Tuple containing (features, labels), where each is a NumPy array.
    :param model: The trained model to be evaluated (Random Forest model).
    :param metrics: List of metric names to calculate (e.g., ['accuracy', 'mae', 'mse']).
    :return: Dictionary containing accuracy, MAE, MSE, and evaluation time.
    """
    start = time.time()
    
    X, y = data
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    end = time.time()
    test_time = end - start

    all_metrics = {
        'accuracy': accuracy,
        'mae': mae,
        'mse': mse,
        'test_time': test_time,
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(params):
    """
    Main function to execute the evaluation workflow, including data preparation and model evaluation.
    """
    model_url, data, metrics = params.values()

    model = utils.load_model_from_s3(model_url)

    metrics = test(data, model_url, model, metrics)
    
    return metrics
