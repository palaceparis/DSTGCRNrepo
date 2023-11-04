import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    abs_errors = np.abs(y_true - y_pred)
    percent_errors = abs_errors / y_true
    mape = np.mean(percent_errors)
    return mape


def mean_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    abs_errors = np.abs(y_true - y_pred)
    percent_errors = abs_errors / y_true
    mspe = np.mean(percent_errors**2)
    return mspe


def adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r2 = r2_score(y_true, y_pred)
    n_samples = y_true.shape[0]
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - 1 - 1)
    return adj_r2


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        "RMSPE": np.sqrt(mean_squared_percentage_error(y_true, y_pred)),
        "R-squared": r2_score(y_true, y_pred),
        "Adjusted R-squared": adjusted_r_squared(y_true, y_pred),
    }
    return metrics
