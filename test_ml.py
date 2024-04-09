import pytest
import numpy as np
from ml.model import train_model, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier

def test_train_model_returns_model():
    # Sample data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    # Train the model
    model = train_model(X_train, y_train)
    # Assert the type of the trained model
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics_expected_values():
    # Sample true and predicted labels
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    # Compute model metrics
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    # Assert expected values
    assert round(precision, 4) == 0.6667
    assert round(recall, 4) == 0.6667
    assert round(f1, 4) == 0.6667

def test_data_types():
    # Sample data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[7, 8], [9, 10]])
    y_test = np.array([1, 0])
    # Assert data types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert X_train.dtype == np.int64
    assert y_train.dtype == np.int64
    assert X_test.dtype == np.int64
    assert y_test.dtype == np.int64


