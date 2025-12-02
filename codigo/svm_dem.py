import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVC
import pandas as pd

def kfold_cross_validation(
    X: np.ndarray, y: np.ndarray, n_splits: int, algorithm_func, *args, **kwargs
) -> tuple:
    """
    Performs k-fold cross-validation using the specified algorithm function.

    Parameters:
        X : np.ndarray
            Input data.
        y : np.ndarray
            Target labels.
        n_splits : int
            Number of folds.
        algorithm_func : callable
            Algorithm function to be used for each fold.
        *args : Any
            Variable length arguments to be passed to the algorithm function.
        **kwargs : Any
            Keyword arguments to be passed to the algorithm function.

    Returns:
        tuple
            Results and test data for each fold.
    """
    results = []
    tests = []
    kf = KFold(n_splits=n_splits)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result = algorithm_func(X_train, y_train, X_test, *args, **kwargs)
        results.extend(result)
        tests.extend(y_test)
    return np.array(results), np.array(tests)

def find_best_params(
    model, param_grid: dict, X_rx: np.ndarray, sym_tx: np.ndarray
) -> dict:
    """
    Finds the best parameters for a given model using specific data.

    Parameters:
        model: ML model to optimize.
        param_grid: Dictionary of model parameters.
        X_rx: Input data.
        sym_tx: Validated output data.

    Returns:
        dict: Optimized parameter dictionary.
    """
    X = X_rx
    y = sym_tx

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3)

    grid = GridSearchCV(model(), param_grid, verbose=0)

    grid.fit(X_train, y_train)

    return grid.best_params_

def best_parameters_svm(df, tx):
    param_grid_svm = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    best_params = find_best_params(SVC, param_grid_svm, df, tx)
    return best_params['C'], best_params["gamma"]

def svm(X_rx: np.ndarray, sym_tx: np.ndarray, n_splits: int = 5) -> tuple:
    """
     Demodulates using SVM with k-fold cross-validation.

    Parameters:
        X_rx : np.ndarray
            Received constellation.
        sym_tx : np.ndarray
            Transmitted symbols.
        C : float
            Parameter C for the SVM algorithm.
        gamma : float
            Parameter gamma for the SVM algorithm.
        n_splits : int, optional
            Number of folds for k-fold cross-validation. Default is 5.

    Returns:
        tuple
            Demodulated constellation and test data.
    """
    def algorithm_func(X_train, y_train, X_test, C, gamma):
        model = SVC(C=C, gamma=gamma)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    X = X_rx
    y = sym_tx

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    c1, gamma1 = best_parameters_svm(X_train, y_train)

    return kfold_cross_validation(X, y, n_splits, algorithm_func, C=c1, gamma=gamma1)
