import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
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

def decisiontree(X_rx: np.ndarray, sym_tx: np.ndarray, n_splits: int = 5) -> tuple:
    """
     Demodulates using decision trees with k-fold cross-validation.

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
    def algorithm_func(X_train, y_train, X_test):
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        return model.predict(X_test)

    X = X_rx
    y = sym_tx

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return kfold_cross_validation(X, y, n_splits, algorithm_func)
