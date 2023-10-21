# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape (D,).

    """
    N, D = tx.shape
    # Compute the inner terms of the formula
    XTX = tx.T @ tx
    lambda_I = lambda_ * np.identity(D)
    A = XTX + 2 * N * lambda_I 
    b = tx.T @ y

    # Solve for the optimal weights w using the formula
    w = np.linalg.solve(A, b)

    return w
