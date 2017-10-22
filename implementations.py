import numpy as np
from costs import *
from tools import *


"""Linear Regression Gradient Descend using MSE"""

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent"""
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss and gradient
        loss = compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w)
        # update w by gradient
        w -= gamma * grad

    return w, loss



"""Linear Regression Stochastic Gradient Descend using MSE"""

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent"""
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    loss = 0
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size = batch_size, num_batches = 1):
            # compute stochastic loss and gradient
            loss = compute_loss(y, tx, w)
            grad = compute_gradient(y_batch, tx_batch, w)
            # update w by stochastic gradient
            w -= gamma * grad

    return w, loss


"""Least Squares"""


def least_squares(y, tx):
    """ Least squares regression using normal equations"""
    m = tx.T.dot(tx)
    n = tx.T.dot(y)
    w = np.linalg.solve(m, n) #sovle: mX=n ==> X
    loss = compute_loss(y, tx, w)
    return w, loss


"""Ridge Regression"""


def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations"""
    regularizer = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + regularizer
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


"""Logistic Regression"""

def learning_by_gradient_descent(y, tx, w, gamma):
    loss = compute_loss_neg_log_likelihood(y, tx, w)
    grad = tx.T.dot(sigmoid(tx.dot(w)) - y)
    w -= gamma * grad
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):

    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    w = initial_w
    losses = []
    threshold = 1e-8

    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)

        # Stop criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]


"""Regularized Logistic Regression"""


def penalized_logistic_regression(y, tx, w, lambda_):
    loss = compute_loss_neg_log_likelihood(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = tx.T.dot(sigmoid(tx.dot(w)) - y) + 2 * lambda_ * w

    return loss, gradient


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    w = initial_w
    losses = []
    threshold = 1e-8


    for n_iter in range(max_iters):
        # get loss and update w.
        loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient
        losses.append(loss)

        # Stop criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]


