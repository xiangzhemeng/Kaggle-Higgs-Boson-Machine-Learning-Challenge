import numpy as np
from costs import *
from tools import *


"""least_squares_gd"""

def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    # Set default weight
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    loss = 0
    weight = initial_w

    for i in range(max_iters):
        loss = compute_loss(y, tx, weight)
        gradient = compute_gradient(y, tx, weight)
        weight -= gamma * gradient
    return weight, loss


"""least_squares_sgd"""

def least_squares_sgd(y, tx, initial_w, max_iters, gamma):

    # Set default weight
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    loss = 0
    weight = initial_w
    batch_size = 1

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size = batch_size, num_batches = 1):
            loss = compute_loss(y, tx, weight)
            gradient = compute_gradient(y_batch, tx_batch, weight)
            weight -= gamma * gradient
    return weight, loss


"""least squares"""

def least_squares(y, tx):
    m = tx.T.dot(tx)
    n = tx.T.dot(y)

    # solve: mX=n --> X
    weight = np.linalg.solve(m, n)
    loss = compute_loss(y, tx, weight)
    return weight, loss


"""ridge regression"""

def ridge_regression(y, tx, lambda_):

    reg = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    m = tx.T.dot(tx) + reg
    n = tx.T.dot(y)
    # solve: mX=n --> X
    weight = np.linalg.solve(m, n)
    loss = compute_loss(y, tx, weight)
    return weight, loss


"""logistic regression"""

def learning_by_gradient_descent(y, tx, w, gamma):
    loss = compute_loss_neg_log_likelihood(y, tx, w)
    gradient = tx.T.dot(sigmoid(tx.dot(w)) - y)
    w -= gamma * gradient
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):

    # Set default weight
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])


    weight = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):

        # get loss and update w.
        weight, loss = learning_by_gradient_descent(y, tx, weight, gamma)
        losses.append(loss)

        # termination condition
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return weight, losses[-1]


"""Regularized Logistic Regression"""

def penalized_logistic_regression(y, tx, w, lambda_):
    loss = compute_loss_neg_log_likelihood(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = tx.T.dot(sigmoid(tx.dot(w)) - y) + 2 * lambda_ * w

    return loss, gradient

def regularized_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    # Set default weight
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    weight = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):

        # get loss and update w.
        loss, gradient = penalized_logistic_regression(y, tx, weight, lambda_)
        weight = weight - gamma * gradient
        losses.append(loss)

        # termination condition
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return weight, losses[-1]
