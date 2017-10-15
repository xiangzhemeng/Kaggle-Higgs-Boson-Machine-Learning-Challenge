# native libraries
import numpy as np
from helpers import *

# load and organize data
path_dataset = "exerciseData/height_weight_genders.csv"
data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
height = data[:, 0]
weight = data[:, 1]
gender = np.genfromtxt(
    path_dataset, delimiter=",", skip_header=1, usecols=[0],
    converters={0: lambda x: 0 if b"Male" in x else 1})
# Convert to metric system
height *= 0.025
weight *= 0.454

x1, mean_x1, std_x2 = standardize(height)
x2, mean_x2, std_x2 = standardize(weight)
y, tx = build_model_data(x1, x2, gender)

m = num_samples = len(y)
tx = np.c_[np.ones(m), x1, x2]

# required functions
def calculate_mse(y, tx, w):
    """mean square error"""
    err = y - tx.dot(w)
    return 1/2*np.mean(err**2)

def compute_gradient(y, tx, w):
    """gradient computation for linear regression"""
    """(x transpose times w) is linear predictor"""
    err = tx.dot(w) - y
    grad = tx.T.dot(err) / len(err)
    return grad, err

# main functions
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """linear regression using gradient descent"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient, loss
        grad, _ = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        # calculate loss
        loss = calculate_mse(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD({bi}/{ti}): loss={l}, weights={},{},{}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, *w.round(5)))

    return losses, ws

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """linear regression using stochastic gradient descent"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = calculate_mse(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, weights={},{},{}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, *w.round(5)))

    return losses, ws

def least_squares(y, tx):
    """least squares regression using normal equations"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = calculate_mse(y, tx, w)
    return loss, w

# required functions
def ridge_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

# main function
def ridge_regression(y, tx, lambda_):
    """rige regression using normal equations"""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = ridge_mse(y, tx, w)
    return loss, w

# required function(s)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_cost(y, tx, w):
    """cost for logistic regression """
    sig = sigmoid(tx.dot(w));
    cost = (-y) * np.log(sig) - (1-y) * np.log(1-sig)
    return np.mean(cost)

def logistic_gradient(y, tx, w):
    """gradient for logistic regression """
    err = sigmoid(tx.dot(w)) - y
    grad = tx.T.dot(err) / len(err)
    return grad, err

# main function(s)
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """logistic regression using GD"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient, loss
        grad, _ = logistic_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        # calculate loss
        loss = logistic_cost(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD({bi}/{ti}): loss={l}, weights={},{},{}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, *w.round(5)))

    return losses, ws

def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """linear regression using stochastic SGD"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = logistic_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = logistic_cost(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, weights={},{},{}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, *w.round(5)))
    return losses, ws

# required function(s)
def reg_logistic_cost(y, tx, w, alpha):
    """cost for logistic regression with regularization"""
    sig = sigmoid(tx.dot(w));
    cost = (-y) * np.log(sig) - (1-y) * np.log(1-sig)
    reg = np.dot(w,w) * alpha / (2 * len(y))
    return np.mean(cost) + reg

def reg_logistic_gradient(y, tx, w, alpha):
    """gradient for logistic regression with with regularization"""
    err = sigmoid(tx.dot(w)) - y
    grad = tx.T.dot(err) / len(err)
    reg = w * alpha / len(err)
    return grad - reg, err

# main function(s)
def reg_logistic_regression(y, tx, alpha, initial_w, max_iters, gamma):
    """regularized logistic regression using GD"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient, loss
        grad, _ = reg_logistic_gradient(y, tx, w, alpha)
        # gradient w by descent update
        w = w - gamma * grad
        # calculate loss
        loss = reg_logistic_cost(y, tx, w, alpha)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD({bi}/{ti}): loss={l}, weights={},{},{}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, *w.round(5)))

    return losses, ws

def reg_logistic_regression_SGD(y, tx, alpha, initial_w, max_iters, gamma, batch_size=1):
    """regularized logistic regression using SGD"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = reg_logistic_gradient(y_batch, tx_batch, w, alpha)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = reg_logistic_cost(y, tx, w, alpha)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, weights={},{},{}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, *w.round(5)))
    return losses, ws
