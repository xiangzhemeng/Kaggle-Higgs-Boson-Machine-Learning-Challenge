import numpy as np

def sigmoid(t):
    return np.exp(-np.logaddexp(0, -t))

def calculate_mse(e):
    return 1/2 * np.mean(e**2)

def calculate_mae(e):
    return np.mean(np.abs(e))

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_loss_neg_log_likelihood(y, tx, w):
    epsilon = 1e-12
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred + epsilon)) + (1 - y).T.dot(np.log(1 - pred + epsilon))
    return np.squeeze(- loss)
