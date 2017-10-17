import numpy as np
#from costs import *
#from proj1_helpers import *
from tools import *
#from implementations import *


def cross_validation(y, x, k_indices, k, regression_method, **kwargs):
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)

    y_test = y[test_indice]
    y_train = y[train_indice]
    x_test = x[test_indice]
    x_train = x[train_indice]

    x_train, x_test = process_data(x_train, x_test)
    x_train = build_poly(x_train, 0)
    x_test = build_poly(x_test, 0)

    weights, loss_train = regression_method(y = y_train, tx = x_train, **kwargs)
    y_test_pred = predict_labels(weights, x_test)
    loss_test = compute_loss(y_test, x_test, weights)

    return loss_train, loss_test


#create a jet dictionary
def group_features_by_jet(x):
    return {  0: x[:, 22] == 0,
              1: x[:, 22] == 1,
              2: np.logical_or(x[:, 22] == 2, x[:, 22] == 3)  }


#split trianing set into 3 subsets
#for each subset, we discard the NaN value and unuseful(due to the correlation) columns
def cross_validation_ridge_regression(y, x, k_indices, k, lambda_, degree):
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)

    y_test_all_jets = y[test_indice]
    y_train_all_jets = y[train_indice]
    x_test_all_jets = x[test_indice]
    x_train_all_jets = x[train_indice]

    # split the training set into 3 subsets
    dict_jets_train = group_features_by_jet(x_train_all_jets)
    dict_jets_test = group_features_by_jet(x_test_all_jets)

    losses_train = []
    losses_test = []

    for idx in range(len(dict_jets_train)):
        x_train = x_train_all_jets[dict_jets_train[idx]]
        x_test = x_test_all_jets[dict_jets_test[idx]]
        y_train = y_train_all_jets[dict_jets_train[idx]]

        #correlation --> delete columns
        if idx == 0:
            x_train = np.delete(x_train, [4,5,6,8,12,22,23,24,25,26,27,28], 1)
            x_test = np.delete(x_test, [4,5,6,8,12,22,23,24,25,26,27,28], 1)
        elif idx == 1:
            x_train = np.delete(x_train, [4,5,6,12,22,26,27,28,29], 1)
            x_test = np.delete(x_test, [4,5,6,12,22,26,27,28,29], 1)

        #data processing
        x_train, x_test = process_data_ridge_regression(x_train, x_test,idx)

        tX_train = build_poly(x_train, degree[idx])
        tX_test = build_poly(x_test, degree[idx])

        #compute weights using given method
        weights, loss = ridge_regression(y = y_train, tx = tX_train, lambda_ = lambda_[idx])

        loss_train = compute_loss(y_train_all_jets[dict_jets_train[idx]], tX_train, weights)
        losses_train.append(loss_train)
        loss_test = compute_loss(y_test_all_jets[dict_jets_test[idx]], tX_test, weights)
        losses_test.append(loss_test)

    return np.mean(losses_train), np.mean(losses_test)
