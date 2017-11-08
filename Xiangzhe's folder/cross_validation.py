import numpy as np
from costs import *
from proj1_helpers import *
from tools import *
from implementations import *


def cross_validation(y, x, k_indices, k, regression_method, **kwargs):
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)

    y_test = y[test_indice]
    y_train = y[train_indice]
    x_test = x[test_indice]
    x_train = x[train_indice]

    #data processing
    x_train, x_test = process_data(x_train, x_test)
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    weight, loss_train = regression_method(y = y_train, tx = x_train, **kwargs)

    y_train_pred = predict_labels(weight, x_train)
    y_test_pred = predict_labels(weight, x_test)

    accuracy_train = compute_accuracy(y_train_pred, y_train)
    accuracy_test = compute_accuracy(y_test_pred, y_test)

    return accuracy_train, accuracy_test


def cross_validation_ridge_regression(y, x, k_indices, k, lambdas, degrees):
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)

    all_y_test = y[test_indice]
    all_y_train = y[train_indice]
    all_x_test = x[test_indice]
    all_x_train = x[train_indice]

    # split the training set into 3 subsets
    dict_jets_train = group_features_by_jet(all_x_train)
    dict_jets_test = group_features_by_jet(all_x_test)

    y_train_pred = np.zeros(len(all_y_train))
    y_test_pred = np.zeros(len(all_y_test))

    for index in range(len(dict_jets_train)):
        x_train = all_x_train[dict_jets_train[index]]
        x_test = all_x_test[dict_jets_test[index]]
        y_train = all_y_train[dict_jets_train[index]]

        #data processing
        x_train, x_test = process_data(x_train, x_test)
        x_train = build_polynomial_features(x_train, degrees[index])
        x_test = build_polynomial_features(x_test, degrees[index])
        x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
        x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

        weight, loss_train = ridge_regression(y = y_train, tx = x_train, lambda_ = lambdas[index])

        y_train_pred[dict_jets_train[index]] = predict_labels(weight, x_train)
        y_test_pred[dict_jets_test[index]] = predict_labels(weight, x_test)

    accuracy_train = compute_accuracy(y_train_pred, all_y_train)
    accuracy_test = compute_accuracy(y_test_pred, all_y_test)

    return accuracy_train, accuracy_test
