# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import *
from tools import *
from implementations import *

print('Start loading data...\n')

DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'

y_train, tx_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
_, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Generate empty output prediction matrix
y_pred = np.zeros(tx_test.shape[0])

# Split data into three subset by jet_no
dict_jets_train = group_features_by_jet(tx_train)
dict_jets_test = group_features_by_jet(tx_test)


# Set individual parameter, lambda and polynomial degree  for each subset
lambdas = [0.0001, 0.0001, 0.0001]
degrees =  [12, 13, 10]

print('Start training...\n')

for index in range(len(dict_jets_train)):
    x_train = tx_train[dict_jets_train[index]]
    x_test = tx_test[dict_jets_test[index]]
    y_train = y[dict_jets_train[index]]

    # data processing
    x_train, x_test = process_data(x_train, x_test)
    x_train = build_polynomial_features(x_train, degrees[index])
    x_test = build_polynomial_features(x_test, degrees[index])
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    weight, loss_train = ridge_regression(y_train, x_train, lambdas[index])

    temp_test_pred = predict_labels(weight, x_test)

    y_pred[dict_jets_test[index]] = temp_test_pred

print('Start generating prediction files...\n')

OUTPUT_PATH = 'data/output_ridge_regression3.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print('Finish!')