import numpy as np


def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def standardize(x, mean_x=None, std_x=None):
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]
    return x, mean_x, std_x


def build_poly(x, degree):
    poly = np.ones((len(x), 1))
    for deg in range(1,degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def batch_iter(y, tx, batch_size, num_batches = None, shuffle = True):
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def replace_missing_data_by_frequent_value(x_train, x_test):
    for i in range(x_train.shape[1]):
        if np.any(x_train[:, i] == -999):
            temp_train = (x_train[:, i] != -999) #return a list of true or false
            temp_test = (x_test[:, i] != -999)
            values, counts = np.unique(x_train[temp_train, i], return_counts = True)
            if (len(values) > 1):
                x_train[~temp_train, i] = values[np.argmax(counts)]
                x_test[~temp_test, i] = values[np.argmax(counts)]
            else:
                x_train[~temp_train, i] = 0
                x_test[~temp_test, i] = 0
    return x_train, x_test


def process_data(x_train, x_test):
    x_train, x_test = replace_missing_data_by_frequent_value(x_train, x_test)

    log_cols_index = [0, 1, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]

    # Create inverse log values of features which are positive in value.
    x_train_log_cols_index = np.log(1 / (1 + x_train[:, log_cols_index]))
    x_train = np.hstack((x_train, x_train_log_cols_index))

    x_test_log_cols_index = np.log(1 / (1 + x_test[:, log_cols_index]))
    x_test = np.hstack((x_test, x_test_log_cols_index))

    x_train, mean_x_train, std_x_train = standardize(x_train)
    x_test, mean_x_test, std_x_test = standardize(x_test, mean_x_train, std_x_train)

    return x_train, x_test


def process_data_ridge_regression(x_train, x_test,jet_index):
    #After discarding all the NaN value colums, there are still some NaN values in serveral columns
    x_train, x_test = replace_missing_data_by_frequent_value(x_train, x_test)

    #choose positive columns to log
    if jet_index == 0:
        log_cols_index = [0, 1, 2, 4, 5, 6, 8, 11, 14, 16]
    elif jet_index == 1:
        log_cols_index = [0, 1, 2, 4, 6, 7, 9, 12, 15, 17, 18]
    else: #jet_index == 2 or 3
        log_cols_index = [0, 1, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]

    # Create inverse log values of features which are positive in value.
    x_train_log_cols_index = np.log(1/(1 + x_train[:, log_cols_index]))
    x_train = np.hstack((x_train, x_train_log_cols_index))

    x_test_log_cols_index = np.log(1/(1 + x_test[:, log_cols_index]))
    x_test = np.hstack((x_test, x_test_log_cols_index))

    x_train, mean_x_train, std_x_train = standardize(x_train)
    x_test, mean_x_test, std_x_test = standardize(x_test, mean_x_train, std_x_train)

    return x_train, x_test
