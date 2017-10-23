import numpy as np


def compute_accuracy(y_pred, y_real):
    count = 0
    for idx, value in enumerate(y_real):
        if value == y_pred[idx]:
            count += 1
    return count / len(y_real)


def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def standardize(x, mean = None, std = None):
    if mean is None:
        mean = np.mean(x, axis=0)
    x = x - mean
    if std is None:
        std = np.std(x, axis=0)
    x[:,std > 0] = x[:,std > 0] / std[std > 0]
    return x, mean, std

def build_polynomial_features(x, degree):
    temp_dict = {}
    count = 0
    for i in range(x.shape[1]):
        for j in range(i+1,x.shape[1]):
            temp = x[:,i] * x[:,j]
            temp_dict[count] = [temp]
            count += 1
    poly_length = x.shape[1] * (degree + 1) + count + 1
    poly = np.zeros(shape = (x.shape[0], poly_length))
    for deg in range(1,degree+1):
        for i in range(x.shape[1]):
            poly[:,i + (deg-1) * x.shape[1]] = np.power(x[:,i],deg)
    for i in range(count):
        poly[:, x.shape[1] * degree + i] = temp_dict[i][0]
    for i in range(x.shape[1]):
        poly[:,i + x.shape[1] * degree + count] = np.abs(x[:,i])**0.5
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


def group_features_by_jet(x):
    return {  0: x[:, 22] == 0,
              1: x[:, 22] == 1,
              2: np.logical_or(x[:, 22] == 2, x[:, 22] == 3)  }


def process_data(x_train, x_test):
    x_train, x_test = replace_missing_data_by_frequent_value(x_train, x_test)

    log_cols_index = [0, 1, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]
    x_train_log_cols_index = np.log(1 / (1 + x_train[:, log_cols_index]))
    x_train = np.hstack((x_train, x_train_log_cols_index))
    x_test_log_cols_index = np.log(1 / (1 + x_test[:, log_cols_index]))
    x_test = np.hstack((x_test, x_test_log_cols_index))

    x_train, mean_train, std_train = standardize(x_train)
    x_test, mean_test, std_test = standardize(x_test, mean_train, std_train)

    x_train = np.delete(x_train, [15,18,20,25,28], 1)
    x_test = np.delete(x_test, [15,18,20,25,28], 1)

    return x_train, x_test
