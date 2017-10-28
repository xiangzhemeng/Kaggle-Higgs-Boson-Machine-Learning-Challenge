# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def replace_nan_with_frequent(x):
    """replace nan values with the most frequent values within the column"""
    for i in range(x.shape[1]):
        if np.any(x[:, i] == -999):
            tmp = (x[:, i] != -999)
            values, counts = np.unique(x[tmp, i], return_counts = True)
            if (len(values) > 1):
                x[~tmp, i] = values[np.argmax(counts)]
            else:
                x[~tmp, i] = 0
    return x

def convert_categorical(x, column_num):
    """convert a column to one-hot representation"""
    arr = x[:,column_num]

    cat_reference = dict()
    cat_unique = np.unique(arr)

    for i, cat in enumerate(cat_unique):
        code =  len(cat_unique) * [0]
        code[i] = 1
        cat_reference[cat] = code


    cat_columns = list()
    for cat in arr:
        cat_columns.append(cat_reference[cat])

    x = np.delete(x, (column_num), axis=1)

    return x, cat_columns

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def poly_features(x, degree):
    """degrees of existing features"""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]

    return poly[:,1:]

def cross_column_features(x, num_features=0):
    """cross-column features created by combining existing ones"""

    # 2s combination of columns
    if not num_features:
        num_features = x.shape[1]

    double_features = list()
    for i in range(num_features):
        # print("Feature(%d/%d)" % (i,num_features-1)) #DEBUG
        for j in range(i+1, num_features):
            tmp = x[:,i] * x[:,j]
            double_features.append(tmp)

    double_features = np.array(double_features).T

    return np.c_[x, double_features]

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def standardize_columns(x):
    """standardize all the columns of given dataset"""
    num_samples = len(x[:,0])
    num_features = len(x[0,:])
    tx = np.zeros(num_samples)
    for i in range(num_features):
        # print("Standardize(%d/%d)"%(i,num_features-1)) #DEBUG
        tmp,_,_ = standardize(x[:,i])
        tx = np.c_[tx, tmp]

    return tx[:,1:]

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    # yb[np.where(y=='b')] = -1
    yb[np.where(y=='b')] = 0

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, pos=1, neg=-1):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = neg
    y_pred[np.where(y_pred > 0)] = pos

    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
