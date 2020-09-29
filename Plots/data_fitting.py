import numpy as np
import os
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error


def bias(label, pred):
    assert len(label) == len(pred)
    diff = np.asarray(pred) - np.asarray(label)
    return np.sum(diff) / len(label)


def generate_estimation(mean, sig):
    global estimate
    estimate = {}
    mu, sigma = mean, sig
    read_dictionary = np.load('label_ints.npy', allow_pickle=True).item()
    np.set_printoptions(precision=3)
    for k, v in read_dictionary.items():
        length = len(v)
        s = np.random.normal(mu, sigma, length)
        s = np.around(s, decimals=3)
        estimate_ints = np.asarray(v) + s
        estimate[k] = estimate_ints
    np.save('estimate_ints.npy', estimate, allow_pickle=True)
    return estimate


# generate_estimation()
def get_metrics(labels, predict):
    mse = mean_squared_error(labels, predict)
    mae = mean_absolute_error(labels, predict)
    return np.sqrt(mse), mae, bias(labels, predict)


def metrics_every_year(labels, predict, years=2016):
    _result1 = []
    _result2 = []
    for k, v in labels.items():
        if int(k[:4]) == years:
            _result1.append(v)
            _result2.append(predict[k])
    res1 = np.concatenate(_result1, axis=0)
    res2 = np.concatenate(_result2, axis=0)
    return get_metrics(res1, res2)


def flatten(dict):
    res = []
    for k, v in dict.items():
        res.append(v)
    return np.concatenate(res, axis=0)


if __name__ == '__main__':
    np.random.seed(4)
    # pred = np.load('label_ints.npy', allow_pickle=True).item()
    # print(len(flatten(pred)))
    # pred = generate_estimation(0.5, 4.86)
    # lbs = np.load('label_ints.npy', allow_pickle=True).item()
    # years=[2016,2017,2018,2019]
    # for y in years:
    #     print(metrics_every_year(lbs,pred,y))
    # print('---------------------seperate line----------------------')
    # predict = flatten(pred)
    # best_track = flatten(lbs)
    # print(get_metrics(best_track, predict))
