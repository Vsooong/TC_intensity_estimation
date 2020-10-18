from data.label_match import initDataFromDictionary
from utils.Utils import args
import os
from visualize.get_model_result import build_one_ty, estimate_one_ty, get_model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from visualize.vis_samples import plot_error_with_value, flatten

sns.set_theme(color_codes=True)
sns.set_style("whitegrid")


def bias(label, pred):
    assert len(label) == len(pred)
    diff = np.asarray(pred) - np.asarray(label)
    return diff


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def cls_intensity(ints):
    if ints <= 17.2:
        w = 'TD'
    elif ints <= 24.4:
        w = 'TS'
    elif ints <= 32.7:
        w = 'STS'
    elif ints <= 41.4:
        w = 'TY'
    elif ints <= 51:
        w = 'STY'
    else:
        w = 'SuperTY'
    return w


def error_for_categories(label=None, estimation=None,test_years=None):
    if label is None or estimation is None:
        label = np.load('data/label_ints.npy', allow_pickle=True).item()
        estimation = np.load('data/estim_ints.npy', allow_pickle=True).item()
    if test_years is not None:
        new_label={}
        new_estim={}
        for k,v in label.items():
            if int(k[:4]) in test_years:
                new_label[k]=v
        for k,v in estimation.items():
            if int(k[:4]) in test_years:
                new_estim[k]=v
        label=new_label
        estimation=new_estim
    label = flatten(label,True)
    estimation = flatten(estimation,True)
    mae = mean_absolute_error(label, estimation)
    rmse = np.sqrt(mean_squared_error(label, estimation))
    diff = bias(label, estimation)
    bia = np.sum(diff) / len(label)
    mape = mean_absolute_percentage_error(label, estimation)
    print(rmse,mae,bia,mape)

    metric_table = {}
    for index in range(len(label)):
        lb = label[index]
        pd = estimation[index]
        cate = cls_intensity(lb)
        if cate not in metric_table.keys():
            metric_table[cate] = [[lb, pd]]
        else:
            metric_table[cate].append([lb, pd])
    for key in metric_table.keys():
        lbs, pds = zip(*metric_table[key])
        mae = mean_absolute_error(lbs, pds)
        rmse = np.sqrt(mean_squared_error(lbs, pds))
        diff = bias(lbs, pds)
        bia = np.sum(diff) / len(lbs)
        mape = mean_absolute_percentage_error(lbs, pds)
        median=np.median(diff)
        print(key,len(lbs), rmse, mae, bia,median, mape)

def linear_fit(label=None, estimation=None):
    if label is None or estimation is None:
        label = np.load('data/label_ints.npy', allow_pickle=True).item()
        estimation = np.load('data/estim_ints.npy', allow_pickle=True).item()
    label = flatten(label)
    estimation = flatten(estimation)
    reg = linear_model.LinearRegression()
    reg.fit(X=label.reshape(-1, 1), y=estimation.reshape(-1, 1))
    print(reg.coef_)
    print(reg.intercept_)
    print(r2_score(label, estimation))


def plot_box(label=None, estimation=None):
    if label is None or estimation is None:
        label = np.load('data/label_ints.npy', allow_pickle=True).item()
        estimation = np.load('data/estim_ints.npy', allow_pickle=True).item()
    label = flatten(label, use_interp=False)
    estimation = flatten(estimation, use_interp=False)
    index = [cls_intensity(i) for i in label]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=index, y=bias(label, estimation),showmeans=True,meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"8"})
    ax.tick_params(axis='y', labelsize=20)  # y轴
    ax.tick_params(axis='x', labelsize=20)  # y轴
    plt.show()


def plot_curve(label=None, estimation=None):
    if label is None or estimation is None:
        label = np.load('data/label_ints.npy', allow_pickle=True).item()
        estimation = np.load('data/estim_ints.npy', allow_pickle=True).item()
    label = flatten(label, True)
    estimation = flatten(estimation, use_interp=True)
    print(max(label), min(estimation))
    plot_error_with_value(label, estimation, 10, 75, 5, ymin=-20, ymax=20)


def plot_scattor(label=None, estimation=None, order=1):
    if label is None or estimation is None:
        label = np.load('data/label_ints.npy', allow_pickle=True).item()
        estimation = np.load('data/estim_ints.npy', allow_pickle=True).item()
    label = flatten(label, use_interp=True)
    estimation = flatten(estimation, use_interp=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='y', labelsize=20)  # y轴
    ax.tick_params(axis='x', labelsize=20)  # y轴
    sns.regplot(x=label, y=estimation, order=order, marker=".")
    plt.ylim(5, 75)
    plt.xlim(5, 75)
    plt.show()


def get_estimation(data_file='F:/data/TC_IR_IMAGE/', which=1):
    labels = {}
    predictions = {}
    tys = initDataFromDictionary(data_file, args.train_years)
    model = get_model(which=which)
    for path in tys:
        basename = os.path.basename(path)
        number = basename.split('_')[0]
        print(path)
        if which == 1:
            X_im, X_ef, X_sst, target, times = build_one_ty(path, split=True)
        else:
            X_im, X_ef, X_sst, target, times = build_one_ty(path, split=False)
            target = target[0]
        pred, f_div_C, W_y = estimate_one_ty(X_im, X_ef, X_sst, model)
        pred = pred.squeeze().cpu().detach().numpy()
        target = target.squeeze().cpu().detach().numpy()
        assert len(pred) == len(target)
        # print(pred.shape)
        # print(target.shape)
        labels[number] = target
        predictions[number] = pred
        # print(predictions)
        # print(labels)

    np.save('data/label_ints.npy', labels, allow_pickle=True)
    np.save('data/estim_ints.npy', predictions, allow_pickle=True)
    return labels, predictions
    # labels = flatten(labels)
    # print(len(labels))
    # return labels


if __name__ == '__main__':
    # labels, predictions = get_estimation(which=2)
    # plot_scattor()
    # linear_fit()
    # plot_curve()
    # plot_box()
    error_for_categories(test_years=[2016,2017,2018,2019])

    pass
