from data.label_match import initDataFromDictionary
from utils.Utils import args
import os
from visualize.get_model_result import build_one_ty, estimate_one_ty, get_model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from visualize.vis_samples import plot_error_with_value,flatten,bias
sns.set_theme(color_codes=True)
sns.set_style("whitegrid")


def linear_fit(label=None, estimation=None):
    if label is None or estimation is None:
        label = np.load('data/label_ints.npy', allow_pickle=True).item()
        estimation = np.load('data/estim_ints.npy', allow_pickle=True).item()
    label = flatten(label)
    estimation= flatten(estimation)
    reg = linear_model.LinearRegression()
    reg.fit(X=label.reshape(-1,1),y=estimation.reshape(-1,1))
    print(reg.coef_)
    print(reg.intercept_)
    print(r2_score(label, estimation))

def plot_curve(label=None, estimation=None):
    if label is None or estimation is None:
        label = np.load('data/label_ints.npy', allow_pickle=True).item()
        estimation = np.load('data/estim_ints.npy', allow_pickle=True).item()
    label = flatten(label)
    estimation = flatten(estimation)
    print(max(label), min(estimation))
    plot_error_with_value(label, estimation, 10, 75, 5,ymin=-20,ymax=20)


def plot_scattor(label=None, estimation=None,order=1):
    if label is None or estimation is None:
        label = np.load('data/label_ints.npy', allow_pickle=True).item()
        estimation = np.load('data/estim_ints.npy', allow_pickle=True).item()
    label = flatten(label)
    estimation= flatten(estimation)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='y', labelsize=20)  # y轴
    ax.tick_params(axis='x', labelsize=20)  # y轴
    sns.regplot(x=label, y=estimation,order=order,marker=".")
    plt.ylim(5, 75)
    plt.xlim(5, 75)
    plt.show()


def get_estimation(data_file='F:/data/TC_IR_IMAGE/'):
    labels = {}
    predictions = {}
    tys = initDataFromDictionary(data_file, args.train_years)
    model = get_model(1)
    for path in tys:
        basename = os.path.basename(path)
        number = basename.split('_')[0]
        print(path)
        X_im, X_ef, X_sst, target, times = build_one_ty(path, split=True)
        pred, f_div_C, W_y = estimate_one_ty(X_im, X_ef, X_sst, model)
        pred = pred.squeeze(-1).cpu().detach().numpy()
        target = target.squeeze(-1).cpu().detach().numpy()
        assert len(pred) == len(target)
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
    # labels, predictions = get_estimation()
    # plot_scattor()
    # linear_fit()
    plot_curve()
    pass
