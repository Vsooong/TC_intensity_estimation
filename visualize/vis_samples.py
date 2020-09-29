import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas
import seaborn as sns;
from Plots.data_fitting import flatten,bias
from matplotlib import pyplot as plt

def cate_intensity(ints):
    w = np.zeros_like(ints)
    for index in range(len(ints)):
        if  ints[index]>=17.2 and ints[index]<=24.4:
            w[index]=1
        elif ints[index]>=32.7 and ints[index]<=41.4:
            w[index] = 1
        elif ints[index]>=51:
            w[index] = 1
    return w

def plot_hist():
    lbs = np.load('F:/Python_Project/TC_intensity_prediction/Plots/label_ints.npy', allow_pickle=True).item()
    labels = flatten(lbs)
    labels=np.repeat(labels, 2)
    a4_dims = (8, 6)
    sns.set_style("whitegrid")
    min_value, max_value=min(labels),max(labels)
    print(min_value,max_value)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax.tick_params(axis='y', labelsize=20)  # y轴
    ax.tick_params(axis='x', labelsize=20)  # y轴
    ymin,ymax=0,9000
    plt.ylim(ymin, ymax)
    plt.xlim(10, 75)
    # sns.histplot( x=labels,bins=np.arange(35,125,5), kde=True)
    sns.histplot( x=labels,bins=np.arange(10,80,5), kde=False,element="step", fill=False)
    t = np.arange(int(min_value), int(max_value) + 1, 1)
    w=cate_intensity(t)
    print(w)
    ax.fill_between(t, y1=ymin, y2=ymax, where=w,
                    color='grey', alpha=0.3)
    plt.show()


def plot_error_with_value(labels, preds,min_value, max_value, intv):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    x = np.arange(min_value, max_value + 1, intv)
    print(x)
    loss_dict = {}
    metric_dict = {}
    for i in x:
        loss_dict[i] = []

    for index in range(len(labels)):
        lb = labels[index]
        pd = preds[index]
        inv = int(lb / intv) * intv
        index=x[np.where(x<=inv)][-1]
        # print(inv)
        loss_dict[index].append([lb, pd])
    for k, v in loss_dict.items():
        if len(v) > 0:
            lbs, pds = zip(*v)
            mae = mean_absolute_error(lbs, pds)
            rmse = np.sqrt(mean_squared_error(lbs, pds))
            # r = np.corrcoef(lbs, pds)[0][1]
            bia=bias(lbs,pds)
            metric_dict[k] = [mae, rmse,bia]
    mae_list = []
    mse_list = []
    bia_list=[]
    factor = 3.152
    for i in x:
        if i in metric_dict.keys():
            mae_list.append(metric_dict[i][0])
            mse_list.append(metric_dict[i][1])
            bia_list.append(metric_dict[i][2])
        else:
            mae_list.append(0)
            mse_list.append(0)
            bia_list.append(0)

    # d = {'Best Track': x, 'MAE': mae_list, 'RMSE': mse_list, 'R': r_list,'Bias':bia_list}
    d = {'Best Track': x, 'MAE': mae_list, 'RMSE': mse_list,'Bias':bia_list}

    data = pandas.DataFrame(data=d)
    import seaborn as sns
    a4_dims = (8, 6)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=a4_dims)
    ax.tick_params(axis='y', labelsize=20)  # y轴
    ax.tick_params(axis='x', labelsize=20)  # y轴
    sns.lineplot(x='Best Track', y='value', hue='variable', data=pandas.melt(data, ['Best Track']))
    ymin,ymax=-0.5, 6.5
    plt.ylim(ymin,ymax)
    plt.xlim(min_value,max_value)
    t=np.arange(min_value,max_value+1,1)
    w=np.zeros_like(t)
    w[2:5]=1
    ax.fill_between(t,y1=ymin,y2=ymax,where=w,
                    color='green',alpha=0.1)
    plt.show()
def plot_error():
    lbs = np.load('F:/Python_Project/TC_intensity_prediction/Plots/label_ints.npy', allow_pickle=True).item()
    pred = np.load('F:/Python_Project/TC_intensity_prediction/Plots/estimate_ints.npy', allow_pickle=True).item()
    labels = flatten(lbs)
    predicts = flatten(pred)
    print(max(labels), min(labels))
    plot_error_with_value(labels, predicts, 32, 126, 2)



if __name__ == '__main__':
    import xarray
    # data=xarray.open_dataset('F:/data/msc/relative-humidity.nc',decode_times=True)
    # print(data)
    plot_hist()
    # t = np.arange(18, 65, 1)
    # print(cate_intensity(t))

