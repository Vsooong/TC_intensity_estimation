import math
import pandas
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from visualize.get_model_result import build_one_ty, get_model
import os
from utils.Utils import args
from global_land_mask import globe
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import torch

register_matplotlib_converters()
sns.set_style("whitegrid")


def build_att(atts):
    attentions = []
    for layer_att in atts:
        attention = {'Modality': [], 'Time step': [], 'Avg attention': []}
        rows, cols = np.shape(layer_att)
        for i in range(rows):
            for j in range(cols):
                attention['Modality'].append(i + 1)
                attention['Time step'].append(j - cols + 1)
                attention['Avg attention'].append(layer_att[i][j])
        attention = pandas.DataFrame(attention)
        attentions.append(attention)

    return attentions


def plot_attentin(attentions):
    nlayers = len(attentions)
    sns.set()
    atts = build_att(attentions)
    fig, axs = plt.subplots(nlayers, sharey=True)
    for i in range(nlayers):
        att = atts[i]
        data = att.pivot('Modality', 'Time step', 'Avg attention')
        xticklabels = True if i == nlayers - 1 else False

        # Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r

        sns.heatmap(data, ax=axs[i], cbar=False, robust=True, cmap="Greys_r",
                    yticklabels=True, xticklabels=xticklabels)
        axs[i].set(ylabel='Modalities')
    #
    # sns.heatmap(flights, ax=axs[0], cmap='coolwarm', cbar=False, robust=True, yticklabels=False, xticklabels=False)
    # axs[0].set(ylabel='layer1')
    # sns.heatmap(flights, ax=axs[1], cbar=False, yticklabels=False, xticklabels=False)
    # axs[1].set(ylabel='layer5')
    # sns.heatmap(flights, ax=axs[2], cbar=False, yticklabels=False)
    # axs[2].set(ylabel='layer10')

    for ax in axs.flat:
        ax.label_outer()
    plt.show()


def min_times_number(a, b):
    return a * b / math.gcd(a, b)


def is_on_land(points):
    leng = len(points)
    w = []
    for index in range(leng):
        [lat, lon] = points[index]
        is_on = globe.is_land(lat, lon)
        if is_on:
            w.append(True)
        else:
            w.append(False)
    return w


def is_rapid_intensification(intensities):
    w = []
    for index, i in enumerate(intensities):
        preindex = index - 8
        if preindex < 0:
            w.append(False)
        elif intensities[index] - intensities[preindex] >= 30 * 0.5144:
            w.append(True)
        else:
            w.append(False)
    return w


def plot_track(X_ef, target, pred, times):
    target = target.squeeze().cpu().detach().numpy() * 10
    pred = pred.squeeze().cpu().detach().numpy() * 10
    X_ef = X_ef.squeeze().cpu().detach().numpy()
    points = np.array([[i[0] * 50, i[1] * 80 + 100] for i in X_ef])

    lgt = len(target)
    a = np.arange(0, lgt)
    index = np.arange(0, lgt - 0.5, 0.5)
    new_lat = np.interp(index, xp=a, fp=points[:, 0])
    new_lon = np.interp(index, xp=a, fp=points[:, 1])
    points = np.asarray([[new_lat[index], new_lon[index]] for index in range(len(new_lon))])
    target = np.interp(index, xp=a, fp=target[:])
    pred = np.interp(index, xp=a, fp=pred[:])

    new_times = []
    for i in times:
        new_times.append(i)
        hour = int(i[11:]) + 300
        new_times.append('-'.join([i[0:10], str(hour).zfill(4)]))
    new_times.__delitem__(-1)
    times = new_times

    times = pandas.to_datetime(pandas.Series(times))
    # new_times = []
    # for index, i in enumerate(times):
    #     if index % 2 == 0 or index + 1 == len(times):
    #         new_times.append(i)
    d1 = {'UTC': times, 'Best track MSW': target}
    d2 = {'UTC': times, 'Estimated intensities': pred}
    data1 = pandas.Series(d1)
    data2 = pandas.Series(d2)
    # data2 = data2.set_index('UTC')
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.tick_params(axis='y', labelsize=20)  # y轴
    ax.tick_params(axis='x', labelsize=15, rotation=45)  # y轴
    # sns.lineplot(x='UTC',y='value',hue='variable',data=pandas.melt(data,['UTC']))
    ax = sns.lineplot(data=data1, color='black', x='UTC', y='Best track MSW')
    ax = sns.lineplot(data=data2, color='orangered', x='UTC', y='Estimated intensities')

    ymin, ymax = 10, 75
    on_land = is_on_land(points)
    plt.fill_between(data2['UTC'].values, y1=ymin, y2=ymax, where=on_land,
                     color='skyblue', alpha=0.3)
    is_RI = is_rapid_intensification(target)
    plt.fill_between(data2['UTC'].values, y1=ymin, y2=ymax, where=is_RI,
                     color='coral', alpha=0.2)
    plt.legend([], [], frameon=False)
    ax.margins(x=0)
    plt.ylim(ymin, ymax)
    # plt.xticks(new_times)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.grid(False)
    plt.show()


def parse_one_ty(which_model=1,ty='F:/data/TC_IR_IMAGE/2015/201513_SOUDELOR'):
    args.past_window = 5
    model = get_model(which_model)
    model.eval()
    if which_model == 1:
        X_im, X_ef, X_sst, target, times = build_one_ty(ty,split=True)
        assert X_im.size(0) == len(times)
        pred, f_div_C, W_y = model(X_im, X_ef, X_sst, return_nl_map=True)
        X_ef = X_ef[:, -1, :]
    else:
        X_im, X_ef, X_sst, target, times = build_one_ty(ty,split=False)
        assert X_im.size(1) == len(times)
        pred, f_div_C, W_y = model(X_im, X_ef, X_sst, return_nl_map=True)
        target = target[0]

    plot_track(X_ef, target, pred, times)
    W_y = W_y.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    f_div_C = f_div_C.cpu().detach().numpy()

    # W_y = np.abs(W_y)

    m = np.mean(W_y, axis=1)
    max_value = np.max(m)
    min_value = np.min(m)
    # max_value=0.5
    # min_value=0

    points = pred.size(0)
    length = args.past_window
    for point in range(points):
        attention = W_y[point]
        attention = attention.mean(axis=0).transpose()
        assert np.shape(attention) == (3, length)

        if args.past_window != 5:
            new_attention = []
            for one_view in attention:
                a = np.arange(0, length)
                index = np.arange(0, length - 0.5, 0.5)
                new_atts = np.interp(index, xp=a, fp=one_view)
                new_attention.append(new_atts)
            attention = new_attention
        xlabels = [f't-{length - t}' if t != length else f't' for t in range(1, length + 1)]
        ylabels = [f'{m}' for m in range(1, 4)]
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(attention, cmap="Greys", vmax=max_value, vmin=0, xticklabels=xlabels,
                    yticklabels=ylabels, annot=False)
        name = '-'.join([str(times[point]), str(target[point])])
        ax.tick_params(axis='y', labelsize=20)  # y轴
        ax.tick_params(axis='x', labelsize=20)  # y轴
        # plt.show()
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        plt.tight_layout()
        dic = '/home/dl/data/TCIE/Attentions'
        if not os.path.exists(dic):
            dic = 'D:/DATA/attentions3/'
        path = os.path.join(dic, f'{name}.jpg')
        plt.savefig(path)
        plt.clf()

        # atts.append(f_div_C)
    # plot_attentin(atts)


if __name__ == '__main__':
    version = 1
    # parse_one_ty(version,ty='F:/data/TC_IR_IMAGE/2018/201805_MALIKSI')
    sns.heatmap(np.random.rand(3,5), cmap="Greys", vmax=1, vmin=0)
    plt.show()
    pass
