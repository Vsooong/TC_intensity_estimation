import math
import pandas
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from visualize.get_model_result import estimate_one_ty, build_one_ty, get_model
import os
from utils.Utils import args
from global_land_mask import globe
import matplotlib.dates as mdates

sns.set_theme(color_codes=True)
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
        if preindex < 0: w.append(False)
        elif intensities[index] - intensities[preindex] >= 30 * 0.5144:
            w.append(True)
        else:
            w.append(False)
    return w


def case_study_track():
    X_im, X_ef, X_sst, target, times = build_one_ty(split=False)
    assert X_im.size(1) == len(times)
    model = get_model(2)
    pred, f_div_C, W_y = estimate_one_ty(X_im, X_ef, X_sst, model)
    target = target.squeeze().cpu().detach().numpy()
    pred = pred.squeeze().cpu().detach().numpy()
    X_ef = X_ef.squeeze().cpu().detach().numpy()
    points = np.array([[i[0], i[1] + 100] for i in X_ef])

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
    data1 = pandas.DataFrame(d1)
    data2 = pandas.DataFrame(d2)
    data2 = data2.set_index('UTC')
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.tick_params(axis='y', labelsize=20)  # y轴
    ax.tick_params(axis='x', labelsize=15, rotation=45)  # y轴
    # sns.lineplot(x='UTC',y='value',hue='variable',data=pandas.melt(data,['UTC']))
    ax = sns.lineplot(data=data1, color='black', x='UTC', y='Best track MSW')
    ax = sns.lineplot(data=data2, color='orangered', x='UTC', y='Estimated intensities')

    ymin, ymax = 10, 75
    on_land = is_on_land(points)
    plt.fill_between(data2.index, y1=ymin, y2=ymax, where=on_land,
                     color='skyblue', alpha=0.3)
    is_RI= is_rapid_intensification(target)
    plt.fill_between(data2.index, y1=ymin, y2=ymax, where=is_RI,
                     color='coral', alpha=0.2)
    plt.legend([], [], frameon=False)
    ax.margins(x=0)
    plt.ylim(ymin, ymax)
    # plt.xticks(new_times)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.grid(False)
    plt.show()


def parse_one_ty(which_model=1):
    args.past_window = 3
    if which_model == 1:
        X_im, X_ef, X_sst, target, times = build_one_ty(split=True)
        assert X_im.size(0) == len(times)
    else:
        X_im, X_ef, X_sst, target, times = build_one_ty(split=False)
        target = target[0]
        assert X_im.size(1) == len(times)

    model = get_model(which_model)
    pred, f_div_C, W_y = estimate_one_ty(X_im, X_ef, X_sst, model)
    print(W_y.shape)
    print(f_div_C.shape)
    W_y = W_y.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    f_div_C = f_div_C.cpu().detach().numpy()

    # W_y = np.abs(W_y)
    m = np.mean(W_y, axis=1)
    max_value = np.max(m)
    min_value = np.min(m)
    # max_value=0.5
    # min_value=0

    layers = pred.size(0)
    for i in range(layers):
        attention = W_y[i]
        attention = attention.mean(axis=0)
        sns.heatmap(attention.transpose(), cmap="Greys", vmax=max_value, vmin=min_value, annot=True)
        name = '-'.join([str(times[i]), str(target[i])])
        # plt.show()
        dic = '/home/dl/data/TCIE/Attentions'
        if not os.path.exists(dic):
            dic = 'D:/DATA/attentions/'
        path = os.path.join(dic, f'{name}.jpg')
        plt.savefig(path)
        plt.clf()

        # atts.append(f_div_C)
    # plot_attentin(atts)


if __name__ == '__main__':
    # parse_one_ty()
    case_study_track()

    pass
