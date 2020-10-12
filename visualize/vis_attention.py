import math
import pandas
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from visualize.get_model_result import estimate_one_ty, build_one_ty, get_model
import os
from utils.Utils import args


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


def parse_one_ty(which_model=2):
    args.past_window =3
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
    print(target.shape)
    W_y = W_y.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    m = np.mean(W_y, axis=1)
    max_value = np.max(m)+0.2
    min_value = np.min(m)
    # max_value=0.5
    # min_value=0

    layers = pred.size(0)
    for i in range(layers):
        w_i = W_y[i]
        w_i = np.abs(w_i)
        attention = w_i.mean(axis=0)
        sns.heatmap(attention.transpose(), cmap="Greys", vmax=max_value, vmin=min_value, annot=True)
        name = '-'.join([str(times[i]), str(target[i])])
        # plt.show()
        path = os.path.join('D:/DATA/attentions/', f'{name}.jpg')
        plt.savefig(path)
        plt.clf()

        # atts.append(f_div_C)
    # plot_attentin(atts)


if __name__ == '__main__':
    parse_one_ty()
