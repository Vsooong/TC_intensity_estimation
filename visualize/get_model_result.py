from utils.Utils import args
from TC_data import TC_Data
import xarray
import numpy as np
import torch.nn as nn
from TC_data import getOneTyphoon
import torch
from TC_estimate import MSFN_v1, MSFN
import os

Sea_Surface_Temperature = None


def evaluate(model, dataset, version=1):
    evaluateL1 = nn.L1Loss(reduction='sum')
    evaluateL2 = nn.MSELoss(reduction='sum')
    model.eval()
    n_samples = 0
    total_loss1 = 0
    total_loss2 = 0
    labels = []
    predicts = []
    if version == 1:
        for minibatch in dataset.get_batches():
            images, efactors, envsst, targets = minibatch
            pred = model(images, efactors, envsst)
            if np.isnan(pred.data.cpu()).sum() != 0:
                print(efactors)
                print(targets)
                print(pred)
            targets = targets[:, -1, :]
            total_loss1 += evaluateL1(targets, pred).data.item()
            total_loss2 += evaluateL2(targets, pred).data.item()
            n_samples += len(targets)
            for index in range(len(targets)):
                labels.append(targets[index].data.item())
                predicts.append(pred[index].data.item())
        r = np.corrcoef(labels, predicts)[0][1]
        return total_loss1 / n_samples, np.sqrt(total_loss2 / n_samples), r
    else:
        for one_ty in dataset.get_one_ty():
            images, efactors, envsst, targets = one_ty
            pred = model(images, efactors, envsst).squeeze(-1)
            targets = targets[0, :, :]
            if np.isnan(pred.data.cpu()).sum() != 0:
                print(efactors)
                print(targets)
                print(pred)
            total_loss1 += evaluateL1(targets, pred).data.item()
            total_loss2 += evaluateL2(targets, pred).data.item()
            n_samples += len(targets)
            for index in range(len(targets)):
                labels.append(targets[index].data.item())
                predicts.append(pred[index].data.item())
        r = np.corrcoef(labels, predicts)[0][1]
        return total_loss1 / n_samples, np.sqrt(total_loss2 / n_samples), r


def build_one_ty(ty='F:/data/TC_IR_IMAGE/2015/201513_SOUDELOR', split=False):
    if not os.path.exists(ty):
        ty = '/home/dl/data/TCIE/TC_IR_IMAGE/2015/201513_SOUDELOR'

    global Sea_Surface_Temperature
    if Sea_Surface_Temperature is None:
        Sea_Surface_Temperature = xarray.open_dataarray(args.sea_surface_temperature, cache=True)
    device = args.device
    mvts, isi, sst, times = getOneTyphoon(ty, build_nc_seq=True, SST=Sea_Surface_Temperature)
    efactors = torch.as_tensor(mvts[:, 0:10])
    target = torch.as_tensor(mvts[:, 10:])
    images = torch.as_tensor(isi)
    env_sst = torch.as_tensor(sst)
    if split:
        length = efactors.size(0)
        start_idx = 0
        index = torch.as_tensor(range(length), device=args.device, dtype=torch.long)
        past_window = args.past_window
        X_ef = []
        X_im = []
        X_sst = []
        while start_idx <= length - past_window:
            excerpt = index[start_idx:(start_idx + past_window)]
            X_ef.append(torch.as_tensor(efactors[excerpt]))
            X_im.append(torch.as_tensor(images[excerpt]))
            X_sst.append(torch.as_tensor(env_sst[excerpt]))
            start_idx += 1
        X_ef = torch.stack(X_ef, dim=0).to(device)
        X_im = torch.stack(X_im, dim=0).to(device)
        X_sst = torch.stack(X_sst, dim=0).to(device)
        return X_im, X_ef, X_sst, target[past_window - 1:], times[past_window - 1:]

    else:
        images = images.unsqueeze(0).to(device)
        efactors = efactors.unsqueeze(0).to(device)
        env_sst = env_sst.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        return images, efactors, env_sst, target, times


def estimate_one_ty(X_im, X_ef, X_sst, model):
    model.eval()
    pred, f_div_C, W_y = model(X_im, X_ef, X_sst, return_nl_map=True)
    return pred, f_div_C, W_y


def get_model(which=1):
    if which == 1:
        # 按照past window 分段
        # model_name = 'MSFN_v1.pth'
        model_name = 'MSFN_v1.pth'
        model = MSFN_v1.get_MSFN_v1(True, model_name)
    else:
        # 全序列
        model_name = 'MSFN-5-4.8.pth'
        model = MSFN.get_MSFN(True, model_name)
    print('use model:', model_name)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    return model


def main():
    version = 1
    model = get_model(version)
    dataset_test = TC_Data(years=[2000, 2005, 2010, 2015, 2019])
    # dataset_test = TC_Data(years=args.test_years)
    print('Test samples:', len(dataset_test.targets))
    print('------------------------------------------\n')
    loss1, loss2, r = evaluate(model, dataset_test, version)
    print(loss1, loss2, r)


if __name__ == '__main__':
    main()
    #
    # X_im, X_ef, X_sst, target, times = build_one_ty()
    # print(X_im.shape)
    # print(X_sst.shape)
    # print(target.shape)
    # model = get_model()
    # pred, f_div_C, W_y = estimate_one_ty(X_im, X_ef, X_sst, model)
    # print(pred.shape)
    # print(len(W_y))
    # print(len(times))
