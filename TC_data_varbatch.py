from utils.Utils import args
import torch
from torch.utils.data import Dataset
from TC_data import getOneTyphoon
import os
from datetime import date
import xarray
import torch.nn.utils.rnn as rnn_utils
import time as TM
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

Sea_Surface_Temperature = None


class TC_Data_varbatch(Dataset):
    def __init__(self, data_root=args.img_root, years=args.train_years, past_window=args.past_window,
                 device=args.device):
        self.typhoons = self.init_years(data_root, years)
        self.past_window = past_window
        self.device = device
        global Sea_Surface_Temperature
        if Sea_Surface_Temperature is None:
            Sea_Surface_Temperature = xarray.open_dataarray(args.sea_surface_temperature, cache=True)
        self.efactors, self.images, self.env_sst, self.targets = self._build_seq_data()

    def get_batches(self, batch_size=args.batch_size):
        length = len(self.efactors)
        start_idx = 0
        index = torch.as_tensor(range(length), device=self.device, dtype=torch.long)
        while start_idx < length:
            X_ef = []
            X_im = []
            X_sst = []
            Y_int = []
            end_idx = min(length, start_idx + batch_size)
            X_ef = self.efactors[start_idx:end_idx]
            X_ef.sort(key=lambda x: len(x), reverse=True)
            X_im = self.images[start_idx:end_idx]
            X_im.sort(key=lambda x: len(x), reverse=True)
            X_sst = self.env_sst[start_idx:end_idx]
            X_sst.sort(key=lambda x: len(x), reverse=True)
            Y_int = self.targets[start_idx:end_idx]
            Y_int.sort(key=lambda x: len(x), reverse=True)
            data_length = [len(sq) for sq in X_ef]
            X_ef = rnn_utils.pad_sequence(X_ef, batch_first=True, padding_value=0).to(self.device)
            X_im = rnn_utils.pad_sequence(X_im, batch_first=True, padding_value=0).to(self.device)
            X_sst = rnn_utils.pad_sequence(X_sst, batch_first=True, padding_value=0).to(self.device)
            Y_int = rnn_utils.pad_sequence(Y_int, batch_first=True, padding_value=0).to(self.device)
            start_idx += batch_size
            yield X_im, X_ef, X_sst, Y_int, data_length

    def init_years(self, data_root, years):
        typhoon_list = []

        for i in sorted(os.listdir(data_root), reverse=True):
            if os.path.isdir(os.path.join(data_root, i)):
                ip = os.path.join(data_root, i)
                for j in sorted(os.listdir(ip)):
                    jp = os.path.join(ip, j)
                    if years and int(i) in years:
                        typhoon_list.append(jp)
        return typhoon_list

    def _build_seq_data(self):
        tphns = len(self.typhoons)
        efactor = []
        images = []
        env_sst = []
        target = []

        for idx in range(0, tphns):
            ty = self.typhoons[idx]
            mvts, isi, ssts, times = getOneTyphoon(ty, True, SST=Sea_Surface_Temperature)
            efactor.append(mvts[:, 0:10])
            images.append(isi)
            env_sst.append(ssts)
            target.append(mvts[:, 10:])
        return efactor, images, env_sst, target

    def __len__(self):
        return len(self.efactors)

    def __getitem__(self, idx):
        return self.efactors[idx], self.images[idx], self.env_sst[idx], self.targets[idx]


if __name__ == '__main__':
    tc_data = TC_Data_varbatch(years=[2000])
    for minibatch in tc_data.get_batches():
        images, efactors, envsst, targets, batch_len = minibatch
        images=rnn_utils.pack_padded_sequence(images, batch_len, batch_first=True)
        efactors=rnn_utils.pack_padded_sequence(efactors, batch_len, batch_first=True)
        envsst=rnn_utils.pack_padded_sequence(envsst, batch_len, batch_first=True)
        targets=rnn_utils.pack_padded_sequence(targets, batch_len, batch_first=True)

        #      targets = targets[:, -1, :]
        print(envsst)
        print(efactors)
        print(targets)
        print(batch_len)
