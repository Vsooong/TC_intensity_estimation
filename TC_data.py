from utils.Utils import args
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from PIL import Image
import numpy as np
from datetime import date
import xarray
import time as TM
import matplotlib.pyplot as plt

Sea_Surface_Temperature = None


class TC_Data(Dataset):
    def __init__(self, data_root=args.img_root, years=args.train_years, past_window=args.past_window,
                 device=args.device):
        self.typhoons = self.init_years(data_root, years)
        self.past_window = past_window
        self.device = device
        global Sea_Surface_Temperature
        if Sea_Surface_Temperature is None:
            Sea_Surface_Temperature = xarray.open_dataarray(args.sea_surface_temperature, cache=True)

        self.efactors, self.images, self.env_sst, self.targets, self.ids = self._build_seq_data()

    def get_batches(self, batch_size=args.batch_size):
        length = self.efactors.size(0)
        start_idx = 0
        index = torch.as_tensor(range(length), device=self.device, dtype=torch.long)
        assert length == self.ids.size(0)

        while start_idx <= length - self.past_window + 1:
            X_ef = []
            X_im = []
            X_sst = []
            Y_int = []
            # piece_len = batch_size - 1 + self.past_window
            # end_idx = min(length, start_idx + piece_len)
            for i in range(batch_size):
                if start_idx + self.past_window - 1 < length and self.ids[start_idx] == self.ids[
                    start_idx + self.past_window - 1]:
                    excerpt = index[start_idx:(start_idx + self.past_window)]
                    X_ef.append(torch.as_tensor(self.efactors[excerpt]).to(self.device))
                    X_im.append(torch.as_tensor(self.images[excerpt]).to(self.device))
                    X_sst.append(torch.as_tensor(self.env_sst[excerpt]).to(self.device))
                    Y_int.append(torch.as_tensor(self.targets[excerpt]).to(self.device))
                start_idx += 1
            # X_im = X_im.transpose(0, 1)
            X_ef = torch.stack(X_ef, dim=0).to(self.device)
            X_im = torch.stack(X_im, dim=0).to(self.device)
            X_sst = torch.stack(X_sst, dim=0).to(self.device)
            Y_int = torch.stack(Y_int, dim=0).to(self.device)
            yield X_im, X_ef, X_sst, Y_int

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
        tc_id = []

        for idx in range(0, tphns):
            ty = self.typhoons[idx]
            mvts, isi, ssts = getOneTyphoon(ty, True)
            efactor.append(mvts[:, 0:10])
            images.append(isi)
            env_sst.append(ssts)
            target.append(mvts[:, 10:])
            plen = mvts.size(0)
            tc_id.append(torch.ones(plen) * idx)
            # lth = len(isi)
            # res = int((lth - self.past_window) / self.stride) + 1
            # if res <= 0: continue
            # for i in range(0, res):
            #     i = i * self.stride
            #     data.append(mvts[i:i + self.past_window, :])
            #     images.append(isi[i:i + self.past_window, ...])
            #     target.append(mvts[i:i + self.past_window, 0])

        efactors = torch.cat(efactor, dim=0)
        infr_img = torch.cat(images, dim=0)
        env_sst = torch.cat(env_sst, dim=0)
        labels = torch.cat(target, dim=0)
        ids = torch.cat(tc_id, dim=0)
        return efactors, infr_img, env_sst, labels, ids

    def __len__(self):
        return self.efactors.size(0)

    def __getitem__(self, idx):
        return self.efactors[idx], self.images[idx], self.env_sst[idx], self.targets[idx]


def get_transform():
    transforms = list()
    transforms.append(T.Resize((args.img_size, args.img_size)))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


start_date = date(2000, 1, 1)


def day_diff(date):
    global start_date
    delta = date - start_date
    return delta.days


def relative_coord(l_lat1, l_lon1, l_lat2, l_lon2, r_lat1=0, r_lon1=100, r_lat2=50, r_lon2=180, resl=4):
    ovlat1 = max(l_lat1, r_lat1)
    ovlat2 = min(l_lat2, r_lat2)
    ovlon1 = max(l_lon1, r_lon1)
    ovlon2 = min(l_lon2, r_lon2)

    rows1 = (r_lat2 - ovlat2) * resl
    rows2 = (r_lat2 - ovlat1) * resl
    cols1 = (ovlon1 - r_lon1) * resl
    cols2 = (ovlon2 - r_lon1) * resl
    return int(rows1), int(rows2), int(cols1), int(cols2)


def get_sst(file_path='F:/data/msc/sst2000-2019.nc'):
    sst = xarray.open_dataarray(file_path, cache=False)
    # - 273.16
    return sst


def getOneTyphoon(dir, build_nc_seq=False):
    global Sea_Surface_Temperature
    mvts = []
    isi = []
    ssts = []
    files = sorted([os.path.join(dir, i) for i in os.listdir(dir)])
    channel1 = sorted(os.listdir(files[0]))
    transform = get_transform()

    for index in range(len(channel1)):
        image = channel1[index]
        if image.endswith('jpg'):
            temp = image.split('-')
            if str(temp[1]) not in args.time_spot:
                continue
            ori_intense = float(temp[-1].split('.')[0])
            if ori_intense == 0:
                continue
            cdate = temp[0]
            hour = temp[1]
            time = date(int(cdate[:4]), int(cdate[4:6]), int(cdate[6:8]))
            time = day_diff(time) * 8 + int(int(hour[0:2]) / 3)

            lat = float(temp[2])
            lon = float(temp[3])
            stp = float(temp[4])
            jdate = float(temp[5])
            centra_sst = float(temp[6])
            mpi = float(temp[7])
            rh600 = float(temp[8])
            t200 = float(temp[9])
            slr200 = float(temp[10])
            if np.isnan(slr200):
                slr200 = 0
            slr800 = float(temp[11])
            if np.isnan(slr800):
                slr800 = 0
            # pres = float(temp[-2])
            if lat > 50 or lat < 0 or lon > 180 or lon < 100:
                continue

            record = [lat, lon - 100, stp, jdate / 10, centra_sst, mpi, rh600, t200 - 273.16, slr200, slr800,
                      ori_intense]
            mvts.append(record)
            # mvts.append([ori_intense])
            # if np.isnan(record).sum() != 0: print()
            if build_nc_seq:
                sst_background = np.zeros(shape=(args.sst_size, args.sst_size))
                cen_lat = int(lat)
                cen_lon = int(lon)
                lon1 = cen_lon - args.sst_size / 8 + 0.25
                lon2 = cen_lon + args.sst_size / 8
                lat1 = cen_lat - args.sst_size / 8 + 0.25
                lat2 = cen_lat + args.sst_size / 8
                rows1, rows2, cols1, cols2 = relative_coord(lat1, lon1, lat2, lon2)
                sst = Sea_Surface_Temperature[time, rows1:rows2 + 1, cols1:cols2 + 1]
                rows1, rows2, cols1, cols2 = relative_coord(0, 100, 50, 180, lat1, lon1, lat2, lon2)
                sst_background[rows1:rows2 + 1, cols1:cols2 + 1] = sst.data - 273.16
                sst_background[np.isnan(sst_background)] = 0
                ssts.append(torch.as_tensor(sst_background))
                # sst.plot()
                # plt.show()
                # print(sst_background.shape)
                # print(lat1, lon1, lat2, lon2)
                # print(sst.coords)

            im1 = Image.open(os.path.join(files[0], image)).convert("L")
            im1 = transform(im1)
            isi.append(im1)
    mvts = torch.tensor(mvts)
    isi = torch.stack(isi, dim=0)
    ssts = torch.stack(ssts, dim=0)
    return mvts, isi, ssts


if __name__ == '__main__':
    # file_path = 'F:/data/msc/sst2000-2019.nc'
    Sea_Surface_Temperature = xarray.open_dataarray('/home/dl/data/TCIE/mcs/sst2000-2019.nc', cache=True)
    # start = TM.time()
    # for i in range(10):
    # mvts, isi, sst = getOneTyphoon('/home/dl/data/TCIE/TC_IR_IMAGE/2019/201929_PHANFONE', build_nc_seq=True)
    # end = TM.time()
    # print(end - start)

    tc_data = TC_Data(years=[2000])
    print(tc_data.env_sst.shape)

    for minibatch in tc_data.get_batches():
        images, efactors, envsst, targets = minibatch
        #     targets = targets[:, -1, :]
        print(envsst.shape)
        print(efactors.shape)
    #     print(efactors.shape)
    #     print(targets.shape)

    # typhoon_list = []
    #
    # for i in sorted(os.listdir(args.img_root), reverse=True):
    #     if os.path.isdir(os.path.join(args.img_root, i)):
    #         ip = os.path.join(args.img_root, i)
    #         for j in sorted(os.listdir(ip)):
    #             jp = os.path.join(ip, j)
    #             if int(i) in args.train_years:
    #                 typhoon_list.append(jp)
    # for ty in typhoon_list:
    #     getOneTyphoon(ty)
