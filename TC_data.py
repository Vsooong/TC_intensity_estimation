from utils.Utils import args
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from PIL import Image
import numpy as np


class TC_Data(Dataset):
    def __init__(self, data_root=args.img_root, years=args.train_years, past_window=args.past_window,
                 device=args.device, build_nc_seq=False):
        self.typhoons = self.init_years(data_root, years)
        self.past_window = past_window
        self.device = device
        self.scale = None
        self.efactors, self.images, self.targets, self.ids = self._build_seq_data()
        self.build_nc_seq = build_nc_seq

    def get_batches(self, batch_size=args.batch_size):
        length = self.efactors.size(0)
        start_idx = 0
        index = torch.as_tensor(range(length), device=self.device, dtype=torch.long)
        assert length == len(self.ids)

        while start_idx <= length - self.past_window + 1:
            X_ef = []
            X_im = []
            Y_int = []
            # piece_len = batch_size - 1 + self.past_window
            # end_idx = min(length, start_idx + piece_len)
            for i in range(batch_size):
                if start_idx + self.past_window - 1 < length and self.ids[start_idx] == self.ids[
                    start_idx + self.past_window - 1]:
                    excerpt = index[start_idx:(start_idx + self.past_window)]
                    X_ef.append(torch.as_tensor(self.efactors[excerpt]).to(self.device))
                    X_im.append(torch.as_tensor(self.images[excerpt]).to(self.device))
                    Y_int.append(torch.as_tensor(self.targets[excerpt]).to(self.device))
                start_idx += 1
            # X_im = X_im.transpose(0, 1)
            X_im = torch.stack(X_im, dim=0).to(self.device)
            X_ef = torch.stack(X_ef, dim=0).to(self.device)
            Y_int = torch.stack(Y_int, dim=0).to(self.device)
            yield X_im, X_ef, Y_int

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
        target = []
        tc_id = []

        for idx in range(0, tphns):
            ty = self.typhoons[idx]
            mvts, isi = getOneTyphoon(ty)
            efactor.append(mvts[:, 0:10])
            images.append(isi)
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
        labels = torch.cat(target, dim=0)
        ids = torch.cat(tc_id, dim=0)
        return efactors, infr_img, labels, ids

    def __len__(self):
        return self.efactors.size(0)

    def __getitem__(self, idx):
        return self.ef[idx], self.images[idx], self.target[idx]


def get_transform():
    transforms = list()
    transforms.append(T.Resize((args.img_height, args.img_width)))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def getOneTyphoon(dir, build_nc_seq=False):
    nc_file = None
    if build_nc_seq:
        global sst
        nc_file = sst
    mvts = []
    isi = []
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
            # month = float(temp[0][-4:-2])
            # jday = d_to_jd(temp[0])
            # hour = float(temp[1][0:2])

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
            if nc_file is not None:
                pass
            im1 = Image.open(os.path.join(files[0], image)).convert("L")
            im1 = transform(im1)
            isi.append(im1)
    mvts = torch.tensor(mvts)
    isi = torch.stack(isi, dim=0)
    return mvts, isi


if __name__ == '__main__':
    pass
    # mvts, isi = getOneTyphoon('F:/data/TC_IR_IMAGE/2010/201001_OMAIS')
    # print(isi.shape)

    # tc_data = TC_Data(years=[1995])
    # for minibatch in tc_data.get_batches():
    #     images, efactors, targets = minibatch
    #     targets = targets[:, -1, :]
    #     print(images.shape)
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
