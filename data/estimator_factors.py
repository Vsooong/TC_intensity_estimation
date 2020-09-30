import xarray
import numpy as np
from datetime import datetime
from math import radians, sin, cos, acos
import matplotlib.pyplot as plt
import time as TM
import os
import torch
from utils.Utils import args


def distance_on_earth(lat1, long1, lat2, long2):
    slat = radians(lat1)
    slon = radians(long1)
    elat = radians(lat2)
    elon = radians(long2)

    dist = 6371.01 * acos(sin(slat) * sin(elat) + cos(slat) * cos(elat) * cos(slon - elon))
    return dist


def select_area_nc(time, clat, clon, radius1, radius2, nc_file, max_lat, min_lon, select_mode):
    times, lats, lons = nc_file.shape
    values = []
    # max_lat = max(nc_file.coords['latitude'].data)
    # min_lon = min(nc_file.coords['longitude'].data)
    if select_mode == 2:
        value = nc_file[time, ...].sel(latitude=clat, longitude=clon, method="nearest").data
        return value
    for i in range(lats):
        for j in range(lons):
            dlon = min_lon + j * 0.25
            dlat = max_lat - i * 0.25

            # if abs(dlat - clat) > (radius1 / 111):
            #     continue
            distance = distance_on_earth(clat, clon, dlat, dlon)
            if radius1 <= distance <= radius2:
                value = nc_file[time, i, j].data
                # print(value)
                if not np.isnan(value):
                    values.append(value)
    # nc_file[time,...].plot()
    # plt.show()

    return values


def label_one_typhoon(dir):
    print(dir)
    files = sorted([os.path.join(dir, i) for i in os.listdir(dir)])
    channel1 = sorted(os.listdir(files[0]))
    for index in range(len(channel1)):
        image = channel1[index]
        if image.endswith('jpg'):
            temp = image.split('-')
            if str(temp[1]) not in args.time_spot:
                continue
            print(image)
            date = temp[0]
            hour = temp[1]
            lat = temp[2]
            lon = temp[3]
            stp = temp[4]
            centra_sst = temp[5]
            # mpi=cal_MPI_from_SST(centra_sst)
            onsea = temp[6]
            sl_ratio = temp[7]
            pres = temp[-2]

            ori_intense = temp[-1].split('.')[0]
            ori_path = '-'.join([date, hour, lat, lon, stp, centra_sst, onsea, sl_ratio, pres, ori_intense])
            print(ori_intense)


def label_ef_to_images():
    def init_years(data_root=args.img_root, years=args.train_years):
        typhoon_list = []

        for i in sorted(os.listdir(data_root), reverse=True):
            if os.path.isdir(os.path.join(data_root, i)):
                ip = os.path.join(data_root, i)
                for j in sorted(os.listdir(ip)):
                    jp = os.path.join(ip, j)
                    if years and int(i) in years:
                        typhoon_list.append(jp)
        return typhoon_list

    typhoons = init_years()
    for ty in typhoons:
        label_one_typhoon(ty)


label_ef_to_images()


# sst = xarray.open_dataarray('/home/dl/data/TCIE/mcs/sst2000-2019.nc', cache=True) - 273.16
# max_lat = max(sst.coords['latitude'].data)
# min_lon = min(sst.coords['longitude'].data)
# start=TM.time()
# values = select_area_nc(time=10923, clat=20, clon=120, radius1=200, radius2=800, nc_file=sst, max_lat=max_lat,
#                         min_lon=min_lon,select_mode=1)
# end=TM.time()
# print(end-start)
# print(values)

def cal_MPI_from_SST(sst_value):
    sst_value = max(16.0, sst_value)
    sst_value = min(32.5, sst_value)
    mpi = 29.59 + 108.1 * np.exp(0.1292 * (sst_value - 30.0))
    return min(mpi, 140)


def d_to_jd(time):
    fmt = '%Y%m%d'
    dt = datetime.strptime(time, fmt)
    tt = dt.timetuple().tm_yday
    return tt

# file1 = '/home/dl/data/TCIE/mcs/sst-1.nc'
# file2 = '/home/dl/data/TCIE/mcs/sst.nc'
#
# data1 = xarray.open_dataarray(file1, decode_times=True)[43824:,...]
#
# data2 = xarray.open_dataarray(file2, decode_times=True)
# merge_data=xarray.concat([data1,data2],'time')
#
# merge_data.to_netcdf('/home/dl/data/TCIE/mcs/sst2000-2019.nc')
# print(merge_data)
