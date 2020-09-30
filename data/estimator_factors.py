import xarray
import numpy as np
from datetime import datetime
from math import radians, sin, cos, acos
import matplotlib.pyplot as plt
import time as TM

def distance_on_earth(lat1, long1, lat2, long2):
    slat = radians(lat1)
    slon = radians(long1)
    elat = radians(lat2)
    elon = radians(long2)

    dist = 6371.01 * acos(sin(slat) * sin(elat) + cos(slat) * cos(elat) * cos(slon - elon))
    return dist


def select_area_nc(time, clat, clon, radius1, radius2, nc_file, max_lat, min_lon,select_mode):

    times, lats, lons = nc_file.shape
    values = []
    # max_lat = max(nc_file.coords['latitude'].data)
    # min_lon = min(nc_file.coords['longitude'].data)
    if select_mode ==2:
        value=nc_file[time,...].sel(latitude=clat,longitude=clon,method="nearest").data
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


sst = xarray.open_dataarray('/home/dl/data/TCIE/mcs/sst2000-2019.nc', cache=True) - 273.16
max_lat = max(sst.coords['latitude'].data)
min_lon = min(sst.coords['longitude'].data)
start=TM.time()
values = select_area_nc(time=10923, clat=20, clon=120, radius1=200, radius2=800, nc_file=sst, max_lat=max_lat,
                        min_lon=min_lon,select_mode=1)
end=TM.time()
print(end-start)
print(values)


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
