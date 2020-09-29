import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import xarray
import numpy as np
from datetime import datetime

file1='F:/data/msc/relative-humidity.nc'
file2='F:/data/msc/rh.nc'

data1=xarray.open_dataarray(file1, cache=True, decode_times=True)
data2=xarray.open_dataarray(file2, cache=True, decode_times=True)
merge_data=xarray.concat([data1,data2],'time')
print(merge_data)
def d_to_jd(time):
    fmt = '%Y%m%d'
    dt = datetime.strptime(time, fmt)
    tt = dt.timetuple().tm_yday
    return tt