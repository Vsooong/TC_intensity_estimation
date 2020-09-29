import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd
sns.set_style("whitegrid")

# data=pd.read_csv('F:/data/typhoon_data/test_res3.csv')
# values=data['RMSE'].values
# sns.despine(left=True)
# a4_dims = (8, 6)
# fig, ax = plt.subplots(figsize=a4_dims)
# # ax.grid(False)
# sns.barplot(x="Year", y="RMSE", hue="Model",data=data,ax=ax,
#             palette=sns.color_palette("Paired", 12))
# ax.tick_params(axis='y',labelsize=20) # y轴
# ax.tick_params(axis='x',labelsize=20) # y轴
#
# index=0
# # for p in ax.patches:
# #     nindex= 6*int(index%4)+int(index/4)
# #     ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%.2f' % np.around(values[nindex],decimals=2),
# #             fontsize=20, color='black', ha='center', va='bottom')
# #     index+=1
# plt.legend([],[], frameon=False)
# plt.ylim(4, 6.0)
# plt.yticks(np.arange(4.0,6.0,0.2))
# plt.show()

data=pd.read_csv('F:/data/typhoon_data/test_res4.csv')
values=data['RMSE'].values
sns.despine(left=True)
a4_dims = (8, 6)
fig, ax = plt.subplots(figsize=a4_dims)
# ax.grid(False)
g=sns.barplot(x="Modality", y="RMSE", hue="Model",data=data,ax=ax,
            palette=sns.color_palette("Paired", 12))
ax.tick_params(axis='y',labelsize=20) # y轴
ax.tick_params(axis='x',labelsize=20) # y轴
plt.ylim(4, 5.8)
plt.yticks(np.arange(4.0,5.8,0.2))
plt.legend([],[], frameon=False)
plt.show()

# a=[0.816, 0.8072,0.8036,0.8029,0.8028]
# b=[1,2,3,4,5]
#
# plt.plot(b,a,'o-',linestyle='--')
# plt.xlabel('window size P')
# plt.tick_params(axis='y',labelsize=15)
# plt.tick_params(axis='x',labelsize=15)
# plt.ylabel('training loss')
# plt.ylim([0.75,0.85])
# plt.grid(True)
# plt.xticks(b)
#
# plt.show()


