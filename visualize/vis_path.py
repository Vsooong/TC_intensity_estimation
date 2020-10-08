import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patches as mpatch
from cartopy.io.img_tiles import Stamen

colors = {'TD': 'blue', 'TS': 'limegreen', 'STS': 'gold', 'TY': 'darkorange', 'STY': 'magenta', 'SuperTY': 'red'}
best_track = {'TD': [], 'TS': [], 'STS': [], 'TY': [], 'STY': [], 'SuperTY': []}


# http://agora.ex.nii.ac.jp/digital-typhoon/summary/wnp/l/201513.html.en
def sample_data():
    lats1 = [13.3, 13.7, 13.6, 13.6, 13.5, 13.4, 13.2, 13.4, 13.4, 13.6, 14, 14.3, 14.4, 14.6, 14.9, 15.1, 15.6, 16.2,
             16.9,
             17.4, 17.9, 18.3, 18.6, 18.9, 19.3, 19.5, 19.9, 20, 20.1, 20.5, 20.9, 21.2, 21.5, 21.9, 22.1, 22.4, 22.7,
             22.9,
             23.2, 23.7, 24.1, 23.9, 23.9, 24.1, 24.6, 25.2, 25.8, 26.6, 27.8, 29.1, 30.2, 31.1, 31.9, 32.4, 32.7, 33,
             33, 33,
             33.3]
    lons1 = [162.2, 160.7, 159.8, 159.3, 158.8, 158.2, 156.8, 155.5, 154.4, 153.4, 152.1, 150.7, 149.6, 148.2, 146.8,
             146, 145,
             144, 143, 141.9, 140.7, 139.6, 138.3, 137.2, 136.2, 134.9, 133.7, 132.6, 131.5, 130.2, 129.2, 128.2, 126.8,
             125.8,
             125.3, 124.8, 124.3, 123.8, 123.3, 122.7, 121.5, 121, 120.5, 120, 119.2, 118.3, 117.6, 116.9, 116.4, 116.8,
             117.5,
             118.2, 119.6, 120.5, 121.4, 122.4, 123.3, 125, 126.7]
    ints1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 40, 50, 60, 70, 80, 85, 90, 95, 105, 115, 115, 115, 105, 100, 95, 95, 90,
             90, 90,
             90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 80, 70, 70, 70, 65, 55, 40, 35, 0, 0, 0, 0, 0, 0, 35, 35, 0, 0, 0]
    return lons1, lats1, ints1


# def get_intensity():
#     intensity = []
#     lons, lats = sample_data()
#     for i in range(len(lons)):
#         inte = random.randint(30, 120)
#         intensity.append(inte)
#     return intensity


def buid_best_track():
    global best_track
    lons, lats, intensities = sample_data()
    assert len(lons) == len(lats) == len(intensities)
    for lat, lon, inty in zip(lats, lons, intensities):
        if inty <= 33:
            best_track['TD'].append([lon, lat])
        elif inty <= 47:
            best_track['TS'].append([lon, lat])
        elif inty <= 63:
            best_track['STS'].append([lon, lat])
        elif inty <= 84:
            best_track['TY'].append([lon, lat])
        elif inty <= 104:
            best_track['STY'].append([lon, lat])
        else:
            best_track['SuperTY'].append([lon, lat])
    return best_track


def main():
    tiler = Stamen('terrain-background')
    mercator = tiler.crs
    lons, lats, ints = sample_data()
    track = sgeom.LineString(zip(lons, lats))
    best_track = buid_best_track()
    fig = plt.figure(figsize=(40, 20))
    ax = fig.add_subplot(1, 1, 1, projection=mercator)
    ax.set_extent([100, 180, 0, 40], crs=ccrs.PlateCarree())

    ax.add_image(tiler, 6)
    ax.add_geometries([track], ccrs.PlateCarree(),
                      facecolor='none', edgecolor='k', linewidth=0.6)
    for k, v in best_track.items():
        trace = list(zip(*v))
        if len(trace) == 0: continue
        catelon = trace[0]
        catelat = trace[1]
        ax.plot(catelon, catelat, markersize=5, marker='o', linestyle='', color=colors[k],
                transform=ccrs.PlateCarree(),
                label=k)

    ax.set_xticks([100, 110, 120, 130, 140, 150, 160, 170, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([0, 10, 20, 30, 40], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.coastlines('10m')
    ax.legend()  # add a legend
    plt.savefig('typhoon.png')  # 保存图片
    plt.show()


if __name__ == '__main__':
    main()
    # for k, v in buid_best_track().items():
    #     trace = list(zip(*v))
    #     print(trace)
    #     catelon = trace[0]
    #     catelat = trace[1]
    #     print(catelon)
