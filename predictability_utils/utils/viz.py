import numpy as np
import matplotlib.pyplot as plt

def visualize_anomaly_corrs(anomaly_corrs_map):

    cmax = np.max(np.abs(anomaly_corrs_map))
    plt.imshow(anomaly_corrs_map, cmap='seismic', vmin=-cmax, vmax=cmax)
    plt.colorbar()
    plt.title(f'anomaly correlation coefficient, avg: {anomaly_corrs_map.mean():.3f}')
    plt.show()

def visualize_example_preds(out_true, out_pred, map_shape, y_train, years):

    plt.figure(figsize=(16,8))
    for i,y in enumerate(years):
        plt.subplot(np.ceil(len(years)/2),2, i+1)
        plt.imshow(np.hstack((out_pred[y,:].reshape(*map_shape), 
                              out_true[y,:].reshape(*map_shape))), cmap='gray')
        plt.ylabel(str(1900 + y_train + y))
    plt.suptitle('example predictions (left: predicted, right: true)')
    plt.show()
    
    
def visualize_ACC(anomaly_corrs, lat, lon, map_shape, title):
    
    anomaly_corrs_map = anomaly_corrs.reshape(*map_shape)
    m = Basemap(projection='cyl', llcrnrlon=min(lon), llcrnrlat=min(lat),
                urcrnrlon=max(lon), urcrnrlat=max(lat))
    clevs = np.linspace(-1, 1, 21)
    lons, lats = m(*np.meshgrid(lon, lat))
    h = m.contourf(lons, lats, anomaly_corrs_map, clevs, cmap=plt.cm.RdBu_r)
    m.contour(lons, lats, anomaly_corrs_map, levels=[0.5], colors=['k'])
    m.drawcoastlines()
    m.drawparallels(np.arange(35.,70.,10.), labels=[1,0,0,1], fontsize=12)
    m.drawmeridians(np.arange(0.,25.,10.), labels=[1,0,0,1], fontsize=12)
    m.drawmapboundary(fill_color='white')
    col = m.colorbar(h, location='bottom', size='15%', pad="12%")
    col.ax.tick_params(labelsize=13, labelrotation=45)
    plt.title(title, fontsize=12)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
