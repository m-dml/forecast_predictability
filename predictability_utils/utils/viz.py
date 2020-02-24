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
