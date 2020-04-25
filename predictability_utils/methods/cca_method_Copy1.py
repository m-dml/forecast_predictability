import numpy as np
from sklearn.cross_decomposition import CCA

from predictability_utils.utils import viz, helpers
import matplotlib.pyplot as plt


def run_cca(source_data, target_data, n_latents, idcs, train_months, test_months, train_years, test_years, if_plot=False, map_shape=None):
    
    T = source_data.shape[0]
    assert T == target_data.shape[0]
    idx_source_train, idx_target_train, idx_source_test, idx_target_test = idcs
    
    # predict T2ms in Summer from soil moisture levels in Spring (1900 - 1950)
    X = source_data.reshape(T, -1)[idx_source_train,:].reshape(len(train_months)*len(train_years),-1)
    Y1 = target_data.reshape(T, -1)[idx_target_train,:].mean(axis=0)
    Y = np.stack((Y1,Y1,Y1),axis=0).reshape(len(train_months)*len(train_years),-1)

    # fit CCA-based model
    ccam = CCA_method(n_latents=n_latents)
    ccam.fit(X,Y)

    # predict T2ms for test data (1951 - 2010)
    X_f = source_data.reshape(T, -1)[idx_source_test,:].reshape(len(test_months)*len(test_years),-1)
    out_pred1 = ccam.predict(X_f)
    out_pred = out_pred1.reshape(len(test_months),len(test_years),-1).mean(axis=0)
    
    # evaluate prediction performance
    out_true = target_data.reshape(T, -1)[idx_target_test,:].mean(axis=0)
    anomaly_corrs = helpers.compute_anomaly_corrs(out_true, out_pred)

    # visualize anomaly correlations
    if if_plot:
        viz.visualize_anomaly_corrs(anomaly_corrs.reshape(*map_shape))

    params = {'U' : ccam._cca.y_loadings_, 'V': ccam._cca.x_rotations_, 'Q': ccam._Q, 
              'out_pred' : out_pred, 'out_true' : out_true }

    return anomaly_corrs, params

class CCA_method():

    def __init__(self, n_latents):

        self._n_latents = n_latents
        self._cca = CCA(n_components=n_latents, 
                        scale=False, max_iter=10000, tol=1e-8)
        self._Q = np.eye(self._n_latents)

    def fit(self, X, Y):

        # projections U'X, V'Y such that U'X and V'Y are maximally correlated
        self._cca.fit(X, Y) 

        # get time-course of projected data
        UX, VY = self._cca.transform(X, Y)

        # learn linear regression VY = UX * Q 
        # (Q will be optimal in least-squares sense)
        self._Q = np.linalg.pinv(UX).dot(VY) 

    def predict(self, X):

        # transform source data into latent space
        UX = self._cca.transform(X)

        # predict latent activity in target space
        QUX = UX.dot(self._Q)

        # predict observed activity in target space
        Ypred = QUX.dot(self._cca.y_loadings_.T)

        return Ypred
