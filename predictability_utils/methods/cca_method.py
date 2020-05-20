import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

from predictability_utils.utils import viz, helpers
import matplotlib.pyplot as plt

def run_cca(source_data, target_data, n_latents, idcs, n_pca_x=None, n_pca_y=None, if_plot=False, map_shape=None):

    T = source_data.shape[0]
    assert T == target_data.shape[0]
    idx_source_train, idx_target_train, idx_source_test, idx_target_test = idcs

    # predict T2ms in Summer from soil moisture levels in Spring (1900 - 1950)
    X = source_data.reshape(T, -1)[idx_source_train,:].mean(axis=0)
    Y = target_data.reshape(T, -1)[idx_target_train,:].mean(axis=0)

    n_pca_x = np.min(X.shape) if n_pca_x is None else n_pca_x
    n_pca_y = np.min(Y.shape) if n_pca_y is None else n_pca_y
    
    pca_x = PCA(n_components=n_pca_x, copy=True, whiten=False)
    pca_y = PCA(n_components=n_pca_y, copy=True, whiten=False)
    
    pca_x.fit(X)
    pca_y.fit(Y)
    
    if n_pca_x < np.min(X.shape):
        X, A = pca_x.transform(X), pca_x.components_  
    else:
        X, A = X, None
    if n_pca_y < np.min(Y.shape):
        Y, B = pca_y.transform(Y), pca_y.components_
    else:
        Y, B = Y, None
    
    # fit CCA-based model
    ccam = CCA_method(n_latents=n_latents)
    ccam.fit(X,Y)

    # predict T2ms for test data (1951 - 2010)
    X_f = source_data.reshape(T, -1)[idx_source_test,:].mean(axis=0)
    X_f = pca_x.transform(X_f) if not A is None else X_f
    
    out_pred = ccam.predict(X_f)
    out_pred = pca_y.inverse_transform(out_pred) if not B is None else out_pred

    # evaluate prediction performance
    out_true = target_data.reshape(T, -1)[idx_target_test,:].mean(axis=0)
    anomaly_corrs = helpers.compute_anomaly_corrs(out_true, out_pred)

    # visualize anomaly correlations
    if if_plot:
        viz.visualize_anomaly_corrs(anomaly_corrs.reshape(*map_shape))

    params = {'U' : ccam._cca.y_loadings_, 'V': ccam._cca.x_rotations_, 'Q': ccam._Q, 
              'out_pred' : out_pred, 'out_true' : out_true, 'A': A, 'B' : B}

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
