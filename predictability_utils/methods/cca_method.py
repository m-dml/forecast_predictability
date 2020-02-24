import numpy as np
from sklearn.cross_decomposition import CCA

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
