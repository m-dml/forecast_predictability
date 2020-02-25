import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

from predictability_utils.utils import viz, helpers
import matplotlib.pyplot as plt

def run_lrlin(source_data, target_data, n_latents, idcs, if_plot=False, map_shape=None):

    T = source_data.shape[0]
    assert T == target_data.shape[0]
    idx_source_train, idx_target_train, idx_source_test, idx_target_test = idcs

    # predict T2ms in Summer from soil moisture levels in Spring
    X = torch.tensor(source_data.reshape(T, -1)[idx_source_train,:].mean(axis=0))
    Y = torch.tensor(target_data.reshape(T, -1)[idx_target_train,:].mean(axis=0))

    # fit CCA-based model
    lrlm = LR_lin_method(n_latents=n_latents)
    loss_vals = lrlm.fit(X,Y, n_epochs=100000, lr=1e-5)

    # predict T2ms for test data (1951 - 2010)
    X_f = source_data.reshape(T, -1)[idx_source_test,:].mean(axis=0)
    out_pred = lrlm.predict(X_f)

    # evaluate prediction performance
    out_true = target_data.reshape(T, -1)[idx_target_test,:].mean(axis=0)
    anomaly_corrs = helpers.compute_anomaly_corrs(out_true, out_pred)

    # visualize anomaly correlations and loss curve during training
    if if_plot:

        viz.visualize_anomaly_corrs(anomaly_corrs.reshape(*map_shape))

        plt.semilogx(loss_vals[100:])
        plt.title('loss curve')
        plt.show()

    params = {'U' : lrlm._U.detach().numpy(), 'V': lrlm._V.detach().numpy() }

    return anomaly_corrs, params


class LR_lin_method():
    
    def __init__(self, n_latents):

        self._n_latents = n_latents
        self._U, self.V = None, None

    def fit(self, X, Y, lr=1e-2, n_epochs=2000):
        
        # initialize guesses for V,U from PCA of respective X,Y spaces
        V_init = PCA(n_components=self._n_latents).fit(X).components_
        U_init = PCA(n_components=self._n_latents).fit(Y).components_

        # loss function and gradients (loss just for tracking convergence of gradient descent)
        self._U = torch.tensor(U_init,   requires_grad=True, dtype=torch.float32)
        self._V = torch.tensor(V_init.T, requires_grad=True, dtype=torch.float32)

        loss_vals = np.zeros(n_epochs)
        for epoch in range(n_epochs):

            Ypred = torch.mm(torch.mm(X, self._V), self._U) 
            sq_froebenius = F.mse_loss(Ypred, Y, reduction='none')
            sq_froebenius = torch.sum(sq_froebenius, axis=1)
            loss = sq_froebenius.mean()
            loss.backward()

            loss_vals[epoch] = loss.detach().numpy()
            with torch.no_grad():
                self._U -= lr * self._U.grad
                self._V -= lr * self._V.grad

            self._U.grad.zero_()
            self._V.grad.zero_()

        return loss_vals

    def predict(self, X):

        Ypred = torch.mm(torch.mm(torch.tensor(X), self._V), self._U)

        return Ypred.detach().numpy()
