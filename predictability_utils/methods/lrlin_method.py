import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

from torch.utils import data
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler

from predictability_utils.utils import viz, helpers
import matplotlib.pyplot as plt


def default_device():
    """gets default device from a test tensor"""
    return torch.ones((1,)).device


def run_lrlin(source_data, target_data, n_latents, idcs, if_plot=False, map_shape=None,
             n_epochs=10000, lr=1e-4, batch_size=None, weight_decay=0., weight_lasso=0.):

    T = source_data.shape[0]
    assert T == target_data.shape[0]
    idx_source_train, idx_target_train, idx_source_test, idx_target_test = idcs

    # predict T2ms in Summer from soil moisture levels in Spring
    X = torch.tensor(source_data.reshape(T, -1)[idx_source_train,:].mean(axis=0))
    Y = torch.tensor(target_data.reshape(T, -1)[idx_target_train,:].mean(axis=0))

    # fit CCA-based model
    lrlm = LR_lin_method(n_latents=n_latents)
    loss_vals = lrlm.fit(X,Y, 
                         n_epochs=n_epochs, 
                         lr=lr, 
                         batch_size=batch_size, 
                         weight_decay=weight_decay,
                         weight_lasso=weight_lasso)

    # predict T2ms for test data (1951 - 2010)
    X_f = source_data.reshape(T, -1)[idx_source_test,:].mean(axis=0)
    out_pred = lrlm.predict(X_f)

    # evaluate prediction performance
    out_true = target_data.reshape(T, -1)[idx_target_test,:].mean(axis=0)
    anomaly_corrs = helpers.compute_anomaly_corrs(out_true, out_pred)

    # visualize anomaly correlations and loss curve during training
    if if_plot:

        viz.visualize_anomaly_corrs(anomaly_corrs.reshape(*map_shape))

        plt.semilogx(loss_vals[np.min((100, np.max((n_epochs-100, 0)))):])
        plt.title('loss curve')
        plt.show()

    params = {'U' : lrlm._U.detach().numpy(), 'V': lrlm._V.detach().numpy() }

    return anomaly_corrs, params


class LR_lin_method():
    
    def __init__(self, n_latents, device=None):

        self._n_latents = n_latents
        self._U, self.V = None, None
        self._device = default_device() if device is None else device
        
    def fit(self, X, Y, lr=1e-2, n_epochs=2000, batch_size=None, weight_decay=0., weight_lasso=0.):

        batch_size = X.shape[0] if batch_size is None else batch_size
        
        # initialize guesses for V,U from PCA of respective X,Y spaces
        V_init = PCA(n_components=self._n_latents).fit(X).components_
        U_init = PCA(n_components=self._n_latents).fit(Y).components_

        # loss function and gradients (loss just for tracking convergence of gradient descent)
        self._U = torch.tensor(U_init,   requires_grad=True, dtype=torch.float32)
        self._V = torch.tensor(V_init.T, requires_grad=True, dtype=torch.float32)

        loss_vals = np.zeros(n_epochs)
        """
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
        """
        
        dataset = data.TensorDataset(X, Y)
        parameters = (self._U, self._V)

        if weight_lasso > 0:
            print('applying lasso regularization')
            def lasso_reg():
                return weight_lasso * sum([parameter.norm(p=2) for parameter in parameters])
        else:
            def lasso_reg():
                return 0.

        # create minibatch loader using a subset sampler
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(np.arange(X.shape[0])),
        )
        optimizer = optim.Adam(parameters, 
                               lr=lr, 
                               weight_decay=weight_decay)

        epochs = 0
        for epoch in range(n_epochs):

            # Train for a single epoch.
            for batch in train_loader:
                optimizer.zero_grad()

                Ypred = torch.mm(torch.mm(batch[0].to(self._device), self._V), self._U) 
                sq_froebenius = F.mse_loss(Ypred, batch[1].to(self._device), reduction='none')
                sq_froebenius = torch.sum(sq_froebenius, axis=1)
                loss = sq_froebenius.mean() + lasso_reg()
                loss.backward()
                #clip_grad_norm_(parameters, max_norm=5.0)
                optimizer.step()

                loss_vals[epoch] = loss.detach().numpy()

            epochs += 1        

        return loss_vals

    def predict(self, X):

        Ypred = torch.mm(torch.mm(torch.tensor(X), self._V), self._U)

        return Ypred.detach().numpy()
