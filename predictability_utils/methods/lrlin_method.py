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


def run_lrlin(train_years, test_years, source_data, target_data, n_pca_x, n_pca_y, n_latents, idcs, n_epochs, lr,
              if_plot=False, map_shape=None, batch_size=None, weight_decay=0., weight_lasso=0.):

    T = source_data.shape[0]
    assert T == target_data.shape[0]
    idx_source_train, idx_target_train, idx_source_test, idx_target_test = idcs

    #Training: predict t2m for train data
    X = torch.tensor(source_data.reshape(T, -1)[idx_source_train,:].mean(axis=0))
    Y = torch.tensor(target_data.reshape(T, -1)[idx_target_train,:].mean(axis=0))
    
    # pca decomposition
    n_pca_x = np.min(Xd.shape) if n_pca_x is None else n_pca_x
    n_pca_y = np.min(Yd.shape) if n_pca_y is None else n_pca_y
    
    pca_x = PCA(n_components=n_pca_x, copy=True, whiten=False)
    pca_y = PCA(n_components=n_pca_y, copy=True, whiten=False)
    
    pca_x.fit(Xd)
    pca_y.fit(Yd)
    #print(f'yi: {pca_y.explained_variance_ratio_}'); print(f'xi: {pca_x.explained_variance_ratio_}')
    #print(f'sumy: {pca_y.explained_variance_ratio_.sum()}'); print(f'sumx: {pca_x.explained_variance_ratio_.sum()}')
    
    if n_pca_x <= np.min(Xd.shape):
        X, A = pca_x.transform(Xd), pca_x.components_  
    else:
        X, A = Xd, None
    if n_pca_y <= np.min(Yd.shape):
        Y, B = pca_y.transform(Yd), pca_y.components_
    else:
        Y, B = Yd, None

    # fit linear model
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    lrlm = LR_lin_method(n_latents=n_latents)
    loss_vals = lrlm.fit(X,Y, 
                         n_epochs=n_epochs, 
                         lr=lr, 
                         batch_size=batch_size, 
                         weight_decay=weight_decay,
                         weight_lasso=weight_lasso)

    #Forecasting: predict t2m for test data
    X_f = source_data.reshape(T, -1)[idx_source_test,:].mean(axis=0)
    X_f = pca_x.transform(X_f) if not A is None else X_f
    X_f = torch.tensor(X_f, dtype=torch.float32)
    
    out_pred = lrlm.predict(X_f)
    out_pred = pca_y.inverse_transform(out_pred) if not B is None else out_pred

    # evaluate prediction performance
    out_true = target_data.reshape(T, -1)[idx_target_test,:].mean(axis=0)
    anomaly_corrs = helpers.compute_anomaly_corrs(out_true, out_pred)
  
    # visualize anomaly correlations and loss curve during training
    if if_plot:

        viz.visualize_anomaly_corrs(anomaly_corrs.reshape(*map_shape))

        plt.semilogx(loss_vals[np.min((100, np.max((n_epochs-100, 0)))):])
        plt.title('loss curve')
        plt.show()

    params = {'U' : lrlm._U.detach().numpy(), 'V': lrlm._V.detach().numpy(), 
              'out_pred': out_pred, 'out_true': out_true, 'PCx': X, 'EOFx': A, 'PCy': Y, 'EOFy': B }

    return anomaly_corrs, params


class LR_lin_method():
    
    def __init__(self, n_latents, device=None):

        self._n_latents = n_latents
        self._U, self.V = None, None
        self._device = default_device() if device is None else device
        
    def fit(self, X, Y, lr, n_epochs, batch_size=None, weight_decay=0., weight_lasso=0.):

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
