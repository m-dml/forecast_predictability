import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

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
