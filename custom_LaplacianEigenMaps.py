from libs import *
from sklearn.manifold import spectral_embedding


class LaplacianEigenMaps:

    def __init__(self, n_components, drop_first=True):
        self.drop_first = drop_first
        self.n_components = n_components

    def fit(self, W):
        W = (W + W.T)/2
        D = np.diag(W.sum(axis=0))
        L = D - W
        _D = np.linalg.inv(np.sqrt(D))
        Ln = _D@L@_D

        lam, V = np.linalg.eigh(Ln)
        start_idx = 1 if self.drop_first else 0
        lam = lam[start_idx: self.n_components+start_idx]
        V = V[:, start_idx:self.n_components+start_idx]
        self.embedding = V
        self.eigenvalues = lam
