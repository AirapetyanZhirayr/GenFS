import numpy as np


class LapInTransformer:
    """
    Computes graphs Laplacian, it's pseudo inverse, it's eigenvalues  and eigenvectors
    given similarity matrix W.


    Parameters
    ----------

    verbose : bool, default=False
        Controls the verbosity.

        Attributes
    ----------
    lam : 1-D ndarray
        All non-zero eigenvalues of normalized Laplacian

    Lam : 2-D ndarray
        Square diagonal matrix with Lam[i][i] = lam[i]

    n_zero_eigval : int
        Number of zero eigenvalues of Laplacian

    condition : ndarray[bool]
        Mask of non-zero eigenvalues

    V : 2-D ndarray of shape (n_objects, x)
        Matrix of eigenvectors of Laplacian with non-zero eigenvalues

    L : 2-D ndarray of shape (n_objects, n_objects)
        Laplacian Matrix

    Ln : 2-D ndarray of shape (n_objects, n_objects)
        Normalized Laplacian Matrix

    L_inv: 2-D ndarray of shape (n_objects, n_objects)
        Pseudo Inverse of Normalized Laplacian Matrix
   """

    def __init__(self, verbose=False):
        self.verbose = verbose

        self.lam = None
        self.condition = None
        self.n_zero_eigval = None
        self.Lam = None
        self.V = None
        self.L = None
        self.Ln = None
        self.L_inv = None

    def fit(self, W):
        """

        Parameters
        ----------
        W: ndarray,
            symmetric similarity matrix of objects

        """

        assert np.allclose(W, W.T), "similarity matrix is not symmetric"

        mat_dim = W.shape[0]  # number of objects
        Identity = np.eye(mat_dim)  # identity matrix

        D = np.diag(W.sum(axis=1))  # matrix of degrees of vertices
        assert np.alltrue(W.sum(axis=1) > 0.), "Similarity matrix W has a zero column"
        _D = np.linalg.inv(np.sqrt(D))

        Ln = Identity - _D@W@_D  # normalized Laplacian
        L = D - W  # unnormalized Laplacian

        lam, V = np.linalg.eigh(Ln)  # computing eigenvectors and eigenvalues
        condition = ~np.isclose(lam, 0, atol=1e-08)
        self.n_zero_eigval = np.sum(~condition)
        if self.verbose:
            print(f'Number of connected components: {self.n_zero_eigval}')

        # computing pseudo inverse of Laplacian
        self.condition = condition
        lam = lam[condition]
        V = V[:, condition]
        Lam = np.diag(lam)  # diagonal matrix of eigenvalues
        L_inv = V@np.linalg.inv(Lam)@V.T

        self.lam = lam
        self.condition = condition
        self.Lam = Lam
        self.V = V
        self.L = L
        self.Ln = Ln
        self.L_inv = L_inv

    def fit_transform(self, W):
        """

        Parameters
        -------
        W: ndarray,
            symmetric similarity matrix of objects

        Returns
        -------
        the Pseudo Inverse of Laplacian matrix
        """
        self.fit(W)
        return self.L_inv


# ZERO_BOUND = 10 ** (-8)
# ENTITY_BOUND = 10 ** (-4)

# def lapin(A):
#     '''
#     '''
#     A = (A + A.T) / 2
#     a_sums = np.ravel(abs(sum(A)))
#     checked = np.array(a_sums > ENTITY_BOUND)
#     is_correct = checked.all()
#
#     if not is_correct:
#         print('These entities are no good - remove them first!!!')
#         print([i for i, j in enumerate(checked, 1) if not j])
#         A = A[:, checked][checked, :]
#         a_sums = a_sums[checked]
#
#     matrix_dim, _ = A.shape
#     C = np.empty((matrix_dim, matrix_dim))
#     for i in range(matrix_dim):
#         for j in range(matrix_dim):
#             C[i, j] = A[i, j] / np.sqrt(a_sums[i] * a_sums[j])
#
#     eig_vals, eig_vecs = np.linalg.eig(np.eye(matrix_dim) - C)
#     eig_vals_diag = np.diag(eig_vals)
#     nonzero_cond = np.array(eig_vals > ZERO_BOUND)
#     nonzero_eig_vals_diag = eig_vals_diag[nonzero_cond, :][:, nonzero_cond]
#     nonzero_eig_vecs = eig_vecs[:, nonzero_cond]
#     B = nonzero_eig_vecs.dot(np.linalg.inv(nonzero_eig_vals_diag)).dot(nonzero_eig_vecs.T)
#
#     return B
