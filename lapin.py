import numpy as np

class LapInTransformer:
    def __init__(self):
        pass

    def fit(self, W):

        mat_dim = W.shape[0]
        I = np.eye(mat_dim)
        D = np.diag(W.sum(axis=1))
        _D = np.linalg.inv(np.sqrt(D))  # _D = D**(-1/2)

        Ln = I - _D@W@_D

        lam, V = np.linalg.eig(Ln)
        condition = ~np.isclose(lam, 0, atol=1e-08  )

        lam = lam[condition]
        V = V[:, condition]

        Lam = np.diag(lam)

        L_inv = V@np.linalg.inv(Lam)@V.T

        self.lam = lam
        self.condition = condition
        self.Lam = Lam
        self.V = V
        self.Ln = Ln
        self.L_inv = L_inv

    def fit_transform(self, W):
        self.fit(W)
        return self.L_inv

ZERO_BOUND = 10 ** (-8)
ENTITY_BOUND = 10 ** (-4)



def lapin(A):
    '''
    '''
    A = (A + A.T) / 2
    a_sums = np.ravel(abs(sum(A)))
    checked = np.array(a_sums > ENTITY_BOUND)
    is_correct = checked.all()

    if not is_correct:
        print('These entities are no good - remove them first!!!')
        print([i for i, j in enumerate(checked, 1) if not j])
        A = A[:, checked][checked, :]
        a_sums = a_sums[checked]

    matrix_dim, _ = A.shape
    C = np.empty((matrix_dim, matrix_dim))
    for i in range(matrix_dim):
        for j in range(matrix_dim):
            C[i, j] = A[i, j] / np.sqrt(a_sums[i] * a_sums[j])

    eig_vals, eig_vecs = np.linalg.eig(np.eye(matrix_dim) - C)
    eig_vals_diag = np.diag(eig_vals)
    nonzero_cond = np.array(eig_vals > ZERO_BOUND)
    nonzero_eig_vals_diag = eig_vals_diag[nonzero_cond, :][:, nonzero_cond]
    nonzero_eig_vecs = eig_vecs[:, nonzero_cond]
    B = nonzero_eig_vecs.dot(np.linalg.inv(nonzero_eig_vals_diag)).dot(nonzero_eig_vecs.T)

    return B