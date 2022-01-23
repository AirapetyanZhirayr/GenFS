''' FADDIS clustering implementation in Python
'''

import numpy as np
import numpy.linalg as LA
from numpy.linalg import eig, norm

ZERO_BOUND = 10 ** (-8)
MIN_CLUSTER_CONTRIBUTION = 3 * 10 ** (-3)
EPSILON = 5 * 10 ** (-2) 
# Maximum number of clusters
MAX_NUM_CLUSTERS = 15

ITERATION = 0


def ensure_np_matrix(A):
    if not isinstance(A, np.matrix):
        A = np.matrix(A)
    return A


def get_intensity(v, W):
    if v.T @ v > 0:
        return v.T @ W @ v/(v.T@v)**2
    return 0.


class FADDIS:

    def __init__(self, ZERO_BOUND=ZERO_BOUND, MAX_NUM_CLUSTERS=15, EPSILON=EPSILON,
                 MIN_CLUSTER_CONTRIBUTION=MIN_CLUSTER_CONTRIBUTION):
        self.ZERO_BOUND = ZERO_BOUND
        self.MAX_NUM_CLUSTERS = MAX_NUM_CLUSTERS
        self.MIN_CLUSTER_CONTRIBUTION = MIN_CLUSTER_CONTRIBUTION
        self.EPSILON = EPSILON

    def fit(self, A):
        n_elements = len(A)
        self.contributions = np.array([])
        self.eigvalues_sequence = np.array([])
        self.intensities = np.empty((0, 2))
        self.n_clusters=0
        W = A.copy()
        W = (W + W.T) / 2  # ensure W is symmetric
        self.matrix_sequence = [W]
        self.scatter = (W**2).sum()
        self.cont_is_positive = True
        curr_cont = 1
        self.res_cont = 1
        self.clusters = []
        curr_itteration = 0
        while ((self.n_clusters < self.MAX_NUM_CLUSTERS) and
               (curr_cont > self.MIN_CLUSTER_CONTRIBUTION) and
               (self.res_cont > self.EPSILON) and
               (self.cont_is_positive)):

            lam, V = np.linalg.eig(W)

            # we care only about eigvectors with positive eigvalues
            pos_eigv_mask = (lam > self.ZERO_BOUND)
            lam = lam[pos_eigv_mask]

            V = V[:, pos_eigv_mask]
            n_eig_vals_pos = len(lam)

            curr_intensities = np.zeros((n_eig_vals_pos, ))
            curr_memberships = np.zeros((n_elements, n_eig_vals_pos))

            for j in range(n_eig_vals_pos):

                v_j = V[:, j]

                v_j = v_j / (v_j.T@v_j)**2  # in case it comes unnormalized

                u_j = v_j.clip(0)

                if LA.norm(u_j) > 0:  # to escape ZERODEVISION
                    u_j = u_j / LA.norm(u_j)

                intensity_j = get_intensity(u_j, W)

                # checking alternative cluster
                v_j_alt = (-v_j).copy()

                u_j_alt = v_j_alt.clip(0)

                if LA.norm(u_j_alt) > 0:
                    # u_j_alt = u_j_alt / np.sqrt(u_j_alt.T@u_j_alt)
                    u_j_alt = u_j_alt / LA.norm(u_j_alt)

                intensity_j_alt = get_intensity(u_j_alt, W)

                if intensity_j > intensity_j_alt:
                    curr_intensities[j] = intensity_j
                    curr_memberships[:, j] = u_j.ravel()
                else:
                    curr_intensities[j] = intensity_j_alt
                    curr_memberships[:, j] = u_j_alt.ravel()
            contrib_max, contrib_max_index = curr_intensities.max(), curr_intensities.argmax()

            if contrib_max > self.ZERO_BOUND:
                self.eigvalues_sequence = np.append(self.eigvalues_sequence, lam[contrib_max_index])
                self.intensities = np.append(self.intensities, np.matrix([np.sqrt(contrib_max),
                                                                contrib_max]), axis=0)

                u = curr_memberships[:, contrib_max_index]

                explained_scatter = ((u.T @ W @ u) / (u.T @ u)) ** 2
                curr_cont = explained_scatter / self.scatter
                self.contributions = np.append(self.contributions, curr_cont)
                self.res_cont -= curr_cont

                self.clusters.append(u)

                W -= contrib_max * u[:, None]@u[:, None].T
                W = (W + W.T) / 2
                self.matrix_sequence.append(W)

                self.n_clusters += 1
            else:
                self.cont_is_positive = True

            curr_itteration += 1

        self.clusters = np.vstack(self.clusters)  # one row -- one cluster
        self.mm = self.clusters.T.copy()


        if not self.cont_is_positive:
            print('No positive weights at spectral clusters')
        elif curr_cont < self.MIN_CLUSTER_CONTRIBUTION:
            print('Cluster contribution is too small')
        elif self.res_cont < self.EPSILON:
            print('Residual is too small')
        elif self.n_clusters > self.MAX_NUM_CLUSTERS:
            print('Maximum number of clusters reached')




def faddis(A):

    # A = ensure_np_matrix(A)

    # minimum cluster's relative contribution to the data scatter
    min_cont = MIN_CLUSTER_CONTRIBUTION
    # minimum relative residual data scatter
    eps = EPSILON
    # maximum number of clusters
    max_clust_num = MAX_NUM_CLUSTERS

    is_positive = True
    matrix_dim, _ = A.shape

    sc = np.power(A, 2)
    # Total data scatter
    scatter = np.sum(sc)

    cluster_counter = 0
    membership_matrix = np.empty((matrix_dim, 0))
    contributions = np.array([])
    final_eigvalues = np.array([])
    intensities = np.empty((0, 2))
    curr_cont = 1
    res_cont = 1

    # 'zero' and 'one' vectors for comparisons
    zeros_vect = np.zeros((matrix_dim, ))
    ones_vect = np.ones((matrix_dim, ))

    # ensure matrix is symmetrical
    W = (A + A.T) / 2
    matrix_sequence = [W]
    curr_itteration = 0
    while is_positive and curr_cont > min_cont and res_cont > eps and cluster_counter <= max_clust_num:
        eig_vals, eig_vecs = LA.eig(W)

        eig_vals_pos = np.argwhere(eig_vals > ZERO_BOUND).ravel()
        n_eig_vals_pos = eig_vals_pos.size

        curr_intensities = np.zeros((n_eig_vals_pos, ))
        curr_memberships = np.zeros((matrix_dim, n_eig_vals_pos))

        # we iterate through positive eig_vals to find proj(eig_vec) that maximizes contribution
        for k in range(n_eig_vals_pos):
            v_k = eig_vecs[:, eig_vals_pos[k]]

            # v_k --> u_k (projection onto non-negative real)
            u_k = np.maximum(zeros_vect, v_k)
            u_k = np.minimum(u_k, ones_vect)  # unnecessary

            if LA.norm(u_k) > 0:
                u_k = u_k / LA.norm(u_k)
            u_k = np.squeeze(np.asarray(u_k))
            intensity_k = get_intensity(u_k, W)

            v_k_alt = (-v_k).copy()

            u_k_alt = np.maximum(zeros_vect, v_k_alt)
            u_k_alt = np.minimum(u_k_alt, ones_vect)

            if LA.norm(u_k_alt) > 0:
                u_k_alt = u_k_alt / LA.norm(u_k_alt)
            u_k_alt = np.squeeze(np.asarray(u_k_alt))

            intensity_k_alt = get_intensity(u_k_alt, W)


            if intensity_k > intensity_k:
                curr_intensities[k] = intensity_k
                curr_memberships[:, k] = u_k.ravel()
            else:
                curr_intensities[k] = intensity_k_alt
                curr_memberships[:, k] = u_k_alt.ravel()

        contrib_max, contrib_max_index = curr_intensities.max(), curr_intensities.argmax()
        if contrib_max > ZERO_BOUND:
            final_eigvalues = np.append(final_eigvalues, eig_vals[eig_vals_pos[contrib_max_index]])
            intensities = np.append(intensities, np.matrix([np.sqrt(contrib_max),
                                                            contrib_max]), axis=0)

            u = curr_memberships[:, contrib_max_index]

            explained_scatter = ((u.T @ W @ u) / (u.T @ u)) ** 2
            curr_cont = explained_scatter / scatter
            contributions = np.append(contributions, curr_cont)
            res_cont -= curr_cont

            membership_matrix = np.append(membership_matrix, np.matrix(u).T, axis=1)

            W -= contrib_max * u[:, None]@u[:, None].T

            W = (W + W.T) / 2
            matrix_sequence.append(W)

            cluster_counter += 1
        else:
            is_positive = False
        curr_itteration += 1

    if not is_positive:
        print('No positive weights at spectral clusters')
    elif curr_cont < min_cont:
        print('Cluster contribution is too small')
    elif res_cont < eps:
        print('Residual is too small')
    elif cluster_counter > max_clust_num:
        print('Maximum number of clusters reached')

    return matrix_sequence, membership_matrix, contributions, intensities, final_eigvalues, cluster_counter








