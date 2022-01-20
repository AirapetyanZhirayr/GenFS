import numpy as np
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
from fcmeans import FCM
from libs import *
import utils

class LaplacianFCM:
    """Laplacian Fuzzy C-Means clustering.

    Parameters
    ----------
    topics : list or ndarray of strings
        The string-characteristics of objects to classify

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate in Fuzzy C-Means

    embedding_dim : int, default=8
        The number of eigenvectors of Laplacian Matrix to use
        as embedding

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.


    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

   """
    def __init__(self,
                 # topics,
                 n_clusters=8, embedding_dim=8, random_state=None, m=1.4):
        # self.topics = np.array(topics)
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)
        self.m = m

    def fit(self, sim_mat):
        """Compute Laplacian FCM clustering.

        Parameters
        ----------
        sim_mat : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.


        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.embedding = spectral_embedding(sim_mat, drop_first=True,
                           n_components=self.embedding_dim, norm_laplacian=True,
                                            random_state=self.random_state)


        self.kmeans = KMeans(n_clusters=self.n_clusters,
            random_state=self.random_state)
        #
        self.kmeans.fit(self.embedding)
        self.kmeans.u = self.create_membership_matrix(self.kmeans.labels_)
        # self.kmeans.u = utils.Clusters(membership_matrix=self.kmeans.u.copy(),
        #                                topics=self.topics,
        #                                random_state=self.random_state)

        self.fcm = FCM(n_clusters=self.n_clusters, max_iter=400, m=self.m,
                       random_state=self.random_state)

        self.fcm.fit(self.embedding)

        # self.membership_matrix = self.fcm.u.copy()

        # self.fcm.u = utils.Clusters(membership_matrix=self.fcm.u.copy(),
        #                             topics=self.topics,
        #                             random_state=self.random_state)
        return self


    def create_membership_matrix(self, labels):
        n_entities = len(labels)
        n_cl = np.max(labels) + 1
        u = np.zeros(shape=(n_entities, n_cl))
        for i in range(n_cl):
            cl_mask = (labels == i)
            u[cl_mask, i] = 1.
        return u

if __name__ == "__main__":
    sim_mat = load_obj(
        name='input_data/similarity_matrices/TFIDF/sim_mat_TFIDF_keywords_enh_std.pkl'
    )
    # topics_unique = load_obj(name='input_data/taxonomies/unique_topics')
    clf = LaplacianFCM(n_clusters=9, embedding_dim=8,
                       # topics=topics_unique,
                       random_state=1)
    clf.fit(sim_mat)

