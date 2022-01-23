from libs import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def blur_array(n, size):
    """
    Computes a linear transformation A(x), that averages(blurs) values in A using window of size n
    Parameters
    ----------
    n : int
        size of blur window

    size: int
        size of vector on which A should apply

    Returns
    -------
    A : ndarray of shape (size, size)
        Matrix of linear transformation A(x)
    """
    A = np.zeros(shape=(size, size))

    for i in range(size):
        if i - n//2 > 0:
            A[i, i-n//2:i+n//2 + 1] = 1
        else:
            A[i, 0:i+n//2 + 1] = 1
        A[i] /= A[i].sum()
    return A



def gaussian_kernel(x1, x2, sigma=12):
    """

    Given two ndim arrays x1 and x2 computers their similarity using gaussian kernel.
    Parameters
    ----------
    x1 : ndarray of shape (x, )
    x2 : ndarray of shape (x, )
    sigma : int, default = 12
        Sigma is the variance hyperparameter in gaussian similarity kernel

    Returns
    -------

    """
    diff_vec = x1 - x2  # difference vector
    diff_len_sq = diff_vec.T@diff_vec # length of difference vector squared
    similarity = np.exp(-diff_len_sq / sigma**2)  # similarity between x1 and x2
    return similarity


def get_unique_topics(taxonomy):
    """
    Given a Taxonomy object extracts unique leaf names from it

    Parameters
    ----------
    taxonomy : Taxonomy object

    Returns
    -------
    topics_unique : list
        list of unique leaf names of the taxonomy
    topics_idx : list
        list of indices of leaves in order from topics_unique

    """
    topics_unique = []
    topics_idx = []
    added_topics = set()
    for leaf in taxonomy.leaves:
        if leaf.name not in added_topics:
            topics_unique.append(leaf.name)
            topics_idx.append(leaf.index)
            added_topics.append(leaf.name)

    return topics_unique, topics_idx


# def get_random_txt_subset(texts, n=5000, r_s=None):
#     '''
#
#     Parameters
#     ----------
#     texts
#     n
#     r_s : int or None, default=None
#         random_seed
#
#     Returns
#     -------
#
#     '''
#     np.random.seed(r_s)
#     idx = np.random.choice(range(len(texts)), n, replace=False, )
#     return texts[idx]


class Clusters:
    """
    This class is created for convenient use of extracted clusters. Operates under fuzzy membership matrix
    and provides different usefull things like saving printing and visualizing the given clusters.
    Parameters
    ----------
    membership_matrix : ndarray of shape (n_objects, n_clusters)
        matrix of extracted clusters
    topics : list or ndarray
        list of object names

    embedding: ndarray of shape (n_objects, emb_dim) or None
        If the clasterization algo provides some embedding in some space then this is that embedding

    """
    def __init__(self, membership_matrix, topics, indices, embeddings=None, random_state=None):
        self.embeddings = embeddings
        self.random_state = random_state

        self.mm = membership_matrix

        # self.mm = self.make_row_stochastic(self.mm)  # normalazing rows according with fuzzy partition
        self.mm = self.denoise(self.mm)  # deleting large(noisy) and too small clusters

        # self.mm = self.cut(self.mm)
        # self.mm = self.denoise(self.mm)

        self.n_clusters = self.mm.shape[1]

        self.labels = self.get_labels(self.mm)  # for further visualization

        self.topics = np.array(topics)
        self.indices = np.array(indices)

        self.n_topics, self.n_clusters = self.mm.shape


        self.colors = ['', 'red', 'peru', 'darkorange', 'yellow', 'lawngreen',
         'forestgreen', 'cyan', 'blue', 'indigo', 'magenta']

    def __iter__(self):
        for i in range(self.n_clusters):
            yield self.get_cl_dict(i)

    @staticmethod
    def make_row_stochastic(mm):
        """
        Normalizing rows to add up to 1.
        """
        z = mm.sum(axis=1)
        z[np.isclose(z, 0)] = 1.  # to escape 0 division
        return mm / z[:, None]


    @staticmethod
    def denoise(mm):
        """
        Deleting clusters with small or large number of elements
        """
        n_topics, n_clusters = mm.shape
        mm_denoised = []
        for cl_idx in range(n_clusters):
            cl_u = mm[:, cl_idx]
            if (cl_u > 0.5).sum() > 25 or (cl_u > 0.).sum() < 4:  # if there are more then 25 worthy elements
                pass
            else:
                mm_denoised.append(cl_u[:, None])

        return np.hstack(mm_denoised)

    def cut(self, mm):
        n_topics, n_clusters = mm.shape
        for cl_idx in range(n_clusters):
            cl_u = mm[:, cl_idx]
            # splitting cluster into three parts with low, mid and high memberships
            # or into 2 parts if cluster is crisp
            _n_cl = 2
            if len(np.unique(cl_u)) == 2:
                _n_cl = 2
            clf = KMeans(n_clusters=_n_cl, random_state=self.random_state)
            clf.fit(cl_u[:, None])
            idx = np.argmax(clf.cluster_centers_)  # taking part with high memberships
            mm[clf.labels_ != idx, cl_idx] = 0.
        return mm

    def get_labels(self, mm):
        """
        assigns label for each topic in membership matrix
        If topic has zero membership in all clusters, it gets label 0 as not clustered element
        """
        n_topics, n_clusters = mm.shape
        labels = np.zeros(n_topics).astype(int)
        for i in range(n_topics):
            if mm[i].sum() > 0:
                labels[i] = np.argmax(mm[i]) + 1
            else:
                labels[i] = 0
        return labels

    def show(self, idx):
        """
        Given the cluster idx printing it in convenient style
        """
        cl_u = self.mm[:, idx]

        cl = sorted(((t_idx, t, t_u) for t_idx, t, t_u in zip(self.indices, self.topics, cl_u)),
               key=lambda x: -x[-1])  # sorting by membership

        cl = list(filter(lambda x: x[2] > 0, cl))  # throwing away zero memberships
        max_idx_len = max(map(lambda x: len(x[0]), cl))
        max_name_len = max(map(lambda x: len(x[1]), cl))

        for (t_idx, t, t_u) in cl:
            s = f'{t_idx: <{max_idx_len}} :: {t: <{max_name_len}} :: {t_u:.2f}'
            print(s)

    def get_cl_dict(self, idx):
        cl_u = self.mm[:, idx]
        return dict(zip(self.topics[cl_u > 0], cl_u[cl_u > 0]))


    def plot_scatter(self):
        pca = PCA(n_components=2)
        pca_emb = pca.fit_transform(self.embeddings)
        # pca_emb = self.embeddings[:, :2]
        plt.figure(figsize=(10,30))
        plt.subplot(3, 1, 1)
        for i in range(0, self.n_clusters//3):
            color = self.colors[i+1]
            cl =  self.get_cl_dict(i)
            for key in cl.keys():
                topic = self.topics[self.topics == key][0]
                point = pca_emb[self.topics == key, :][0]
                x, y = point[0], point[1]
                plt.scatter(x, y, color=color)
                plt.text(x, y, topic)

        plt.subplot(3,1,2)
        for i in range(self.n_clusters//3, 2*self.n_clusters//3):
            color = self.colors[i+1]
            cl =  self.get_cl_dict(i)
            for key in cl.keys():
                topic = self.topics[self.topics == key][0]
                point = pca_emb[self.topics == key, :][0]
                x, y = point[0], point[1]
                plt.scatter(x, y, color=color)
                plt.text(x, y, topic)

        plt.subplot(3,1,3)
        for i in range(2*self.n_clusters//3, self.n_clusters):
            color = self.colors[i+1]
            cl =  self.get_cl_dict(i)
            for key in cl.keys():
                topic = self.topics[self.topics == key][0]
                point = pca_emb[self.topics == key, :][0]
                x, y = point[0], point[1]
                plt.scatter(x, y, color=color)
                plt.text(x, y, topic)