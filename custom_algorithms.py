"""
Custom realization of some algorithms
"""
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
from fcmeans import FCM
from libs import *
import text_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from utils import blur_array
nltk_stop_words = tuple(stopwords.words('english'))


class FSC:
    """
    Fuzzy Spectral Clustering Algorithm
    Idea ::
    Given a symmetric similarity matrix S between objects, the inverse normalized
    Laplacian L_inv is computed. The first k (not a parameter) eigenvectors and eigenvalues
    of L_inv are chosen to approximate L_inv with the specified accuracy mat_diff (parameter),
    in the sense that ||L_inv - L_inv_approx|| < mat_diff, where matrix norm is
    the Frobenius norm divided by the number of objects squared.

    Each of those k eigenvectors are used to extract 2 clusters. One that consists of positive
    entries of eigenvector and one of negative entries. Entries that are close to 0 are left
    unclassified. If we sort the eigenvector by it's values, the locations of unclassified
    entries are the locations where we are flat, or in other words the derivative is close to 0.
    This happens because unclassified objects get nearly the same values in eigenvectors of
    Laplacian.

        Parameters
    ----------
    mat_diff : float, default=0.003
        The max. acceptable value of norm of difference matrix between the approximated L_inv
        and true one. The matrix norm is the Frobenius norm divided by number of objects squared.

    """
    def __init__(self, mat_diff=0.003, derivative_scale=0.001):

        self.mat_diff = mat_diff
        self.eps = derivative_scale

        self.sim_mat = None
        self.lapin = LapInTransformer()

        self.L_inv = None  # inverse Laplacian
        self.k = None  # number of considered eigenvalues
        self.diff_array = None  # list of approximation differences., the index is the rank-1 of the approximation
        self.eigenvectors = None  # ndarray of first k eigenvectors of Laplacian
        self.eigenvalues = None # ndarray of all eigenvalues of Laplacian

    def fit(self, sim_mat):
        self.sim_mat = sim_mat
        self.lapin.fit(self.sim_mat)  # computing inverse Laplacian
        self.L_inv = self.lapin.L_inv

        lam = self.lapin.lam  # eigenvalues of Laplacian
        self.eigenvalues = lam
        V = self.lapin.V  # eigenvectors of Laplacian
        self.k = self.get_n_clusters(lam, V)

        V_k = V[:, :self.k]  # first k eigenvalues of Laplacian
        self.eigenvectors = V_k
        # lam_k = lam[:self.k] # first k eigenvectors of Laplacian

        self.mm = self.extract_clusters(V_k)  # membership matrix

    def get_n_clusters(self, lam, V):
        diff = np.inf
        k = 0
        self.diff_array = []
        while diff > self.mat_diff:
            k += 1

            lam_k = np.diag(lam[:k])  # first k eigvalues of Laplacian
            lam_k_inv = np.linalg.pinv(lam_k)  # largest k eigvalues of inverse Laplacian
            L_inv_approx = V[:, :k]@lam_k_inv@V[:, :k].T

            # diff = np.linalg.norm(L_inv_approx-self.L_inv, ord='fro')
            diff = ((L_inv_approx - self.L_inv)**2).mean()
            self.diff_array.append(diff)

        return k

    def extract_clusters(self, V):
        n_objects, emb_dim = V.shape
        blur_A = blur_array(5, n_objects-1)  # -1 because of diff
        mm = []  # membership matrix
        for i in range(emb_dim):
            v = V[:, i]
            argsort = np.argsort(v)
            _v = v[argsort]
            diff = blur_A @ np.diff(_v)  # last point has no diff
            diff = np.append(diff, diff[-1])  # to account for last point

            # bringing the order back
            diff = diff[np.argsort(argsort)]

            d_mask = (diff > self.eps)

            pos_class = d_mask & (v > 0)
            neg_class = d_mask & (v < 0)

            v_pos = v.copy()
            v_pos[~pos_class] = 0.
            v_neg = v.copy()
            v_neg[~neg_class] = 0.
            v_neg = - v_neg

            mm.append(v_pos)
            mm.append(v_neg)

        return np.c_[mm].T


class LaplacianFCM:
    """Laplacian Fuzzy C-Means clustering.
    Given a symmetric similarity matrix S between objects, objects are
    embedded into the space of first n eigen vectors of normalized Laplacian.
    Those embedded vectors are classified using Fuzzy C-Means

    Parameters
    ----------
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

        self.embedding = None
        self.kmeans = None
        self.fcm = None
        self.mm = None  # membership matrix

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
                                            n_components=self.embedding_dim,
                                            norm_laplacian=True,
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

        self.mm = self.fcm.u.copy()

        # self.fcm.u = utils.Clusters(membership_matrix=self.fcm.u.copy(),
        #                             topics=self.topics,
        #                             random_state=self.random_state)
        return self

    @staticmethod
    def create_membership_matrix(labels):
        n_entities = len(labels)
        n_cl = np.max(labels) + 1
        u = np.zeros(shape=(n_entities, n_cl))
        for i in range(n_cl):
            cl_mask = (labels == i)
            u[cl_mask, i] = 1.
        return u


class LaplacianEigenMaps:
    """
    This class implements Laplacian EigenMaps algorithm available in sklearn as
    sklearn.manifold.spectral_embedding.
    This toy implementations is made just for understanding the steps of the algo.

    Parameters
    ----------
    n_components: int, default=8
        embedding dimension., number of eigenvectors of Laplacian matrix
        used for embedding

    drop_first: bool, default=True
        To use the first eigenvector with zero eigenvalue or not

    Attributes
    -----------
    embedding : ndarray of shape (n_objects, n_components)
        matrix of first eigenvectors of Normalized Laplacian

    eigenvalues: ndarray of shape (n_components, )
        vector of first eigenvalues of Normalized Laplacian
    """
    def __init__(self, n_components=8, drop_first=True):
        self.drop_first = drop_first
        self.n_components = n_components

        self.embedding = None
        self.eigenvalues = None

    def fit(self, W):
        assert np.allclose(W, W.T), "similarity matrix is not symmetric"

        D = np.diag(W.sum(axis=0))  # matrix of degrees of vertices
        L = D - W  # Laplacian matrix
        _D = np.linalg.inv(np.sqrt(D))
        Ln = _D@L@_D  # normalized Laplacian matrix

        lam, V = np.linalg.eigh(Ln)  # computing eigenvectors and eigenvalues of normalized Laplacian
        start_idx = 1 if self.drop_first else 0
        lam = lam[start_idx: self.n_components+start_idx]
        V = V[:, start_idx:self.n_components+start_idx]
        self.embedding = V
        self.eigenvalues = lam


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


class TFIDF:
    """Custom TFIDF Vectorizer

    Given a collection of topics (key words) and a collection of texts
        returns matrix of tfidf scores of each topic to each text

    Parameters
    ----------
    stop_words: list or set or tuple of strings
        The list of words not to account during vectorization

    aggregation_F: {'max', 'min', 'mean'}, default='max'
        Type of function to use during aggregation of several tokens score


    verbose : bool, default=False
        Controls the verbosity.


    """

    def __init__(self, stop_words=nltk_stop_words, aggregation_F='max', verbose=False):
        self.stop_words = set(stop_words)
        self.aggregation_F = aggregation_F
        self.verbose = verbose

        self.vocabulary = set()
        self.normal_form_dict = {}
        self.topics_preproc = []
        self.texts_preproc = []
        self.vectorizer = None
        self.tokens = None

        # matrices
        self.sim_mat = None
        self.rel_mat = None
        self.text2token = None
        self.text_weights = None  # vector

    def fit(self, texts, topics):

        # preprocessing
        topics_preproc = self.preprocess_topics(topics, self.stop_words)
        texts_preproc = self.preprocess_texts(texts)
        vocabulary = self.get_vocabulary(topics_preproc)
        normal_form_dict = self.get_normal_form(vocabulary)
        vocabulary = self.normalize_vocabulary(vocabulary, normal_form_dict)
        topics_preproc = self.normalize_topics(topics_preproc, normal_form_dict)
        texts_preproc = self.normalize_texts(texts_preproc, normal_form_dict)

        self.vocabulary = vocabulary
        self.topics_preproc = topics_preproc
        self.texts_preproc = texts_preproc
        self.normal_form_dict = normal_form_dict

        vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        self.vectorizer = vectorizer
        if self.verbose:
            print('BUILDING TEXT2TOKEN RELEVANCE MATRIX')
        text2token = vectorizer.fit_transform(texts_preproc).toarray()  # relevance matrix of texts to tokens

        # adding some random noise (this is done to increase rank of A.T@A used in other computations)
        text2token = text2token + np.random.uniform(0.000001, 0.000005, size=text2token.shape)
        text2token = np.clip(text2token, 0., 1.)
        self.text2token = text2token
        if self.verbose:
            print('AGGREGATING TEXT2TOKEN --> TEXT2TOPIC (MAIN RELEVANCE MATRIX)')

        # aggregating topic's tokens scores into one topics score
        text2topic = self.aggregate_scores(text2token, vectorizer, topics_preproc)
        self.rel_mat = text2topic

        self.tokens = pd.Series(self.vectorizer.get_feature_names_out())
        # self.build_similarity_matrix(topics_preproc)

    @staticmethod
    def preprocess_topics(topics, stop_words):
        """
        Deleting stop-words and short words from topics
        Moving topics to lowercase

        Returns
        -------
        topics_preproc : list[set]
        """
        topics_preproc = []
        n_topics = len(topics)
        for i in range(n_topics):
            topic = set()
            for token in topics[i].split():
                if len(token) > 2 and token not in stop_words:
                    topic.add(token.lower())
            topics_preproc.append(topic)
        return topics_preproc

    @staticmethod
    def preprocess_texts(texts):
        """Moving texts to lowercase"""
        texts_preproc = []
        n_texts = len(texts)
        for i in range(n_texts):
            text = texts[i].lower()
            texts_preproc.append(text)
        return texts_preproc

    @staticmethod
    def get_vocabulary(topics):
        """Returns union set of tokens from all topics"""
        vocab = set()
        for topic_tokens in topics:
            vocab.update(topic_tokens)
        return vocab

    @staticmethod
    def get_normal_form(vocabulary):
        """Creates and returns a dictionary of words that are not in normal form
            Example : dict[dogs] = dog"""
        normal_form_dict = {}
        vocabulary = list(vocabulary)
        for i in range(len(vocabulary)):
            t1 = vocabulary[i]
            for j in range(len(vocabulary)):
                t2 = vocabulary[j]
                if ((t1 + 's') == t2 or ((t1 + 'es') == t2) or
                        ((t1 + 'ing') == t2) or ((t1[:-1] + 'ing') == t2)):
                    normal_form_dict[t2] = t1

        return normal_form_dict

    @staticmethod
    def normalize_vocabulary(vocabulary, normal_form_dict):
        """Given a dict of normal forms returns normalized vocabulary"""
        vocabulary_norm = set()
        for token in vocabulary:
            if token in normal_form_dict:
                token = normal_form_dict[token]
            vocabulary_norm.add(token)
        return vocabulary_norm

    @staticmethod
    def normalize_topics(topics, normal_form_dict):
        """Given a dict of normal forms returns normalized topics """
        topics_norm = []
        for topic_tokens in topics:
            topic_tokens_norm = set()
            for token in topic_tokens:
                if token in normal_form_dict:
                    token = normal_form_dict[token]
                topic_tokens_norm.add(token)
            topics_norm.append(topic_tokens_norm)
        return topics_norm

    @staticmethod
    def normalize_texts(texts, normal_form_dict):
        """Given a dict of normal forms returns normalized texts"""
        texts_norm = []
        for text in texts:
            text_tokens = text.split()
            for i in range(len(text_tokens)):
                if text_tokens[i] in normal_form_dict:
                    text_tokens[i] = normal_form_dict[text_tokens[i]]
            texts_norm.append(' '.join(text_tokens))
        return texts_norm

    def aggregate_scores(self, text2token, vectorizer, topics):
        """Given matrix text2token of scores of vocabulary tokens to texts,
        computes scores of texts 2 tokens as some aggregate function(min, max, mean) of scores
        of tokens in topic
        """
        n_texts, n_tokens = text2token.shape
        n_topics = len(topics)
        tokens = pd.Series(vectorizer.get_feature_names())
        topic2text = np.empty((n_topics, n_texts))
        for i, topic in enumerate(topics):
            # extracting indices of words in leaf_name
            curr_tokens = topics[i]
            idx = tokens[tokens.isin(curr_tokens)].index.to_numpy()
            topic_scores = eval(f'np.{self.aggregation_F}')(text2token[:, idx], axis=1)  # aggregation
            topic2text[i] = topic_scores
        text2topic = topic2text.T
        return text2topic

    def build_similarity_matrix(self):
        n_topics = len(self.topics_preproc)
        sim_mat = np.ones(shape=(n_topics, n_topics))
        for i in tqdm(range(n_topics)):
            for j in range(i+1, n_topics):
                sim_mat[i][j] = self.similarity(self.topics_preproc[i], self.topics_preproc[j])
                sim_mat[j][i] = sim_mat[i][j]
        self.sim_mat = sim_mat

    def similarity(self, topic1, topic2):
        """Returns similarity score between two topics.
        Doesn't encounter intersection tokens in topics
        """

        assert type(topic1) == set and type(topic2) == set
        assert type(self.tokens) == pd.Series

        intersection = topic1.intersection(topic2)

        tokens1 = topic1 - intersection

        tokens2 = topic2 - intersection

        topic1_mask = self.tokens.isin(tokens1)
        topic2_mask = self.tokens.isin(tokens2)

        idx1 = self.tokens[topic1_mask].index.to_numpy()
        idx2 = self.tokens[topic2_mask].index.to_numpy()

        if len(idx1) == 0 or len(idx2) == 0:
            return 0

        tokens_vectors1 = self.text2token[:, topic1_mask].copy()
        tokens_vectors2 = self.text2token[:, topic2_mask].copy()

        scores1 = eval(f'np.{self.aggregation_F}')(
            tokens_vectors1, axis=1)
        scores2 = eval(f'np.{self.aggregation_F}')(
            tokens_vectors2, axis=1)

        return scores1 @ scores2


class AST:
    """
    Annotated Suffix Trees (AST)

    Given a collection of topics (key words) and a collection of texts
        computes a relevance matrix of each topic to each text using
        Annotated Trees built for texts.
    """

    def __init__(self, ):
        tqdm.pandas()
        self.AST_trees = {}

        self.relevance_matrix = None
        self.topics_ast = None

    def fit(self, texts, topics):

        print("BUILDING AST'S FOR TEXTS")
        for i, text in enumerate(tqdm(texts)):
            if i not in self.AST_trees:
                ast = self.build_ast(text)
                self.AST_trees[i] = ast

        print("BUILDING relevance_matrix")
        self.relevance_matrix = np.empty((len(texts), len(topics)))
        self.topics_ast = self.preprocess_topics(topics)

        for i, ast in tqdm(self.AST_trees.items()):
            self.relevance_matrix[i] = np.array(self.score(ast))

    @staticmethod
    def build_ast(text, n_words=5):
        return east.asts.base.AST.get_ast(east.utils.text_to_strings_collection(text, words=n_words))

    def score(self, ast):
        return [ast.score(t) for t in self.topics_ast]

    @staticmethod
    def preprocess_topics(topics):
        topics_ast = []
        for topic in topics:
            topics_ast.append(east.utils.prepare_text(
                text_utils.preprocess_text(topic)
            )
                              .replace(' ', '')
                              )

        return topics_ast
