"""
Custom realization of some algorithms
"""
from libs import *
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
nltk_stop_words = tuple(stopwords.words('english'))


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
