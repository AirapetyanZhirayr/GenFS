import east
import pandas as pd

from libs import *
import text_utils
import ParGenN
import utils
tqdm.pandas()


class AST_custom:
    def __init__(self, ):
        self.AST_trees = {}

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



    def build_ast(self, text, n_words=5):
        return east.asts.base.AST.get_ast(east.utils.text_to_strings_collection(text, words=n_words))

    def score(self, ast):
        return [ast.score(t) for t in self.topics_ast]

    def preprocess_topics(self, topics):
        topics_ast = [
            east.utils.prepare_text(
                text_utils.preprocess_text(topic)
            )
                .replace(' ', '')
            for topic in topics
        ]

        return topics_ast


if __name__ == "__main__":
    TAXONOMY_TYPE='_ENHANCED'
    PAPERS_TYPE = '_ENHANCED'
    keywords = True
    shortcut = {'_ENHANCED':'enh', '':'std'}  # for saving relevance_matrix

    papers_df = pd.read_csv(f'input_data/text_collections/papers_df{PAPERS_TYPE}.csv',
                            index_col=0)

    if keywords:
        papers_df['keywords'] = papers_df['keywords'].apply(eval)
        # papers_df = papers_df.sample(200)
        texts = papers_df['keywords'].apply(lambda x: ' '.join(x)).str.lower().to_numpy()
    else:
        try:
            texts = pd.read_csv(f'input_data/text_collections/abstracts_preprocessed{PAPERS_TYPE}.csv', index_col=0)
            texts = texts['abstract'].to_numpy()
        except FileNotFoundError:

            texts = (papers_df['abstract'])
            texts = texts.progress_apply(text_utils.preprocess_text)
            texts.to_frame().to_csv(f'input_data/text_collections/abstracts_preprocessed{PAPERS_TYPE}.csv')
            texts = texts.to_numpy()


    taxonomy_df = pd.read_csv(f'input_data/taxonomies/taxonomy_df{TAXONOMY_TYPE}.csv')
    taxonomy = ParGenN.Taxonomy(taxonomy_df)
    topics_unique, topics_idx = utils.get_unique_topics(taxonomy)

    ast_custom = AST_custom()
    try:
        ast_custom.AST_trees = load_obj(name=f'input_data/ASTs/AST_{"keywords_" if keywords else ""}\
{shortcut[PAPERS_TYPE]}.pkl')
    except FileNotFoundError:
        print('testing')
        pass

    ast_custom.fit(texts=texts, topics=topics_unique)

    save_obj(obj=ast_custom.AST_trees, name=f'input_data/ASTs/AST_{"keywords_" if keywords else ""}\
{shortcut[PAPERS_TYPE]}.pkl')
    save_obj(obj=ast_custom.relevance_matrix,
             name = f'input_data/relevance_matrices/AST/rel_mat_AST_{"keywords_" if keywords else ""}\
{shortcut[PAPERS_TYPE]}_{shortcut[TAXONOMY_TYPE]}.pkl')



