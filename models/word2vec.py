"""
Module to create ingredient recommendations with Word2Vec model.
"""

from gensim.models import Word2Vec
from .base import Recommender


class Word2VecRecommender(Recommender):
    def __init__(self):
        self.model = Word2Vec.load("data/artifacts/word2vec_model.pkl")
        self.supported_ingredients = list(self.model.wv.index_to_key)

    def recommend(self, ingredient: str, top_k: int = 15):
        """
        Recommend ingredients for the query ingredient.

        Args:
            ingredient (str): Query ingredient.
            top_k (int, optional): Number of ingredients to return. Defaults to 15.

        Returns:
            _type_: Dictionary of top_k ingredients with their respective cosine similarity values.
        """
        recs = self.model.wv.most_similar(ingredient, topn=top_k)

        return {tup[0]: tup[1] for tup in recs}
