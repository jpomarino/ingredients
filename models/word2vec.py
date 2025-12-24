"""
Module to create ingredient recommendations with Word2Vec model.
"""

import pandas as pd
from gensim.models import Word2Vec
from .base import Recommender


class Word2VecRecommender(Recommender):
    def __init__(self):
        self.model = Word2Vec.load("data/artifacts/word2vec_model.pkl")

    def recommend(self, ingredient: str, top_k: int = 15):
        recs = self.model.wv.most_similar(ingredient, topn=top_k)

        return {tup[0]: tup[1] for tup in recs}
