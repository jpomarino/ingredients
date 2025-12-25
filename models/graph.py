"""
Module to create ingredient recommendations with a graph model.
"""

from typing import Dict, Optional
import pandas as pd
import pickle
import networkx as nx
from scipy.sparse import csr_matrix
from math import log

from .base import Recommender


# TODO: add a method to train the graph and save as a pickle file in setup_data.py
class GraphRecommender(Recommender):
    def __init__(self, data_path: str = "data/artifacts/graph_model.pkl"):
        with open(data_path, "rb") as f:
            self.graph: nx.Graph = pickle.load(f)
        self.supported_ingredients = list(self.graph.nodes())

    def recommend(self, ingredient: str, top_k: int = 15) -> Dict[str, float]:
        """
        Recommend ingredients for the query ingredient.

        Args:
            ingredient (str): Query ingredient.
            top_k (int, optional): Number of ingredients to return. Defaults to 15.

        Returns:
            Dict[str, float]: Dictionary of top_k ingredients with their respective PMI values.
        """

        # Get ingredient's neighbors
        neighbors = self.graph[ingredient]
        recs = {nbr: data["weight"] for nbr, data in neighbors.items()}

        # Sort by PMI descending
        return dict(sorted(recs.items(), key=lambda x: -x[1])[:top_k])
