"""
Module to create ingredient recommendations with a graph model.
"""

from typing import Dict, Optional
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from math import log

from .base import Recommender


class GraphRecommender(Recommender):
    def __init__(
        self,
        data_path: str = "data/raw/recipes_by_ingredients.csv",
        pmi_threshold: float = 0.1,
        min_cooccurrence: int = 10,
        graph: Optional[nx.Graph] = None,
    ):
        """
        Initialize the GraphRecommender class

        Args:
            data_path (str, optional): Path to where the recipe dataset is. Defaults to "data/raw/recipes_by_ingredients.csv".
            pmi_threshold (float, optional): Minimum PMI value to filter edges by. Defaults to 0.1.
            min_cooccurrence (int, optional): Minimum number of co-occurrences for an ingredient pair to be considered. Defaults to 10.
            graph (Optional[nx.Graph], optional): Graph representation of ingredients. Defaults to None.
        """
        self.df = pd.read_csv(data_path, index_col=0)
        self.ingredients = self.df.columns.to_list()
        self.pmi_threshold = pmi_threshold
        self.min_cooccurrence = min_cooccurrence

        if graph is not None:
            self.graph = graph
        else:
            self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        """
        Creates a networkx graph representation of the ingredients.

        Returns:
            nx.Graph: networkx graph representation of the ingredients.
        """

        X = csr_matrix(self.df.values)
        n_recipes = X.shape[0]

        # Ingredient co-occurrence
        cooccurrence = X.T @ X
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()

        # Marginal counts
        counts = X.sum(axis=0).A1

        G = nx.Graph()
        coo = cooccurrence.tocoo()

        # Iterate through all ingredient-ingredient pairs to add them to the graph
        for i, j, count in zip(coo.row, coo.col, coo.data):
            # Keep only ingredient co-occurrences over a certain threshold
            if count < self.min_cooccurrence:
                continue

            # Calculate probabilities
            p_ij = count / n_recipes
            p_i = counts[i] / n_recipes
            p_j = counts[j] / n_recipes

            # Ensure positive probabilities
            if p_ij <= 0 or p_i <= 0 or p_j <= 0:
                continue

            # Compute PMI
            pmi = log(p_ij / (p_i * p_j))

            # Only add edge if PMI is over a threshold
            if pmi > self.pmi_threshold:
                G.add_edge(
                    self.ingredients[i],
                    self.ingredients[j],
                    weight=pmi,
                )

        return G

    def recommend(self, ingredient: str, top_k: int = 15) -> Dict[str, float]:
        """
        Recommend ingredients for the query ingredient.

        Args:
            ingredient (str): Query ingredient.
            top_k (int, optional): Number of ingredients to return. Defaults to 15.

        Returns:
            Dict[str, float]: Dictionary of top_k ingredients with their respective PMI values.
        """

        # Check if ingredient is in the graph
        if ingredient not in self.graph:
            return {}

        # Get ingredient's neighbors
        neighbors = self.graph[ingredient]
        recs = {nbr: data["weight"] for nbr, data in neighbors.items()}

        # Sort by PMI descending
        return dict(sorted(recs.items(), key=lambda x: -x[1])[:top_k])
