"""
Module to create ingredient recommendations with apriori model.
"""

import pandas as pd
from .base import Recommender


class AprioriRecommender(Recommender):
    def __init__(self, rules: pd.DataFrame):
        self.rules = rules

    def recommend(self, ingredient: str) -> dict[str, float]:
        """
        Return ingredient apriori recommendations given an ingredient.

        Args:
            ingredient (str): Ingredient query.

        Returns:
            dict[str, float]: Dictionary containing the top recommendations along with their lift score.
        """

        # Check to see if the ingredient is even present in the rules learned
        ingredient_present = (
            self.rules["antecedents"].apply(lambda li: ingredient in li).any()
        )
        if not ingredient_present:
            print(f"Ingredient '{ingredient}' does not appear in the rules learned.")
            return {}

        # Filter out for ingredient of interest
        filtered_rules = self.rules[
            self.rules["antecedents"].apply(lambda li: ingredient in li)
            & (self.rules["lift"] > 1)
        ].copy()
        filtered_rules = filtered_rules.sort_values("lift", ascending=False)

        # Populate a dictionary with the top ingredient recommendations
        top_recs = {}

        for _, row in filtered_rules.iterrows():
            for ing in row["consequents"]:
                if ing not in top_recs:
                    top_recs[ing] = row["lift"]

        return top_recs
