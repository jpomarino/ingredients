"""
Abstract class for Recommender object
"""

class Recommender:
    def recommend(self, ingredient: str) -> dict:
        raise NotImplementedError