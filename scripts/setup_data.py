"""
Script to set up ingredient recommendation models
"""

import sys
import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules
from gensim.models import Word2Vec

sys.path.append("..")

# Define directories
data_path = "./data/raw/recipes_by_ingredients.csv"
output_dir = "./data/artifacts"


def train_apriori(data_path: str, output_dir: str) -> None:
    """
    Read in recipe x ingredient data and learn apriori association rules. Store association rules as a parquet file.

    Args:
        data_path (str): Dataset path.
        output_dir (str): Output directory
    """

    # Read in the dataset as a pandas DataFrame
    df = pd.read_csv(data_path, index_col=0)
    df = df.astype(bool)

    # Generate frequent itemsets
    frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Change frozenset objects to sorted lists to be able to save as parquet
    rules["antecedents"] = rules["antecedents"].apply(lambda x: sorted(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: sorted(list(x)))

    # Save rules to parquet
    rules.to_parquet(f"{output_dir}/apriori_rules.parquet")

    print("Apriori model trained!")


def train_word2vec(
    data_path: str,
    output_dir: str,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 30,
    workers: int = 4,
    epochs: int = 20,
    sg: int = 1,
) -> None:
    """
    Read in recipe x ingredient data and train word2vec model. Store model as a pickle file.

    Args:
        data_path (str): Dataset path.
        output_dir (str): Output directory
        vector_size (int, optional): Size of the word2vec vectors. Defaults to 100.
        window (int, optional): Max distance between words in sentence. Defaults to 30.
        min_count (int, optional): Minimum count to filter out rare words. Defaults to 5.
        workers (int, optional): Number of workers for parallelization. Defaults to 4.
        epochs (int, optional): Training epochs. Defaults to 20.
        sg (int, optional): Skipgram option. Defaults to 1.
    """
    # Read in the dataset as a pandas DataFrame
    df = pd.read_csv(data_path, index_col=0)

    # Extract recipes as lists of ingredients
    recipes = (
        df.astype(bool).apply(lambda row: row.index[row].tolist(), axis=1).tolist()
    )

    # Instantiate the model
    model = Word2Vec(
        sentences=recipes,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        compute_loss=True,
    )

    # Train the model
    model.train(recipes, total_examples=len(recipes), epochs=epochs)

    # Save model
    model.save(f"{output_dir}/word2vec_model.pkl")

    print("Word2Vec model trained!")


def main():
    train_apriori(data_path, output_dir)
    train_word2vec(data_path, output_dir)


if __name__ == "__main__":
    main()
