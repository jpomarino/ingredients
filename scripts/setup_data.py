"""
Script to set up ingredient recommendation models
"""

import pandas as pd
import sys
from mlxtend.frequent_patterns import apriori, association_rules

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


def main():
    train_apriori(data_path, output_dir)


if __name__ == "__main__":
    main()
