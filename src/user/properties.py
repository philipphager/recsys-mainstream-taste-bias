import pandas as pd
from inequipy.inequipy import gini

from src.model.util import get_similarity, is_complete


def get_user_similarity(df):
    # Compute Pearson correlation between all users
    correlation_matrix = get_similarity(df, is_complete=is_complete(df))
    similarity_df = pd.DataFrame(correlation_matrix)
    similarity_df.index.name = "user"
    similarity_df = similarity_df.reset_index()
    similarity_df = similarity_df.melt(
        id_vars=["user"],
        var_name="neighbor",
        value_name="similarity")

    # Compute mean taste similarity and taste dispersion
    return similarity_df.groupby("user").agg(
        mean_taste_similarity=("similarity", "mean"),
        taste_dispersion=("similarity", "std"),
    ).reset_index()


def get_user_properties(df):
    # Compute item popularity and item-based gini index
    item_df = df.groupby(["item"]).agg(
        popularity=("user", "count"),
        item_gini=("rating", gini),
    ).reset_index()

    df = df.merge(item_df, on="item")

    # Aggregate user features
    return df.groupby(["user"]).agg(
        rated_items=("item", "nunique"),
        mean_rating=("rating", "mean"),
        var_rating=("rating", "var"),
        mean_popularity=("popularity", "mean"),
        user_gini=("rating", gini),
        item_gini=("item_gini", "mean"),
    ).reset_index()
