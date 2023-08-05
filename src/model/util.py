import numpy as np
import pandas as pd


def _get_pearson(frame: pd.DataFrame) -> np.array:
    """
    Fast Numpy implementation of pearson correlation.
    Cannot handle missing values properly, use only for complete datasets.
    """
    with np.errstate(all="ignore"):
        correlation_df = frame.pivot_table(
            index="user", columns="item", values="rating"
        )
        correlation_matrix = np.corrcoef(correlation_df.values)
        correlation_matrix[np.isnan(correlation_matrix)] = 0
        return correlation_matrix


def _get_sparse_pearson(frame: pd.DataFrame, min_items=2) -> np.array:
    """
    Slow Pandas implementation of pearson correlation.
    Handles missing values by computing similarity between users if both rated min_items shared items.
    """
    correlation_df = frame.pivot_table(index="item", columns="user", values="rating")
    correlation_df = correlation_df.corr(method="pearson", min_periods=min_items)
    correlation_df = correlation_df.fillna(0)
    return correlation_df.values


def get_similarity(frame: pd.DataFrame, is_complete: bool = False) -> np.array:
    if is_complete:
        # Use faster numpy implementation, which cannot handle missing values.
        return _get_pearson(frame)
    else:
        return _get_sparse_pearson(frame)


def get_user_item_ratings(frame: pd.DataFrame):
    return frame.pivot_table(index="user", columns="item", values="rating").values


def is_complete(frame: pd.DataFrame):
    """
    Check if all users rated all items.
    """
    return (frame.groupby("user").item.nunique() == frame.item.nunique()).all()
