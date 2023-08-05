import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from src.data.faces import Faces
from src.data.jester import Jester
from src.data.movielens import MovieLens
from src.user.properties import get_user_similarity, get_user_properties


def get_user_level_evaluation(datasets):
    frames = []

    for name, dataset in datasets.items():
        files = glob.glob(f"outputs/data={name}*/test.parquet")

        if len(files) == 0:
            print(f"No test results found for dataset: {name}")
            continue

        # Load rating dataset and compute taste similarity and dispersion
        df = dataset.load()
        user_similarity_df = get_user_similarity(df)
        user_property_df = get_user_properties(df)

        # Load rating dataset, average over multiple repetitions
        test_df = pd.concat([pd.read_parquet(f) for f in files])
        test_df = (
            test_df.groupby(["user", "model", "dataset"])
                .mean(numeric_only=True)
                .reset_index()
        )
        test_df = test_df.merge(user_similarity_df, on="user")
        test_df = test_df.merge(user_property_df, on="user")
        frames.append(test_df)

    return pd.concat(frames)


def evaluate(X, y):
    folds = KFold()
    scores = []

    for train_idx, test_idx in folds.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)

        score = r2_score(y_test, y_predict)
        score = adjust_r2(score, X.shape[0], X.shape[1])
        scores.append(score)

    return np.mean(scores)


def adjust_r2(r2, n_samples, n_variables):
    return 1 - (((1 - r2) * (n_samples - 1)) / (n_samples - n_variables - 1))


def r2_analysis():
    metric = "fcp"
    datasets = {
        "faces": Faces(),
        "jester": Jester(),
        "movielens": MovieLens()
    }
    feature_combinations = [
        ["mean_taste_similarity"],
        ["taste_dispersion"],
        ["mean_taste_similarity", "taste_dispersion"],
        ["mean_rating", "var_rating", "rated_items", "mean_popularity", "user_gini",
         "item_gini"],
        ["mean_taste_similarity", "taste_dispersion", "mean_rating", "var_rating",
         "rated_items", "mean_popularity", "user_gini", "item_gini"],
    ]

    df = get_user_level_evaluation(datasets)

    for dataset in df.dataset.unique():
        rows = []

        for feature_combination in feature_combinations:

            for model in df.model.unique():
                filter_df = df[(df["model"] == model) & (df["dataset"] == dataset)]
                filter_df = filter_df.fillna(0)

                X = filter_df[feature_combination].values
                y = filter_df[metric].values
                score = evaluate(X, y)

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "features": ", ".join(feature_combination),
                    "r2": score
                })

        result_df = pd.DataFrame(rows)
        result_df = result_df.pivot_table(
            index=["dataset", "features"],
            columns=["model"]
        )
        print(result_df.to_string())


if __name__ == '__main__':
    r2_analysis()
