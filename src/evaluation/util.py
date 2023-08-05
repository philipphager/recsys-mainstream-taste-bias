import pandas as pd

from src.evaluation.metrics import get_user_metrics


def compute_metrics(prediction_df):
    rows = []

    for user, group in prediction_df.groupby("user"):
        row = get_user_metrics(
            group.rating.to_numpy(),
            group.predicted_rating.to_numpy()
        )
        row["user"] = int(user)
        rows.append(row)

    return pd.DataFrame(rows)


def batch_predict(model, frame):
    predictions = []

    for i, row in frame.iterrows():
        user = row["user"]
        item = row["item"]

        if model.knows_user(user) and model.knows_item(item):
            predictions.extend(model.predict(user, item))
        else:
            print("Unknown user:", user, "or item:", item)

    prediction_df = pd.DataFrame(predictions)
    prediction_df = prediction_df.merge(frame, on=["user", "item"])
    return prediction_df


def has_improved(value, best_value, minimize):
    if best_value is None:
        return True

    return value < best_value if minimize else value > best_value
