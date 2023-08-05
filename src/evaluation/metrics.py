import numba as nb
import numpy as np


@nb.jit(nopython=True)
def mae(ratings, predicted_ratings):
    return (np.abs(ratings - predicted_ratings)).mean()


@nb.jit(nopython=True)
def mse(ratings, predicted_ratings):
    return ((ratings - predicted_ratings) ** 2).mean()


@nb.jit(nopython=True)
def rmse(ratings, predicted_ratings):
    return ((ratings - predicted_ratings) ** 2).mean() ** 0.5


@nb.jit(nopython=True)
def dcg(ranking):
    rank_bias = np.log2(np.arange(2, len(ranking) + 2))
    ranking = 2 ** ranking - 1
    return (ranking / rank_bias).sum()


@nb.jit(nopython=True)
def ndcg(ratings, predicted_ratings, k=None):
    predicted_order = np.argsort(-predicted_ratings)
    predicted_ranking = ratings[predicted_order][:k]
    ideal_order = np.argsort(-ratings)
    ideal_ranking = ratings[ideal_order][:k]
    predicted_dcg = dcg(predicted_ranking)
    ideal_dcg = dcg(ideal_ranking)
    return predicted_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


@nb.jit(nopython=True)
def fcp(ratings, predicted_ratings):
    concordant_pairs = 0
    discordant_pairs = 0
    equal_pairs = 0

    for i in range(len(ratings)):
        for j in range(len(ratings)):
            if i < j:
                if (
                        ratings[i] > ratings[j]
                        and predicted_ratings[i] > predicted_ratings[j]
                        or ratings[i] < ratings[j]
                        and predicted_ratings[i] < predicted_ratings[j]
                ):
                    concordant_pairs += 1
                elif (
                        ratings[i] < ratings[j]
                        and predicted_ratings[i] > predicted_ratings[j]
                        or ratings[i] > ratings[j]
                        and predicted_ratings[i] < predicted_ratings[j]
                ):
                    discordant_pairs += 1
                elif ratings[i] == ratings[j]:
                    equal_pairs += 1

    concordant_pairs += equal_pairs // 2
    discordant_pairs += equal_pairs // 2

    pairs = concordant_pairs + discordant_pairs
    return concordant_pairs / pairs if pairs > 0 else None


def get_user_metrics(ratings, predicted_ratings):
    return {
        "fcp": fcp(ratings, predicted_ratings),
        "nDCG": ndcg(ratings, predicted_ratings),
        "nDCG@10": ndcg(ratings, predicted_ratings, k=10),
        "mae": mae(ratings, predicted_ratings),
        "mse": mse(ratings, predicted_ratings),
        "rmse": rmse(ratings, predicted_ratings),
    }
