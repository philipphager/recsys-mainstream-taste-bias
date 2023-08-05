import numpy as np
from scipy.sparse import csr_matrix

from src.model.base import BaseModel
from src.util.mapper import PositionalMapping


class EASE(BaseModel):
    def __init__(self, params):
        super().__init__(f"ease")
        self.regularization_rate = params["regularization_rate"]

    def parameters(self):
        return {"regularization_rate": self.regularization_rate}

    def fit(self, frame):
        self.user_mapping = PositionalMapping(frame.user)
        self.item_mapping = PositionalMapping(frame.item)
        frame.user = frame.user.map(self.user_mapping)
        frame.item = frame.item.map(self.item_mapping)

        self.user_item_ratings = csr_matrix((
            frame.rating.values,
            (frame.user.values, frame.item.values)
        ), dtype=np.float64)

        # Compute item-item similarity using EASE
        self.item_similarity = self._get_item_similarity(self.user_item_ratings)
        # Precompute all user-item ratings
        self.predicted_user_item_ratings = self.user_item_ratings.dot(self.item_similarity)

    def predict(self, user, item):
        u = self.user_mapping[user]
        i = self.item_mapping[item]

        assert self.user_item_ratings[u, i] == 0, "User-item pair in train set"
        predicted_rating = self.predicted_user_item_ratings[u, i]

        return [{
            "user": user,
            "item": item,
            "predicted_rating": predicted_rating
        }]

    def knows_user(self, user):
        return user in self.user_mapping

    def knows_item(self, item):
        return item in self.item_mapping

    def _get_item_similarity(self, user_item_ratings):
        # EASE: Embarrassingly Shallow Autoencoder
        # Formulas from: https://arxiv.org/abs/1905.03375
        G = user_item_ratings.T.dot(user_item_ratings)
        G = G.todense()
        diagonal = np.diag_indices(G.shape[0])
        G[diagonal] += self.regularization_rate
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagonal] = 0
        return np.asarray(B)
