import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import ParameterGrid

from src.model.base import BaseModel
from src.util.mapper import PositionalMapping


class SVD(BaseModel):
    def __init__(self, frame, max_k):
        super().__init__("svd")
        self.frame = frame
        self.max_k = max_k

        self.user_mapping = PositionalMapping(frame.user)
        self.item_mapping = PositionalMapping(frame.item)

        self.frame.user = frame.user.map(self.user_mapping)
        self.frame.item = frame.item.map(self.item_mapping)

        user_item_ratings = csr_matrix((frame.rating.values,
                                        (frame.user.values, frame.item.values)),
                                       dtype=np.float64)

        self.user_features, self.feature_importance, self.item_features = svds(
            user_item_ratings, k=self.max_k)

    def predict(self, user, item, parameters: ParameterGrid):
        rows = []
        u = self.user_mapping[user]
        i = self.item_mapping[item]

        for parameter in parameters:
            k = parameter["k"]
            assert k > 0, "Dimensions k must be a positive integer k >= 1"

            user_vector = self.user_features[u, -k:]
            item_vector = self.item_features.T[i, -k:]
            feature_importance = self.feature_importance[-k:]

            predicted_rating = np.dot(user_vector * feature_importance, item_vector)

            rows.append({
                "user": user,
                "item": item,
                "k": k,
                "predicted_rating": predicted_rating
            })

        return rows

    def knows_user(self, user):
        return user in self.user_mapping

    def knows_item(self, item):
        return item in self.item_mapping
