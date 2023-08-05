import numpy as np
from scipy.sparse import csr_matrix
from sklearn import decomposition

from src.model.base import BaseModel
from src.util.mapper import PositionalMapping


class NMF(BaseModel):
    def __init__(self, params):
        super().__init__("nmf")
        self.components = params["components"]
        self.alpha = params["alpha"]
        self.l1_ratio = params["l1_ratio"]
        self.iterations = params["iterations"]
        self.seed = params["seed"]

    def fit(self, frame):
        self.user_mapping = PositionalMapping(frame.user)
        self.item_mapping = PositionalMapping(frame.item)

        frame.user = frame.user.map(self.user_mapping)
        frame.item = frame.item.map(self.item_mapping)

        user_item_ratings = csr_matrix((frame.rating.values,
                                        (frame.user.values, frame.item.values)),
                                       dtype=np.float64)

        model = decomposition.NMF(
            n_components=self.components,
            init="random",
            random_state=self.seed,
            alpha_H=self.alpha,
            alpha_W=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.iterations,
        )

        user_features = model.fit_transform(user_item_ratings)
        item_features = model.components_
        self.predicted_user_item_ratings = user_features @ item_features

    def predict(self, user, item):
        rows = []
        u = self.user_mapping[user]
        i = self.item_mapping[item]

        predicted_rating = self.predicted_user_item_ratings[u, i]

        rows.append({
            "user": user,
            "item": item,
            "components": self.components,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "predicted_rating": predicted_rating,
        })

        return rows

    def knows_user(self, user):
        return user in self.user_mapping

    def knows_item(self, item):
        return item in self.item_mapping
