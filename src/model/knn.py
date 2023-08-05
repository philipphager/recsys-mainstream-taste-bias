import numpy as np

from src.model.base import BaseModel
from src.model.util import is_complete, get_user_item_ratings, get_similarity
from src.util.mapper import PositionalMapping


class KNN(BaseModel):
    def __init__(self, params):
        assert params["method"] in ("item-item", "user-user")
        super().__init__(f"knn-{params['method']}")
        self.method = params["method"]
        self.k = params["k"]
        self.rho = params["rho"]

    def fit(self, frame):
        self.user_mapping = PositionalMapping(frame.user)
        self.item_mapping = PositionalMapping(frame.item)

        frame.user = frame.user.map(self.user_mapping)
        frame.item = frame.item.map(self.item_mapping)

        # Swap users and items when computing item-item recommendations
        if self.method == "item-item":
            frame = frame.rename(columns={"user": "item", "item": "user"})

        self.is_complete = is_complete(frame)
        self.user_item_ratings = get_user_item_ratings(frame)
        self.similarity = get_similarity(frame, is_complete=self.is_complete)

    def predict(self, user: int, item: int):
        # Map outside users to positions in user-item rating matrix
        u, i = self._get_indices(user, item)
        # Get max k in parameters to fetch all neighbors at once
        # Finds neighbors and their ratings
        neighbors, similarities = self._get_neighbors(user=u, item=i, k=self.k)
        ratings = self.user_item_ratings[neighbors, i]

        # Raise similarities to the power of rho while preserving signs
        signs = np.sign(similarities)
        weighted_similarities = (np.abs(similarities) ** self.rho) * signs
        # Calculate weighted rating
        weighted_ratings = weighted_similarities * ratings
        total_similarity = weighted_similarities[:self.k].sum()
        total_ratings = weighted_ratings[:self.k].sum()

        if total_similarity != 0:
            predicted_rating = total_ratings / total_similarity
        else:
            predicted_rating = None

        return [{
            "user": user,
            "item": item,
            "predicted_rating": predicted_rating
        }]

    def knows_user(self, user):
        return user in self.user_mapping

    def knows_item(self, item):
        return item in self.item_mapping

    def _get_indices(self, user, item):
        if self.method == "item-item":
            u = self.item_mapping[item]
            i = self.user_mapping[user]
        else:
            u = self.user_mapping[user]
            i = self.item_mapping[item]
        return u, i

    def _get_neighbors(self, user, item, k=1):
        # Fetch all ratings for the selected item
        item_ratings = self.user_item_ratings[:, item]
        # Fetch all users having rated the item
        neighbors = np.argwhere(~np.isnan(item_ratings)).squeeze(-1)
        # Fetch similarity of users to the anchor user
        similarities = self.similarity[user, neighbors]
        # Sort neighbors ascending based on similarity
        sort_index = np.argsort(-similarities)
        neighbors = neighbors[sort_index]
        similarities = similarities[sort_index]
        # Exclude anchor user from result
        should_return = neighbors != user
        neighbors = neighbors[should_return]
        similarities = similarities[should_return]
        return neighbors[:k], similarities[:k]
