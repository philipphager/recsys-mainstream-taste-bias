import numba as nb
import numpy as np

from src.model.base import BaseModel
from src.util.mapper import PositionalMapping


class FunkSVD(BaseModel):
    def __init__(self, params):
        super().__init__(f"funk")
        self.dimensions = params["dimensions"]
        self.epochs = params["epochs"]
        self.learning_rate = params["learning_rate"]
        self.regularization_rate = params["regularization_rate"]

    def fit(self, frame):
        self.n_users = frame.user.nunique()
        self.n_items = frame.item.nunique()
        self.user_mapping = PositionalMapping(frame.user)
        self.item_mapping = PositionalMapping(frame.item)
        frame.user = frame.user.map(self.user_mapping)
        frame.item = frame.item.map(self.item_mapping)

        self.global_mean, self.user_bias, self.item_bias, self.user_factors, self.item_factors = self.sgd(
            frame.values,
            self.n_users,
            self.n_items,
            self.dimensions,
            self.epochs,
            self.learning_rate,
            self.regularization_rate)

    @staticmethod
    @nb.jit(nopython=True)
    def sgd(train, n_users, n_items, n_factors, n_epochs, learning_rate,
            regularization_rate):
        global_mean = np.mean(train[:, 2])
        user_bias = np.zeros(n_users)
        item_bias = np.zeros(n_items)
        user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        item_factors = np.random.normal(0, 0.1, (n_items, n_factors))

        for e in range(n_epochs):
            for i in range(train.shape[0]):
                user, item, rating = train[i]

                user = int(user)
                item = int(item)

                predicted_rating = global_mean + user_bias[user] + item_bias[item]
                predicted_rating += np.dot(user_factors[user], item_factors[item])
                error = rating - predicted_rating

                # SGD Step
                user_bias[user] += learning_rate * (
                            error - regularization_rate * user_bias[user])
                item_bias[item] += learning_rate * (
                            error - regularization_rate * item_bias[item])
                user_factors[user] += learning_rate * (
                        error * item_factors[item] - regularization_rate * user_factors[
                    user])
                item_factors[item] += learning_rate * (
                        error * user_factors[user] - regularization_rate * item_factors[
                    item])

        return global_mean, user_bias, item_bias, user_factors, item_factors

    def predict(self, user: int, item: int):
        u = self.user_mapping[user]
        i = self.item_mapping[item]

        bias = self.global_mean + self.user_bias[u] + self.item_bias[i]
        predicted_rating = bias + np.dot(self.user_factors[u], self.item_factors[i])
        return [{
            "user": user,
            "item": item,
            "predicted_rating": predicted_rating
        }]

    def knows_user(self, user: int):
        return user in self.user_mapping

    def knows_item(self, item: int):
        return item in self.item_mapping
