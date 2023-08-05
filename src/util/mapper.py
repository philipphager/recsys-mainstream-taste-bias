import numpy as np
import pandas as pd


class PositionalMapping:
    def __init__(self, values: np.array, sort_first: bool = True):
        assert pd.Series(values).notna().all(), "Passed values must not contain NaN"
        self.value2position = self._get_mapping(values, sort_first)

    def __getitem__(self, value):
        return self.value2position[value]

    def __call__(self, value):
        return self.value2position[value]

    def __contains__(self, value):
        return value in self.value2position

    @staticmethod
    def _get_mapping(values: np.array, sort_first: bool):
        value2position = {}
        values = np.sort(values) if sort_first else np.array(values)
        position = 0

        for value in values:
            if value not in value2position:
                value2position[value] = position
                position += 1

        return value2position
