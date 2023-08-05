from abc import ABC, abstractmethod
from typing import List, Dict

import pandas as pd


class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, frame: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, user: int, item: int) -> List[Dict]:
        pass
