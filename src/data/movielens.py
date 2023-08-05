from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

from src.data.base import BaseDataset


class MovieLens(BaseDataset):
    def __init__(self):
        super().__init__("movielens-1m")
        self.url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def _download(self) -> pd.DataFrame:
        # Download dataset as .zip archive
        zip_path = self.output_directory / f"{self.name}.zip"
        urlretrieve(self.url, zip_path)

        with ZipFile(zip_path) as f:
            data = f.open("ml-1m/ratings.dat")

            # Import .dat file, separating columns by "::"
            df = pd.read_csv(data,
                             delimiter="::",
                             header=None,
                             names=["user_id", "movie_id", "rating", "timestamp"],
                             usecols=["user_id", "movie_id", "rating"],
                             engine="python")

            df = df.rename(columns={"user_id": "user", "movie_id": "item"})

            # Delete .zip archive
            zip_path.unlink()

        return df
