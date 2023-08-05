import logging
from pathlib import Path

import pandas as pd

from src.util.mapper import PositionalMapping


class BaseDataset:
    def __init__(self, name: str, output_directory: Path = Path.home() / ".datasets/"):
        self.name = name
        self.output_directory = output_directory

    def load(self, force: bool = False) -> pd.DataFrame:
        path = self._get_path()

        if not path.exists() or force:
            logging.info(f"Dataset {self.name} not on disk. Downloading to: {path}")
            # Download and preprocess dataset
            df = self._download()
            # Check dataset columns
            assert all([c in df.columns for c in ["user", "item", "rating"]])
            # Map user and item ids to incremental ints
            user2position = PositionalMapping(df.user)
            item2position = PositionalMapping(df.item)
            df.user = df.user.map(user2position)
            df.item = df.item.map(item2position)
            # Min-Max scaling of ratings between 0..1
            df.rating = (df.rating - df.rating.min()) / (
                        df.rating.max() - df.rating.min())
            # Store processed dataset locally
            df.to_parquet(path)
        else:
            logging.info(f"Load {self.name} dataset from disk")

        # Read and return local dataset
        return pd.read_parquet(path)

    def _download(self):
        raise NotImplementedError("Dataset must implement _download()")

    def _get_path(self):
        self.output_directory.mkdir(exist_ok=True)
        return self.output_directory / f"{self.name}.parquet"
