from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

from src.data.base import BaseDataset


class Jester(BaseDataset):
    def __init__(self):
        super().__init__("jester")

    def _download(self) -> pd.DataFrame:
        df = pd.concat([
            self._download_part(1),
            self._download_part(2),
            self._download_part(3),
        ])

        # Filter only for users that rated all items
        df = df[df.num_ratings == 100]

        # Create user columns
        df = df.reset_index(drop=True)
        df = df.reset_index().rename(columns={"index": "user"})
        df = df.drop(columns=["num_ratings"])

        # Pivot frame to merge all joke columns and ratings into two columns
        df = df.melt(id_vars=["user"], var_name="item", value_name="rating")

        # Default sorting is by item, sort frame by user id
        df = df.sort_values("user")
        df = df.reset_index(drop=True)
        return df

    def _download_part(self, part: int):
        url = f"https://goldberg.berkeley.edu/jester-data/jester-data-{part}.zip"
        zip_path = self.output_directory / f"{self.name}.zip"
        urlretrieve(url, zip_path)

        with ZipFile(zip_path) as f:
            data = f.open(f"jester-data-{part}.xls")

            # Import Excel file
            # File has no header row, 101 columns, and one row per user
            # First column is the number of rated items
            # Columns 1 - 101 are the ratings for the 100 jokes
            df = pd.read_excel(data, header=None)
            # Generate column names
            df.columns = ["num_ratings"] + [i for i in range(100)]

            # Delete .zip archive
            zip_path.unlink()

        return df
