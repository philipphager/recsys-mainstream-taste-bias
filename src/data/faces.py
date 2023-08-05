import pandas as pd

from src.data.base import BaseDataset


class Faces(BaseDataset):
    def __init__(self):
        super().__init__("faces")
        self.url = "https://ndownloader.figshare.com/files/8542045"

    def _download(self) -> pd.DataFrame:
        df = pd.read_csv(self.url)
        # Remove unused columns
        df = df.drop(columns=["rater_sex", "rater_sexpref", "rater_age"])
        # Rename each column representing a rated face to an int id
        df.columns = range(102)
        # Create user columns
        df = df.reset_index().rename(columns={"index": "user"})
        # Pivot frame to merge all face columns and ratings into two columns
        df = df.melt(id_vars=["user"], var_name="item", value_name="rating")
        # Default sorting is by item, sort frame by user id
        df = df.sort_values("user")
        df = df.reset_index(drop=True)
        return df
