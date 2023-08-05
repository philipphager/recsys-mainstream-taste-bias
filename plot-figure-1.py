import glob

import altair as alt
import pandas as pd

from src.data.faces import Faces
from src.data.jester import Jester
from src.data.movielens import MovieLens
from src.user.properties import get_user_similarity


def get_user_level_evaluation(datasets):
    test_dfs = []

    for name, dataset in datasets.items():
        files = glob.glob(f"outputs/data={name}*/test.parquet")

        if len(files) == 0:
            print(f"No test results found for dataset: {name}")
            continue

        # Load rating dataset and compute taste similarity and dispersion
        df = dataset.load()
        user_df = get_user_similarity(df)

        # Load rating dataset, average over multiple repetitions
        test_df = pd.concat([pd.read_parquet(f) for f in files])
        test_df = (
            test_df.groupby(["user", "model", "dataset"])
                .mean(numeric_only=True)
                .reset_index()
        )
        test_df = test_df.merge(user_df, on="user")
        test_dfs.append(test_df)

    return pd.concat(test_dfs)


def plot():
    datasets = {
        "faces": Faces(),
        "jester": Jester(),
        "movielens": MovieLens()
    }

    df = get_user_level_evaluation(datasets)
    # Drop outliers for plotting
    df = df[df.taste_dispersion > 0.05]

    # Plot figure 1
    chart = alt.Chart(df, width=200, height=200).mark_circle().encode(
        column=alt.Column("dataset", header=alt.Header(title="", labelFontSize=12)),
        row=alt.Row("model", header=alt.Header(title="", labelFontSize=12)),
        x=alt.X(
            "mean_taste_similarity",
            title="Mean taste similarity",
        ),
        y=alt.Y(
            "taste_dispersion",
            title="Taste dispersion",
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color(
            "fcp",
            title="FCP",
            scale=alt.Scale(
                domain=[0.4, 1.0],
                scheme=alt.SchemeParams("inferno", extent=(0.25, 1.0)),
            )
        ),
        tooltip=["user", "fcp", "nDCG", "rmse"],
    ).resolve_scale(
        y="independent",
    ).configure_axis(
        titleFontSize=12,
        titleFontWeight="normal",
    ).interactive()

    chart.save("figures/figure-1.html")


if __name__ == '__main__':
    plot()
