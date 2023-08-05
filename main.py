import os

import hydra
import pandas as pd
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import ParameterGrid

from src.evaluation.util import compute_metrics, batch_predict, has_improved


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    print("Working directory : {}".format(os.getcwd()))

    data = instantiate(config.data)
    df = data.load()

    kfolds = instantiate(config.test_folds)
    results = []

    # Repeatedly split dataset into train/val/test into 60/20/20
    for fold, (train_idx, test_idx) in enumerate(kfolds.split(df, df.user)):
        # Split train/test into 80/20
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        # Split train/val into 60/20
        inner_train_df, val_df = call(config.val_fold, train_df)

        best_params = {}
        best_metric = None

        for params in ParameterGrid(dict(config.model.params)):
            print(f"Fold-{fold}: {params}")
            model = instantiate(config.model, params=params)
            model.fit(inner_train_df)

            predict_df = batch_predict(model, val_df)
            metric_df = compute_metrics(predict_df)
            val_metric = metric_df[config.val_metric].mean()

            if has_improved(val_metric, best_metric, config.val_metric_minimize):
                best_metric = val_metric
                best_params = params
                print(f"Best {config.val_metric}: {val_metric} - {params}")

        model = instantiate(config.model, params=best_params)
        model.fit(train_df)

        predict_df = batch_predict(model, test_df)
        test_metric_df = compute_metrics(predict_df)
        test_metric_df["model"] = model.name
        test_metric_df["dataset"] = data.name
        test_metric_df["fold"] = fold

        for name, value in best_params.items():
            test_metric_df[name] = value

        results.append(test_metric_df)

    metric_df = pd.concat(results)
    metric_df.to_parquet(f"test.parquet")


if __name__ == '__main__':
    main()
