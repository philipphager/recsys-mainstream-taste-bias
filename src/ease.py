import json
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, step, Parameter, JSONType
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold, StratifiedKFold

from mainstream_taste_bias.algorithm.ease import EASE
from mainstream_taste_bias.dataset import get_dataset
from mainstream_taste_bias.evaluation.metrics import list_metrics
from mainstream_taste_bias.evaluation.util import get_predictions, evaluate_predictions, get_parameter_names


class EvaluateEASE(FlowSpec):
    dataset = Parameter("dataset",
                        help="Dataset for evaluation: blinkist, faces, movielens, jester",
                        default="faces")

    test_folds = Parameter("test_folds",
                           help="(Outer K-Fold) Test splits to evaluate the best parameters",
                           default=5)

    validation_folds = Parameter("validation_folds",
                                 help="(Inner K-Fold) Validation splits to find the best parameters per user",
                                 default=4)

    validation_repeats = Parameter("validation_repeats",
                                   help="(Inner K-Fold) CV repetitions to find the best parameters per user",
                                   default=10)

    parameters = Parameter("parameters",
                           help="Grid of parameters to evaluate per model",
                           type=JSONType,
                           default=json.dumps({
                               "regularization_rate": list(range(100, 2_100, 100)),
                           }))

    @step
    def start(self):
        self.output_directory = Path("output") / "ease"
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.parameter_grid = ParameterGrid(self.parameters)
        self.group_columns = ["user"] + get_parameter_names(self.parameter_grid)
        self.next(self.load_dataset)

    @step
    def load_dataset(self):
        dataset = get_dataset(self.dataset)

        df = dataset.load()
        df = df.sample(frac=1.0, random_state=0)
        self.df = df
        self.next(self.split_test)

    @step
    def split_test(self):
        # Split the full datasets into K train / test datasets
        kfold = StratifiedKFold(n_splits=self.test_folds, shuffle=True, random_state=0)
        evaluations = []

        for fold, (train_index, test_index) in enumerate(kfold.split(self.df, self.df.user)):
            for parameter_id, parameters in enumerate(self.parameter_grid):
                evaluations.append([fold, parameter_id, parameters, train_index, test_index])

        self.evaluations = evaluations
        self.next(self.evaluate_test, foreach="evaluations")

    @step
    def evaluate_test(self):
        # Sweep all parameters on train / test for each user
        test_fold, parameter_id, test_parameters, train_index, test_index = self.input

        print("Test Fold:", test_fold, "Parameter:", parameter_id, test_parameters)
        train_df = self.df.iloc[train_index].copy()
        test_df = self.df.iloc[test_index].copy()

        file = f"{self.dataset}_test_{test_fold}_{parameter_id}.parquet"
        self.test_path = self.output_directory / file
        evaluation_df = self.evaluate(test_fold, train_df, test_df, test_parameters)
        evaluation_df.to_parquet(self.test_path)

        self.test_fold = test_fold
        self.train_df = train_df
        self.parameter_id = parameter_id
        self.validation_parameters = test_parameters
        self.next(self.split_validation)

    @step
    def split_validation(self):
        # Split train into K train / validation datasets
        kfold = RepeatedStratifiedKFold(
            n_splits=self.validation_folds,
            n_repeats=self.validation_repeats,
            random_state=0)
        self.fold_indices = list(enumerate(kfold.split(self.train_df, self.train_df.user)))
        self.next(self.evaluate_validation, foreach="fold_indices")

    @step
    def evaluate_validation(self):
        # Sweep all parameters on train / validation for each user
        validation_fold, (train_index, validation_index) = self.input
        print("Validation Fold:", self.test_fold, validation_fold,
              "Parameter:", self.parameter_id, self.validation_parameters)
        train_df = self.train_df.iloc[train_index].copy()
        validation_df = self.train_df.iloc[validation_index].copy()

        file = f"{self.dataset}_validation_{self.test_fold}_{validation_fold}_{self.parameter_id}.parquet"
        self.validation_path = self.output_directory / file
        evaluation_df = self.evaluate(self.test_fold, train_df, validation_df, self.validation_parameters)
        evaluation_df.to_parquet(self.validation_path)

        self.next(self.join_validation)

    @step
    def join_validation(self, inputs):
        self.merge_artifacts(inputs, exclude=["validation_path"])
        paths = [input.validation_path for input in inputs]

        file = f"{self.dataset}_validation_{self.test_fold}_{self.parameter_id}.parquet"
        self.validation_path = self.output_directory / file

        df = self.concat_partitions(paths)
        df.to_parquet(self.validation_path)
        self.delete_partitions(paths)

        self.next(self.join_test)

    @step
    def join_test(self, inputs):
        self.merge_artifacts(inputs, include=["dataset", "group_columns", "output_directory"])
        validation_paths = [input.validation_path for input in inputs]
        test_paths = [input.test_path for input in inputs]

        file = f"{self.dataset}.parquet"
        path = self.output_directory / file

        validation_df = self.concat_partitions(validation_paths)
        test_df = self.concat_partitions(test_paths)

        columns = ["fold"] + self.group_columns
        validation_df = validation_df.groupby(columns)[list_metrics()].mean().reset_index()
        test_df = test_df.groupby(columns)[list_metrics()].mean().reset_index()
        df = validation_df.merge(test_df, on=columns, suffixes=("_validation", "_test"))
        df.to_parquet(path)

        self.delete_partitions(validation_paths)
        self.delete_partitions(test_paths)
        self.next(self.end)

    @step
    def end(self):
        pass

    def evaluate(self, fold, train_df, test_df, parameters):
        model = EASE(train_df, parameters["regularization_rate"])
        prediction_df = get_predictions(model, test_df, None)
        evaluation_df = evaluate_predictions(prediction_df, groupby=self.group_columns)
        evaluation_df["fold"] = fold
        return evaluation_df

    @staticmethod
    def concat_partitions(paths):
        partitions = []

        for path in paths:
            partition_df = pd.read_parquet(path)
            partitions.append(partition_df)

        return pd.concat(partitions)

    @staticmethod
    def delete_partitions(paths):
        for path in paths:
            Path(path).unlink()


if __name__ == '__main__':
    EvaluateEASE()
