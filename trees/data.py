from time import time
from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
import pandas as pd


from tqdm import tqdm

from trees.parser import Predicate
from copy import deepcopy


MIN_ROWS = 10


class Data:
    def __init__(self):
        self.p = None

    def generate(
        self,
        n=10000,
        p=10,
        sigma=3.0,
        seed=42,
        mode=2,
        override_p_with_2=False,
    ):
        np.random.seed(seed)

        Y, X, T, tau, _, _ = synthetic_data(
            mode=mode, n=n, p=p, sigma=sigma
        )  # Simulate randomized trial: mode=2

        df = pd.DataFrame(X)
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df.columns = self.feature_names
        df["outcome"] = Y
        df["treatment"] = T
        df["ITE"] = tau
        df["id"] = df.index
        self.max_t = df["ITE"].max()
        self.p = None

        self.hte_features = ["feature_0", "feature_1"]  # This should be computed
        self.rest_features = [
            f for f in self.feature_names if f not in self.hte_features
        ]

        # if override_p_with_2:
        #     self.feature_names = ["feature_0", "feature_1"]
        #     df = df[["feature_0", "feature_1", "outcome", "treatment", "ITE", "id"]]

        self.df = pl.DataFrame(df)

        min_values = df[self.feature_names].min()
        max_values = df[self.feature_names].max()

        self.min_max = {
            column: (round(min_values[column], 3), round(max_values[column], 3))
            for column in self.feature_names
        }

        return df.describe()

    def execute(self, condition: str, on="D"):
        population = self.df if on == "D" else self.q_df
        ctx = pl.SQLContext(population=population, eager=True)
        return ctx.execute("select * from population where " + condition)

    def CATE(self, condition: str):
        ctx = pl.SQLContext(population=self.df, eager=True)
        return round(
            ctx.execute("select Abs(AVG(ITE)) from population where " + condition)[
                "ITE"
            ][0],
            3,
        )

    def user_condition(self, p: str):
        self.p = p
        self.q_df = self.execute(p, on="D")
        self.pp = Predicate(p, self)

    def calculate_selectivity(self, condition: str):
        return self.execute(condition).shape[0] / self.df.shape[0]

    def generate_random_condition(
        self, min_s=0.3, max_s=0.95, features_in_P="all", print_condition=True
    ):
        while True:
            if features_in_P == "all":
                features = self.feature_names
            elif features_in_P == "hte_only":
                features = self.hte_features
            else:
                features = self.rest_features
            p = np.random.choice(features)
            threshold = round(np.random.uniform(self.df[p].min(), self.df[p].max()), 3)

            if np.random.choice([True, False]):
                cond = "<="
            else:
                cond = ">"

            full_cond = f"{p} {cond} {threshold}"

            s = self.calculate_selectivity(full_cond)

            if s > min_s and s < max_s:
                self.user_condition(full_cond)
                if print_condition:
                    print("User condition:", full_cond, "Selectivity:", s)
                return full_cond, s

    def jaccard_distance(self, df1: pl.DataFrame, df2: pl.DataFrame):
        intersection = df1.join(df2, how="inner", on="id").shape[0]
        union = df1.shape[0] + df2.shape[0] - intersection
        return intersection / union

    def jaccard_over_preds(self, s1: str, s2: str):
        return self.jaccard_distance(self.execute(s1), self.execute(s2))

    def overlap_coefficient(self, df1: pl.DataFrame, df2: pl.DataFrame):
        intersection = df1.join(df2, how="inner", on="id").shape[0]
        return intersection / min(df1.shape[0], df2.shape[0])

    def get_valid_subgroups(self, options: list, min_rows=MIN_ROWS, both_checks=True):
        """
        Get valid subgroups based on the provided Predicate.

        Args:
            P (str): The user's predicate in string format (WHERE only).
            options (list): The list of subgroups to evaluate.
        """
        if self.p is None:
            raise ValueError("no P given")

        start = time()
        n_options = len(options)

        max_t = 0

        accepted_subgroups = []
        pruned_min_rows = 0
        pruned_not_subsets = 0
        for opt in deepcopy(options):
            r = f"{self.p} AND {opt['condition']}"
            df = self.execute(r)
            if df.shape[0] > min_rows:
                if (
                    both_checks and self.pp.includes(opt["combined"])
                ) or not both_checks:
                    opt["rows"] = df.shape[0]
                    opt["total_options"] = n_options
                    accepted_subgroups.append(opt)
                    if opt["t_est"] > max_t:
                        max_t = opt["t_est"]
                elif both_checks:
                    pruned_not_subsets += 1
                    # print(opt["combined"], "not subset")
            else:
                pruned_min_rows += 1

        end = time()
        for accepted in accepted_subgroups:
            accepted["validate_time"] = round(end - start, 2)
            accepted["pruned_min_rows"] = pruned_min_rows
            accepted["pruned_not_subsets"] = pruned_not_subsets

        print("Pruned due to min rows:", pruned_min_rows)
        print("Pruned due to not being subsets:", pruned_not_subsets)

        return accepted_subgroups, max_t

    def compute_overlap_matrix(self, options):
        n_options = len(options)
        op_matrix = np.zeros((n_options, n_options))

        for i in tqdm(range(n_options)):
            for j in range(i + 1, n_options):
                overlap = self.jaccard(options[i]["combined"], options[j]["combined"])
                op_matrix[i][j] = overlap
                op_matrix[j][i] = overlap

        return op_matrix

    def intersection_range(self, interval1: tuple, interval2: tuple):
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        if start > end:
            raise ValueError("No intersection")
        return (start, end)

    def is_subset(self, superset: tuple, subset: tuple):
        return superset[0] <= subset[0] and superset[1] >= subset[1]

    def intersection(self, interval1: tuple, interval2: tuple):
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        return end - start if start < end else 0

    def jaccard_between_intervals(self, interval1: tuple, interval2: tuple):
        intersection = self.intersection(interval1, interval2)
        union = interval1[1] - interval1[0] + interval2[1] - interval2[0] - intersection
        return intersection / union

    def jaccard(self, c1: dict[str, tuple], c2: dict[str, tuple]):
        common_keys = c1.keys() & c2.keys()
        n_common = len(common_keys)

        if n_common == 0:
            return 0

        total = 0
        for common in common_keys:
            overlap = 0
            if c1[common] == c2[common]:
                overlap = 1
            else:
                overlap = self.jaccard_between_intervals(c1[common], c2[common])

            total += overlap

        return total / n_common
