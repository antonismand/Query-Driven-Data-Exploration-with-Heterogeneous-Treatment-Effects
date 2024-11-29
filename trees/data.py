from time import time
from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
import pandas as pd
from trees import params


from tqdm import tqdm

from trees.parser import Predicate
from copy import deepcopy


class Data:
    def __init__(self):
        self.p = None

    def generate(
        self,
        n=params.DATA.N_ROWS,
        p=params.DATA.N_FEATURES,
        sigma=params.DATA.SIGMA,
        seed=params.DATA.SEED,
        mode=params.DATA.MODE,
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

    def n_rows(self, condition: str):
        return self.execute(condition).shape[0]

    def calculate_selectivity(self, condition: str):
        return self.n_rows(condition) / self.df.shape[0]

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
            full_cond = self.random_condition_on_feature(p)

            if np.random.choice([True, False]):  # randomly add another condition
                p2 = np.random.choice(features)
                full_cond += " AND " + self.random_condition_on_feature(p2)

            s = self.calculate_selectivity(full_cond)

            if s > min_s and s < max_s:
                self.user_condition(full_cond)
                if print_condition:
                    print("User condition:", full_cond, "Selectivity:", s)
                return full_cond, s

    def random_condition_on_feature(self, feature):
        threshold = round(
            np.random.uniform(self.df[feature].min(), self.df[feature].max()), 3
        )
        if np.random.choice([True, False]):
            cond = "<="
        else:
            cond = ">"

        return f"{feature} {cond} {threshold}"

    def jaccard_distance(self, df1: pl.DataFrame, df2: pl.DataFrame):
        intersection = df1.join(df2, how="inner", on="id").shape[0]
        union = df1.shape[0] + df2.shape[0] - intersection
        return intersection / union

    def jaccard_over_preds(self, s1: str, s2: str):
        return self.jaccard_distance(self.execute(s1), self.execute(s2))

    def overlap_coefficient(self, df1: pl.DataFrame, df2: pl.DataFrame):
        intersection = df1.join(df2, how="inner", on="id").shape[0]
        return intersection / min(df1.shape[0], df2.shape[0])

    def get_valid_subgroups(self, subgroups: dict, min_rows=params.DATA.N_MIN_ROWS):
        """
        Get valid subgroups based on the provided Predicate.

        Args:
            P (str): The user's predicate in string format (WHERE only).
            subgroups (dict): The subgroups to evaluate.
        """
        if self.p is None:
            raise ValueError("no P given")

        start = time()
        # n_subgroups = len(subgroups)

        valid_subgroups = []
        pruned_min_rows = 0
        pruned_not_subsets = 0
        valid_parents = set()
        for id, sub in reversed(subgroups.items()):
            if self.pp.includes(sub["combined"]):
                if id not in valid_parents:
                    r = f"{self.p} AND {sub['condition']}"
                    df = self.execute(r)
                    if df.shape[0] > min_rows:
                        valid_subgroups.append(id)
                        valid_parents.update(sub["parents"])
                    else:
                        pruned_min_rows += 1
                        # print(opt["combined"], "not subset")
                else:
                    valid_subgroups.append(id)
            else:
                pruned_not_subsets += 1

        end = time()
        # for accepted in accepted_subgroups:
        #     accepted["validate_time"] = round(end - start, 2)
        #     accepted["pruned_min_rows"] = pruned_min_rows
        #     accepted["pruned_not_subsets"] = pruned_not_subsets
        print(
            "Total subgroups:", len(subgroups), "Valid subgroups:", len(valid_subgroups)
        )
        print("Filtered due to min rows:", pruned_min_rows)
        print("Filtered due to not being subsets:", pruned_not_subsets)
        print("Filtering Time:", round(end - start, 2))

        return valid_subgroups

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

    # def jaccard_between_intervals(self, interval1: tuple, interval2: tuple):
    #     intersection = self.intersection(interval1, interval2)
    #     union = interval1[1] - interval1[0] + interval2[1] - interval2[0] - intersection
    #     return intersection / union

    # def jaccard(self, c1: dict[str, tuple], c2: dict[str, tuple]):
    #     common_keys = c1.keys() & c2.keys()
    #     n_common = len(common_keys)

    #     if n_common == 0:
    #         return 0

    #     total = 0
    #     for common in common_keys:
    #         overlap = 0
    #         if c1[common] == c2[common]:
    #             overlap = 1
    #         else:
    #             overlap = self.jaccard_between_intervals(c1[common], c2[common])

    #         total += overlap

    #     return total / n_common
