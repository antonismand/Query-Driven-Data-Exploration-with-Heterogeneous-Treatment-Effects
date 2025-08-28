from time import time
from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
import pandas as pd
from trees.params import DATA, P
from trees.parser import Predicate
from loguru import logger


class Data:
    def __init__(self):
        self.p = None

    def generate(
        self,
        n=DATA.N_ROWS,
        p=DATA.N_FEATURES,
        sigma=DATA.SIGMA,
        seed=DATA.SEED,
        mode=DATA.MODE,
        # override_p_with_2=False,
    ):
        np.random.seed(seed)
        self.mode = mode

        if mode == "OTC":
            return self.ab_data()
        elif mode in ["CU-visit", "CU-conversion"]:
            return self.uplift_data()

        if mode == "RCT":
            self.hte_features = ["feature_0", "feature_1"]
            mode = 2
        elif mode == "UTC":
            self.hte_features = ["feature_" + str(i) for i in range(5)]
            mode = 4
        elif mode == "UTC-f1-f4":  # missing first
            self.hte_features = ["feature_" + str(i) for i in range(1, 5)]
            mode = 4
        elif mode == "UTC-f0-f9":  # all
            self.hte_features = ["feature_" + str(i) for i in range(p)]
            mode = 4
        # elif mode == 5:
        #     self.hte_features = ["feature_" + str(i) for i in range(p)]

        Y, X, T, tau, _, _ = synthetic_data(mode=mode, n=n, p=p, sigma=sigma)

        df = pd.DataFrame(X)
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df.columns = self.feature_names
        df["outcome"] = Y
        df["treatment"] = T
        df["ITE"] = tau
        df["id"] = df.index
        self.max_t = df["ITE"].max()
        self.p = None
        self.Z = None

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

    def ab_data(self):

        ab_data = pd.read_csv(DATA.AB_DATA)

        ab_data.rename(columns={"easier_signup": "instrument"}, inplace=True)
        ab_data.rename(columns={"became_member": "treatment"}, inplace=True)
        ab_data.rename(columns={"days_visited_post": "outcome"}, inplace=True)

        self.feature_names = [
            col
            for col in ab_data.columns
            if col not in {"instrument", "treatment", "outcome"}
        ]

        self.hte_features = [
            "days_visited_free_pre",
            "days_visited_hs_pre",
            "os_type_osx",
        ]

        def TE_fn(X):
            return (
                0.2
                + 0.3 * X["days_visited_free_pre"]
                - 0.2 * X["days_visited_hs_pre"]
                + X["os_type_osx"]
            ).values

        ab_data["ITE"] = TE_fn(ab_data)
        ab_data["id"] = ab_data.index
        self.max_t = ab_data["ITE"].max()
        self.p = None

        self.rest_features = [
            f for f in self.feature_names if f not in self.hte_features
        ]

        # if override_p_with_2:
        #     self.feature_names = ["feature_0", "feature_1"]
        #     df = df[["feature_0", "feature_1", "outcome", "treatment", "ITE", "id"]]

        self.df = ab_data.copy()
        # self.df = pl.DataFrame(ab_data)

        min_values = ab_data[self.feature_names].min()
        max_values = ab_data[self.feature_names].max()

        self.min_max = {
            column: (round(min_values[column], 3), round(max_values[column], 3))
            for column in self.feature_names
        }

        return ab_data.describe()

    def uplift_data(self):

        df = pd.read_csv(DATA.UPLIFT_DATA)
        # df = df.sample(frac=0.3, random_state=42)

        self.feature_names = list(df.columns[0:12])

        if self.mode == "CU-visit":
            df.rename(columns={"visit": "outcome"}, inplace=True)
            self.hte_features = ["f0", "f2", "f3", "f6", "f8", "f9"]
        else:
            df.rename(columns={"conversion": "outcome"}, inplace=True)
            self.hte_features = ["f2", "f3", "f4", "f6", "f8", "f9", "f10", "f11"]

        df["ITE"] = 1
        df["id"] = df.index
        self.max_t = 1
        self.p = None

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

    def execute(self, condition: str):
        ctx = pl.SQLContext(population=self.df, eager=True)
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
        self.q_df = self.execute(p)
        self.pp = Predicate(p, self)

    def n_rows(self, condition: str):
        return self.execute(condition).shape[0]

    def calculate_selectivity(self, condition: str):
        return self.n_rows(condition) / self.df.shape[0]

    def generate_random_condition(
        self, min_s=P.MIN_S, max_s=P.MAX_S, features_in_P=P.FEATURES_IN_P
    ):
        while True:
            if features_in_P == "all":
                features = self.feature_names
            elif features_in_P == "hte_only":
                features = self.hte_features
            elif isinstance(features_in_P, list):
                features = features_in_P
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
                logger.debug(f"User condition: {full_cond}, Selectivity: {s}")
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

    # def get_valid_subgroups(self, subgroups: dict, min_rows=DATA.N_MIN_ROWS):
    #     """
    #     Get valid subgroups based on the provided Predicate.

    #     Args:
    #         P (str): The user's predicate in string format (WHERE only).
    #         subgroups (dict): The subgroups to evaluate.
    #     """
    #     if self.p is None:
    #         raise ValueError("no P given")

    #     start = time()
    #     # n_subgroups = len(subgroups)

    #     valid_subgroups = []
    #     pruned_min_rows = 0
    #     pruned_not_subsets = 0
    #     valid_parents = set()
    #     for id, sub in reversed(subgroups.items()):
    #         if self.pp.includes(sub["combined"]):
    #             if id not in valid_parents:
    #                 r = f"{self.p} AND {sub['condition']}"
    #                 df = self.execute(r)
    #                 if df.shape[0] > min_rows:
    #                     valid_subgroups.append(id)
    #                     valid_parents.update(sub["parents"])
    #                 else:
    #                     pruned_min_rows += 1
    #                     # print(opt["combined"], "not subset")
    #             else:
    #                 valid_subgroups.append(id)
    #         else:
    #             pruned_not_subsets += 1

    #     end = time()
    #     # for accepted in accepted_subgroups:
    #     #     accepted["validate_time"] = round(end - start, 2)
    #     #     accepted["pruned_min_rows"] = pruned_min_rows
    #     #     accepted["pruned_not_subsets"] = pruned_not_subsets

    #     logger.debug(
    #         f"Total subgroups: {len(subgroups)}, Valid subgroups: {len(valid_subgroups)}"
    #     )
    #     logger.debug(f"Filtered due to min rows: {pruned_min_rows}")
    #     logger.debug(f"Filtered due to not being subsets: {pruned_not_subsets}")
    #     logger.debug(f"Filtering Time: {round(end - start, 2)}")

    #     return valid_subgroups

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

    def interval_distance(self, interval1: tuple, interval2: tuple, full_range):
        L, U = full_range
        return (abs(interval1[0] - interval2[0]) + abs(interval1[1] - interval2[1])) / (
            2 * (U - L)
        )

    def dissimilarity(self, s1: dict[str, tuple], s2: dict[str, tuple]):
        total_distance = 0
        for attr in self.hte_features:
            interval1 = s1.get(attr, self.min_max[attr])
            interval2 = s2.get(attr, self.min_max[attr])
            total_distance += self.interval_distance(
                interval1, interval2, self.min_max[attr]
            )
        return total_distance / len(self.hte_features)

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
