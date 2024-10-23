from time import time
from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
import pandas as pd


from tqdm import tqdm

from trees.parser import Predicate
from copy import deepcopy


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

        if override_p_with_2:
            self.feature_names = ["feature_0", "feature_1"]
            df = df[["feature_0", "feature_1", "outcome", "treatment", "ITE", "id"]]

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

    def generate_random_condition(self, min_s=0.3, max_s=0.95, print_condition=True):
        while True:
            p = np.random.choice(self.feature_names)
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

    def get_topK(self, alg, w=0.5, k=5):
        if self.p is None:
            raise ValueError("no P given")

        self.w = w
        self.k = k

        n_options = len(alg.valid_options)

        start = time()
        subgroups = alg.scan_method(
            self, deepcopy(alg.valid_options), alg.max_t, alg.op_matrix
        )
        end = time()
        scan_time = round(end - start, 2)
        # print(scan_method.__name__, "time:", scan_time)

        for s in subgroups:

            s["t"] = self.CATE(s["condition"])
            # comb = " AND ".join(
            #     [f"{k} ∈ {format_interval(v)}" for k, v in s["combined"].items()]
            # )
            # s["combined"] = comb
            s["scan_method"] = alg.scan_method.__name__
            s["scan_time"] = scan_time
            s["valid_options"] = n_options
            # s['true_score'] = w * s["t"] / self.max_t + (1 - w) * s["overlap_penalty"]

        return pd.DataFrame(subgroups)

    def get_valid_subgroups(self, options: list, min_rows=5):
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
        for opt in options:
            if self.pp.includes(opt["combined"]):
                r = f"{self.p} AND {opt['condition']}"
                df = self.execute(r)
                if df.shape[0] > min_rows:
                    opt["rows"] = df.shape[0]
                    opt["total_options"] = n_options
                    accepted_subgroups.append(opt)
                    if opt["t_est"] > max_t:
                        max_t = opt["t_est"]

        end = time()
        for accepted in accepted_subgroups:
            accepted["validate_time"] = round(end - start, 2)

        return accepted_subgroups, max_t

    def compute_scores_for_subgroups(
        self, subgroups: list, max_t: float, op_matrix: list
    ):

        total_score = 0

        for i, s1 in enumerate(subgroups):
            overlap = 0
            for j, s2 in enumerate(subgroups):
                if i != j:
                    overlap += op_matrix[s1["id"]][s2["id"]]

            s1["overlap"] = overlap / self.k
            s1["score"] = self.w * s1["t_est"] / max_t + (1 - self.w) * (
                1 - s1["overlap"]
            )
            total_score += s1["score"]

        return subgroups, total_score

    def compute_score_for_subgroup(
        self, top_subgroups: list, s, max_t: float, op_matrix: list
    ):
        overlap = (
            sum([op_matrix[top_sub["id"]][s["id"]] for top_sub in top_subgroups])
            / self.k
        )

        s["overlap"] = overlap
        s["score"] = self.w * s["t_est"] / max_t + (1 - self.w) * (1 - overlap)

        return s

    def compute_overlap_matrix(self, options):
        n_options = len(options)
        op_matrix = [[0] * n_options] * n_options
        for i in tqdm(range(n_options)):
            for j in range(i, n_options):
                if i == j:
                    op_matrix[i][j] = 1
                else:
                    overlap = self.jaccard(
                        options[i]["combined"], options[j]["combined"]
                    )
                    op_matrix[i][j] = overlap
                    op_matrix[j][i] = overlap
        return op_matrix

    def intersection_range(self, interval1: tuple, interval2: tuple):
        return (max(interval1[0], interval2[0]), min(interval1[1], interval2[1]))

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
