from time import time
from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
import pandas as pd

from trees.parser import Predicate, format_interval
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
        self.pp = Predicate(p)

    def calculate_selectivity(self, condition: str):
        return self.execute(condition).shape[0] / self.df.shape[0]

    def generate_random_condition(self, min_s=0.3, max_s=0.95):
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
        subgroups = alg.scan_method(self, deepcopy(alg.valid_options), alg.max_t)
        end = time()
        scan_time = round(end - start, 2)
        # print(scan_method.__name__, "time:", scan_time)

        for s in subgroups:

            s["t"] = self.CATE(s["condition"])
            comb = " AND ".join(
                [f"{k} ∈ {format_interval(v)}" for k, v in s["combined"].items()]
            )
            s["combined"] = comb
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

    def compute_scores_for_subgroups(self, subgroups: list, max_t: float):

        min_score = 999999994299999999
        min_score_i = -1
        total_score = 0

        for i, s1 in enumerate(subgroups):
            overlap = 0
            for j, s2 in enumerate(subgroups):
                if i != j:
                    overlap += self.jaccard_over_preds(s1["condition"], s2["condition"])

            s1["overlap"] = overlap / self.k
            s1["score"] = self.w * s1["t_est"] / max_t + (1 - self.w) * (
                1 - s1["overlap"]
            )
            total_score += s1["score"]
            if s1["score"] < min_score:
                min_score = s1["score"]
                min_score_i = i

        return subgroups, min_score, min_score_i, total_score

    def compute_score_for_subgroup(self, top_subgroups: list, s, max_t: float):
        s_overlap = (
            sum(
                [
                    self.jaccard_over_preds(top_sub["condition"], s["condition"])
                    for top_sub in top_subgroups
                ]
            )
            / self.k
        )

        return self.w * s["t_est"] / max_t + (1 - self.w) * (1 - s_overlap)
