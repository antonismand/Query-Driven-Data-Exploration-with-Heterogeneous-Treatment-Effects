from time import time
from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
import pandas as pd

from trees.parser import Predicate, format_interval
from trees.scanners import Greedy


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

    def calculate_selectivity(self, condition: str):
        return self.execute(condition).shape[0] / self.df.shape[0]

    def set_overlap_measure(self, overlap_measure="jaccard_distance"):
        self.overlap_measure = getattr(self, overlap_measure)

    def generate_random_condition(self, min_s=0.3, max_s=0.95):
        while True:
            p = np.random.choice(self.feature_names)
            threshold = np.random.uniform(self.df[p].min(), self.df[p].max())
            left = np.random.choice([True, False])
            cond = "<=" if left else ">"
            full_cond = f"{p} {cond} {threshold}"

            s = self.calculate_selectivity(full_cond)

            if s > min_s and s < max_s:
                self.user_condition(full_cond)
                # print("User condition:", full_cond, "Selectivity:", s)
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

    # def overlap_measure(self, df1: pl.DataFrame, df2: pl.DataFrame):
    #     return self.overlap_coefficient(df1, df2)

    def remove_duplicates(self, scores, max_overlap_duplicate, k):
        accepted = [scores.iloc[0]]

        for i, row in scores.iterrows():
            df1 = self.execute(row["condition"])
            no_overlap = True
            for row2 in accepted:
                df2 = self.execute(row2["condition"])
                overlap = self.jaccard_distance(df1, df2)
                if overlap > max_overlap_duplicate:
                    no_overlap = False
                    break

            if no_overlap:
                accepted.append(row)
                if len(accepted) == k:
                    return pd.DataFrame(accepted)

        if len(accepted) != k:
            raise ValueError("Not enough recommendations to return")

    def get_topK(self, options: list, w=0.6, k=5, scan_method=Greedy, P=None):
        if P is None:
            if self.p is None:
                raise ValueError("no P given")
            P = self.p

        self.w = w
        self.k = k

        subgroups = self.get_valid_subgroups(P, options)

        start = time()
        subgroups = scan_method(self, subgroups)
        end = time()
        scan_time = round(end - start, 2)
        print(scan_method.__name__, "time:", scan_time)

        for s in subgroups:

            s["t"] = self.CATE(s["condition"])

            s["scan_method"] = scan_method.__name__
            s["scan_time"] = scan_time

            # s['true_score'] = w * s["t"] / self.max_t + (1 - w) * s["overlap_penalty"]

        return pd.DataFrame(subgroups)

    def get_valid_subgroups(self, p: str, options: list, min_rows=5):
        """
        Get valid subgroups based on the provided Predicate.

        Args:
            P (str): The user's predicate in string format (WHERE only).
            options (list): The list of subgroups to evaluate.
        """
        start = time()
        pp = Predicate(f"select * from x where {p}")
        accepted_subgroups = []
        for opt in options:
            s = Predicate("select * from x where " + opt["condition"], combine=True)
            if s.satisfies(pp):
                r = f"{p} AND {opt['condition']}"
                df = self.execute(r)
                if df.shape[0] > min_rows:
                    opt["rows"] = df.shape[0]
                    opt["combined"] = " AND ".join(
                        [f"{k} ∈ {format_interval(v)}" for k, v in s.combined.items()]
                    )
                    opt["features"] = len(s.combined)
                    accepted_subgroups.append(opt)

        end = time()
        print(
            f"Validating {len(options)} subgroups. Accepted subgroups: {len(accepted_subgroups)}. Time: {round(end - start, 2)}"
        )
        return accepted_subgroups

    def compute_topK_scores(self, subgroups: list):
        max_t = 0
        max_op = 0

        for i, s1 in enumerate(subgroups):
            overlap = 0
            for j, s2 in enumerate(subgroups):
                if i != j:
                    overlap += self.jaccard_over_preds(s1["condition"], s2["condition"])

            s1["overlap"] = overlap

            if s1["t_est"] > max_t:
                max_t = s1["t_est"]

            if s1["overlap"] > max_op:
                max_op = s1["overlap"]

        min_score = 999999994299999999
        min_score_i = -1

        for i, s in enumerate(subgroups):
            if max_op == 0:
                if s["overlap"] == 0:
                    op_penalty = 0
                else:
                    op_penalty = 1
            else:
                op_penalty = s["overlap"] / max_op

            s["score"] = self.w * s["t_est"] / max_t + (1 - self.w) * (1 - op_penalty)
            if s["score"] < min_score:
                min_score = s["score"]
                min_score_i = i

        return subgroups, max_t, max_op, min_score, min_score_i

    def compute_score_for_subgroup(self, top_subgroups: list, s, max_t, max_op):
        s_overlap = sum(
            [
                self.jaccard_over_preds(top_sub["condition"], s["condition"])
                for top_sub in top_subgroups
            ]
        )
        if max_op == 0:
            if s_overlap == 0:
                op_penalty = 0
            else:
                op_penalty = 1
        else:
            op_penalty = s_overlap / max_op

        score = self.w * s["t_est"] / max_t + (1 - self.w) * (1 - op_penalty)
        return score
