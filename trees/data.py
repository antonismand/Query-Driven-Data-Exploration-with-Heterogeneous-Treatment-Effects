from time import time
from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
import pandas as pd

from trees.parser import Predicate, format_interval


class Data:
    def __init__(self):
        pass

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

    def get_topK(self, alg, w=0.6, k=5, max_overlap_duplicate=0.8):
        start = time()
        scores = pd.DataFrame(alg.scores)
        scores["norm_cate"] = scores["t_est"] / scores["t_est"].max()
        scores["score"] = w * scores["norm_cate"] + (1 - w) * scores["overlap"]

        scores = scores.sort_values("score", ascending=False)

        top = self.remove_duplicates(
            scores, k=k, max_overlap_duplicate=max_overlap_duplicate
        )
        top.reset_index(drop=True, inplace=True)

        ids = set()
        total = 0
        for i, row in top.iterrows():
            top.loc[i, "t"] = self.CATE(row["condition"])
            top.loc[i, "features"] = len(
                set(row["condition"].split()) & set(self.feature_names)
            )

            executed = self.execute(row["condition"], on="D")["id"]
            ids.update(executed)
            # print(
            #     f"{alg.algorithm} K={i+1}, rows={len(executed)}, new unique={len(ids)}"
            # )
            total += row["rows"]

        top["unique_between_K"] = len(ids) / total

        # best_t = top["t"].max()
        top["true_score"] = w * top["t"] / self.max_t + (1 - w) * top["overlap"]

        # rounding
        top["true_score"] = top["true_score"].apply(lambda x: round(x, 2))
        top["score"] = top["score"].apply(lambda x: round(x, 2))
        top["norm_cate"] = top["norm_cate"].apply(lambda x: round(x, 2))

        end = time()
        top["algorithm"] = alg.algorithm
        top["execution_time"] = round(end - start, 2) + alg.scan_time

        if alg.debug:
            print(f"Time to retrieve top-K {round(end - start, 2)}")

        return top

    def get_valid_subgroups(self, p: str, options: list, min_rows=5):
        """
        Get valid subgroups based on the provided Predicate.

        Args:
            P (str): The user's predicate in string format.
            options (list): The list of subgroups to evaluate.
        """

        pp = Predicate(f"select * from x where {p}")
        accepted_subgroups = []
        for opt in options:
            s = Predicate("select * from x where " + opt["condition"], combine=True)
            if s.satisfies(pp):
                r = f"{p} AND {opt['condition']}"
                df = self.execute(r)
                if df.shape[0] > min_rows:
                    opt["rows"] = df.shape[0]
                    opt["combined"] = {
                        k: format_interval(v) for k, v in s.combined.items()
                    }
                    opt["features"] = len(s.combined)
                    accepted_subgroups.append(opt)

        return accepted_subgroups
