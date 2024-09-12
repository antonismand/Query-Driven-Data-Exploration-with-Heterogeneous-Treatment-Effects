from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
import pandas as pd


class Data:
    def __init__(self):
        pass

    def generate(
        self, n=10000, p=10, sigma=3.0, seed=42, mode=2, override_p_with_2=False
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

        if override_p_with_2:
            self.feature_names = ["feature_0", "feature_1"]
            df = df[["feature_0", "feature_1", "outcome", "treatment", "ITE", "id"]]

        self.df = pl.DataFrame(df)
        return df.describe()

    def execute(self, conditions: str, on="D"):
        population = self.df if on == "D" else self.q_df
        ctx = pl.SQLContext(population=population, eager=True)
        return ctx.execute("select * from population where " + conditions)

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

    def calculate_selectivity(self, query: str):
        return self.execute(query).shape[0] / self.df.shape[0]

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

    def get_topK(self, scores, w=0.6, k=5, print_summaries=True):
        scores = pd.DataFrame(scores)
        scores["norm_cate"] = scores["t_est"] / scores["t_est"].max()
        scores["score"] = w * scores["norm_cate"] + (1 - w) * scores["distance"]

        top = scores.sort_values("score", ascending=False).head(k)

        for i, row in top.iterrows():
            top.loc[i, "t"] = self.CATE(row["condition"])
            top.loc[i, "features"] = len(
                set(row["condition"].split()) & set(self.feature_names)
            )

        best_t = top["t"].max()
        top["true_score"] = w * top["t"] / best_t + (1 - w) * top["distance"]

        # rounding
        top["true_score"] = top["true_score"].apply(lambda x: round(x, 2))
        top["score"] = top["score"].apply(lambda x: round(x, 2))
        top["norm_cate"] = top["norm_cate"].apply(lambda x: round(x, 2))

        if print_summaries:
            print(
                "Score mean:",
                round(top["score"].mean(), 2),
                "±",
                round(top["score"].std(), 2),
            )

            print(
                "True Score mean:",
                round(top["true_score"].mean(), 2),
                "±",
                round(top["true_score"].std(), 2),
            )

            print(
                "Estimated CATE mean:",
                round(top["t_est"].mean(), 2),
                "±",
                round(top["t_est"].std(), 2),
            )
            print(
                "True CATE mean:",
                round(top["t"].mean(), 2),
                "±",
                round(top["t"].std(), 2),
            )

            print(
                "Distance mean:",
                round(top["distance"].mean(), 2),
                "±",
                round(top["distance"].std(), 2),
            )

            print("Depth:", round(top["depth"].mean(), 1))

        return top
