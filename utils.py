from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
from causalml.inference.tree import CausalTreeRegressor
import matplotlib.pyplot as plt
from causalml.inference.tree.plot import plot_causal_tree
import pandas as pd


class Experiment:
    def __init__(self):
        self.scores = []

    def synth_data(self, n=1000, p=10, sigma=3.0, seed=42):
        np.random.seed(seed)

        Y, X, T, tau, _, _ = synthetic_data(
            mode=2, n=n, p=p, sigma=sigma
        )  # Simulate randomized trial: mode=2

        df = pd.DataFrame(X)
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df.columns = self.feature_names
        df["outcome"] = Y
        df["treatment"] = T
        df["ITE"] = tau
        df["id"] = df.index

        self.df = pl.DataFrame(df)
        self.ctx = pl.SQLContext(population=self.df, eager=True)
        return df.describe()

    def execute(self, conditions: str):
        return self.ctx.execute("select * from population where " + conditions)

    def user_condition(self, p: str):
        self.p = p
        self.p_df = self.execute(p)

    def calculate_selectivity(self, query: str):
        return self.ctx.execute(query).shape[0] / self.df.shape[0]

    def generate_random_condition(self, selectivity_threshold=0.1):
        while True:
            p = np.random.choice(self.feature_names)
            threshold = np.random.uniform(self.df[p].min(), self.df[p].max())
            left = np.random.choice([True, False])
            cond = "<=" if left else ">"
            full_cond = f"{p} {cond} {threshold}"

            if (
                self.calculate_selectivity(
                    f"select * from population where {full_cond}"
                )
                > selectivity_threshold
            ):
                return full_cond

    def fit_tree(self, fit_on="D"):
        self.ctree: CausalTreeRegressor = CausalTreeRegressor(groups_cnt=True)
        if fit_on == "D":
            self.ctree.fit(
                X=self.df[self.feature_names].to_numpy(),
                y=self.df["outcome"].to_numpy(),
                treatment=self.df["treatment"].to_numpy(),
            )
        else:
            self.ctree.fit(
                X=self.p_df[self.feature_names].to_numpy(),
                y=self.p_df["outcome"].to_numpy(),
                treatment=self.p_df["treatment"].to_numpy(),
            )
            self.ctx = pl.SQLContext(population=self.p_df, eager=True)

        self.tree = self.ctree.tree_

    def plot_tree(self, max_depth=4):
        plt.figure(figsize=(20, 20))
        plot_causal_tree(
            self.ctree, max_depth=max_depth, feature_names=self.feature_names
        )

    def jaccard_distance(self, df1: pl.DataFrame, df2: pl.DataFrame):
        intersection = df1.join(df2, how="inner", on="id").shape[0]
        union = df1.shape[0] + df2.shape[0] - intersection
        return intersection / union

    def print_tree(
        self,
        node_id=0,
        parent_id=None,
        left=False,
        prev_conditions="",
        depth=0,
        max_depth=4,
    ):

        cate = round(self.tree.value[node_id][1][0] - self.tree.value[node_id][0][0], 3)

        full_condition = ""

        if parent_id is not None:
            cond = "<=" if left else ">"

            full_condition = f"feature_{self.tree.feature[parent_id]} {cond} {round(self.tree.threshold[parent_id],3)}"
            if prev_conditions != "":
                full_condition = f"{prev_conditions} AND {full_condition}"

            df2 = self.execute(full_condition)
            distance = round(self.jaccard_distance(self.p_df, df2), 2)

            print(
                f"{depth * '  '}{full_condition},  CATE: {cate}, distance: {distance}"
            )
            self.scores.append(
                {
                    "condition": full_condition,
                    "cate": abs(cate),
                    "distance": distance,
                    "T0": self.tree.value[node_id][0][0],
                    "T1": self.tree.value[node_id][1][0],
                    "depth": depth,
                }
            )
        else:
            print(f"Root CATE: {cate}")

        if self.tree.children_left[node_id] != -1 and depth < max_depth:
            self.print_tree(
                self.tree.children_left[node_id],
                node_id,
                True,
                full_condition,
                depth + 1,
            )
            self.print_tree(
                self.tree.children_right[node_id],
                node_id,
                False,
                full_condition,
                depth + 1,
            )

    def get_topK(self, w=0.6, k=5):
        scores = pd.DataFrame(self.scores)
        scores["norm_cate"] = scores["cate"] / scores["cate"].max()
        scores["score"] = w * scores["norm_cate"] + (1 - w) * scores["distance"]

        top = scores.sort_values("score", ascending=False).head(k)

        print(
            "Score mean:",
            round(top["score"].mean(), 2),
            "±",
            round(top["score"].std(), 2),
        )

        print(
            "CATE mean:",
            round(top["cate"].mean(), 2),
            "±",
            round(top["cate"].std(), 2),
        )

        print(
            "Distance mean:",
            round(top["distance"].mean(), 2),
            "±",
            round(top["distance"].std(), 2),
        )

        print("Depth:", round(top["depth"].mean(), 1))

        return top
