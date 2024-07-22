import polars as pl
from causalml.inference.tree import CausalTreeRegressor
import matplotlib.pyplot as plt
from causalml.inference.tree.plot import plot_causal_tree
import pandas as pd
from trees.data import Data


class CT:
    def __init__(
        self,
        D: Data,
        on="D",
    ):
        self.scores = []
        self.D = D
        self.on = on

        if on == "D":
            self.df = D.df
        else:
            self.df = D.q_df

        self.fit_tree()

    def fit_tree(self):
        self.ctree: CausalTreeRegressor = CausalTreeRegressor(groups_cnt=True)
        self.ctree.fit(
            X=self.df[self.D.feature_names].to_numpy(),
            y=self.df["outcome"].to_numpy(),
            treatment=self.df["treatment"].to_numpy(),
        )
        self.tree = self.ctree.tree_

    def plot_tree(self, max_depth=4):
        plt.figure(figsize=(20, 20))
        plot_causal_tree(
            self.ctree, max_depth=max_depth, feature_names=self.D.feature_names
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

            df2 = self.D.execute(full_condition, on=self.on)
            distance = round(self.jaccard_distance(self.D.q_df, df2), 2)

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
