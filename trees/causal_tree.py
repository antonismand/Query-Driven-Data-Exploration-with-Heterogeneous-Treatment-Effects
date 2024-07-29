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

    def parse_tree(
        self,
        node_id=0,
        parent_id=None,
        left=False,
        prev_conditions="",
        depth=0,
        max_depth=4,
        print_tree=False,
    ):

        cate = round(self.tree.value[node_id][1][0] - self.tree.value[node_id][0][0], 3)

        full_condition = ""

        if parent_id is not None:
            cond = "<=" if left else ">"

            full_condition = f"feature_{self.tree.feature[parent_id]} {cond} {round(self.tree.threshold[parent_id],3)}"
            if prev_conditions != "":
                full_condition = f"{prev_conditions} AND {full_condition}"

            df2 = self.D.execute(full_condition, on=self.on)
            distance = round(self.D.jaccard_distance(self.D.q_df, df2), 2)

            if print_tree:
                print(
                    f"{depth * '  '}{full_condition},  CATE: {cate}, distance: {distance}"
                )
            self.scores.append(
                {
                    "condition": full_condition,
                    "t_est": abs(cate),
                    "distance": distance,
                    "T0": self.tree.value[node_id][0][0],
                    "T1": self.tree.value[node_id][1][0],
                    "depth": depth,
                }
            )
        else:
            if print_tree:
                print(rf"Root $\hat{{\tau}}(x)$: {cate}")

        if self.tree.children_left[node_id] != -1 and depth < max_depth:
            self.parse_tree(
                node_id=self.tree.children_left[node_id],
                parent_id=node_id,
                left=True,
                prev_conditions=full_condition,
                depth=depth + 1,
                max_depth=max_depth,
                print_tree=print_tree,
            )
            self.parse_tree(
                node_id=self.tree.children_right[node_id],
                parent_id=node_id,
                left=False,
                prev_conditions=full_condition,
                depth=depth + 1,
                max_depth=max_depth,
                print_tree=print_tree,
            )

    def get_topK(self, w=0.6, k=5, print_summaries=True):
        scores = pd.DataFrame(self.scores)
        scores["norm_cate"] = scores["t_est"] / scores["t_est"].max()
        scores["score"] = w * scores["norm_cate"] + (1 - w) * scores["distance"]
        scores["algorithm"] = "CT on " + self.on

        top = scores.sort_values("score", ascending=False).head(k)

        for i, row in top.iterrows():
            top.loc[i, "t"] = self.D.CATE(row["condition"])

        if print_summaries:
            print(
                "Score mean:",
                round(top["score"].mean(), 2),
                "±",
                round(top["score"].std(), 2),
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
