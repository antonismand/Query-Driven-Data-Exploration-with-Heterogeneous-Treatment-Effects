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

        self.algorithm = "CT on " + self.on
        # self.fit()

    def fit(self, max_depth=6):
        self.ctree: CausalTreeRegressor = CausalTreeRegressor(
            groups_cnt=True, max_depth=max_depth
        )
        self.ctree.fit(
            X=self.df[self.D.feature_names].to_numpy(),
            y=self.df["outcome"].to_numpy(),
            treatment=self.df["treatment"].to_numpy(),
        )
        self.tree = self.ctree.tree_

        self.parse_tree()

    def plot_tree(self, max_depth=6):
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
        max_depth=10,
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
            distance = round(self.D.jaccard_distance(self.D.q_df, df2), 3)

            if print_tree:
                print(
                    f"{depth * '  '}{full_condition},  CATE: {cate}, distance: {distance}"
                )
            self.scores.append(
                {
                    "condition": full_condition,
                    "t_est": abs(cate),
                    "distance": distance,
                    "T0": round(self.tree.value[node_id][0][0], 2),
                    "T1": round(self.tree.value[node_id][1][0], 2),
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

    def get_topK(self, w=0.6, k=5, print_summaries=False):
        top = self.D.get_topK(self.scores, w=w, k=k, print_summaries=print_summaries)

        top["algorithm"] = self.algorithm

        return top
