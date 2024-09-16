from time import time
from causalml.inference.tree import CausalTreeRegressor
import matplotlib.pyplot as plt
from causalml.inference.tree.plot import plot_causal_tree
import pandas as pd
from trees.data import Data


class CT:
    def __init__(self, D: Data, on="D", debug=False):
        self.scores = []
        self.D = D
        self.on = on
        self.debug = debug

        if on == "D":
            self.df = D.df
        else:
            self.df = D.q_df

        self.algorithm = "CT on " + self.on
        self.scan_time = 0

    def fit(self, max_depth=6):
        if self.on == "P":
            start = time()
        self.ctree: CausalTreeRegressor = CausalTreeRegressor(
            groups_cnt=True, max_depth=max_depth
        )
        self.ctree.fit(
            X=self.df[self.D.feature_names].to_numpy(),
            y=self.df["outcome"].to_numpy(),
            treatment=self.df["treatment"].to_numpy(),
        )
        self.tree = self.ctree.tree_
        if self.on == "P":
            self.scan_time += round(time() - start, 2)

    def scan(self):
        start = time()
        self.parse_tree()
        end = time()
        self.scan_time += round(end - start, 2)

        if self.debug:
            print(f"Tree Scan: {self.scan_time}")
            print("Total options: ", len(self.scores))

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

            df2 = self.D.execute(full_condition)  # why self.on HERE?!
            overlap = round(self.D.overlap_measure(self.D.q_df, df2), 3)

            if print_tree:
                print(
                    f"{depth * '  '}{full_condition},  CATE: {cate}, overlap: {overlap}"
                )
            self.scores.append(
                {
                    "condition": full_condition,
                    "t_est": abs(cate),
                    "overlap": overlap,
                    "T0": round(self.tree.value[node_id][0][0], 2),
                    "T1": round(self.tree.value[node_id][1][0], 2),
                    "depth": depth,
                    "rows": df2.shape[0],
                    "selectivity": df2.shape[0] / self.D.df.shape[0],
                    "selectivity_to_P_ratio": df2.shape[0] / self.D.q_df.shape[0],
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
