from causalml.inference.tree import CausalTreeRegressor
import matplotlib.pyplot as plt
from causalml.inference.tree.plot import plot_causal_tree
from trees.data import Data
from trees.scanners import Greedy


class CT:
    def __init__(self, D: Data, scan_method=Greedy):
        self.options = []
        self.D = D
        self.df = D.df
        self.scan_method = scan_method
        self.algorithm = "CT on D" + f" ({scan_method.__name__})"

    def fit(self, parse_depth=100, max_depth=8, min_samples_leaf=50):
        self.parse_depth = parse_depth
        self.ctree: CausalTreeRegressor = CausalTreeRegressor(
            groups_cnt=True, max_depth=max_depth, min_samples_leaf=min_samples_leaf
        )
        self.ctree.fit(
            X=self.df[self.D.feature_names].to_numpy(),
            y=self.df["outcome"].to_numpy(),
            treatment=self.df["treatment"].to_numpy(),
        )
        self.tree = self.ctree.tree_
        self.parse_tree()
        self.op_matrix = self.D.compute_overlap_matrix(self.options)

    def online(self):
        self.valid_options, self.max_t = self.D.get_valid_subgroups(self.options)

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
        prev_combined: dict[str, tuple] = {},
        depth=0,
        print_tree=False,
    ):

        cate = round(self.tree.value[node_id][1][0] - self.tree.value[node_id][0][0], 3)

        full_condition = ""
        combined = prev_combined.copy()

        if parent_id is not None:
            # cond = "<=" if left else ">"
            feature = "feature_" + str(self.tree.feature[parent_id])
            num = round(self.tree.threshold[parent_id], 3)
            if left:
                cond = "<="
                interval = (self.D.min_max[feature][0], num)
            else:
                cond = ">"
                interval = (num, self.D.min_max[feature][1])

            if feature in combined:
                combined[feature] = self.D.intersection_range(
                    combined[feature], interval
                )
            else:
                combined[feature] = interval

            full_condition = f"{feature} {cond} {num}"
            if prev_conditions != "":
                full_condition = f"{prev_conditions} AND {full_condition}"

            if print_tree:
                print(f"{depth * '  '}{full_condition},  CATE: {cate}")
            self.options.append(
                {
                    "id": len(self.options),
                    "condition": full_condition,
                    "combined": combined,
                    "features": len(combined),
                    "t_est": abs(cate),
                    "T0": round(self.tree.value[node_id][0][0], 2),
                    "T1": round(self.tree.value[node_id][1][0], 2),
                    "depth": depth,
                    "algorithm": self.algorithm,
                }
            )
        else:
            if print_tree:
                print(rf"Root $\hat{{\tau}}(x)$: {cate}")

        if self.tree.children_left[node_id] != -1 and depth < self.parse_depth:
            self.parse_tree(
                node_id=self.tree.children_left[node_id],
                parent_id=node_id,
                left=True,
                prev_conditions=full_condition,
                prev_combined=combined,
                depth=depth + 1,
                print_tree=print_tree,
            )
            self.parse_tree(
                node_id=self.tree.children_right[node_id],
                parent_id=node_id,
                left=False,
                prev_conditions=full_condition,
                prev_combined=combined,
                depth=depth + 1,
                print_tree=print_tree,
            )


class CTP(CT):
    def __init__(self, D: Data, scan_method=Greedy):
        self.options = []
        self.D = D
        self.scan_method = scan_method
        self.algorithm = "CT on P" + f" ({scan_method.__name__})"

    def fit(self, parse_depth=100, max_depth=8, min_samples_leaf=50):
        self.parse_depth = parse_depth
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def online(self):
        if self.D.p is None:
            raise ValueError("P is not set")

        self.df = self.D.q_df
        self.options = []
        super().fit(
            parse_depth=self.parse_depth,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )
        self.valid_options, self.max_t = self.D.get_valid_subgroups(self.options)
