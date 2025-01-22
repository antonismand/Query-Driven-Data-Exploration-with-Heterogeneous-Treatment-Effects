from copy import deepcopy
from time import time
from causalml.inference.tree import CausalTreeRegressor
from loguru import logger
import matplotlib.pyplot as plt
from causalml.inference.tree.plot import plot_causal_tree
import pandas as pd
from trees.data import Data
from trees import params


class CT:
    def __init__(self):
        self.subgroups = {}
        self.algorithm = "[Pretrained] CT"
        self.is_online = False

    def fit(
        self,
        D: Data,
        parse_depth=100,
        criterion=params.CT.CRITERION,
        max_depth=params.CT.MAX_DEPTH,
        min_samples_leaf=params.CT.MIN_SAMPLES_LEAF,
        on="D",
        train_in_all_features=params.CT.TRAIN_IN_ALL_FEATURES,
    ):
        self.D = D
        if on == "D":
            self.df = D.df
        else:
            self.df = D.q_df

        self.features = (
            self.D.feature_names if train_in_all_features else self.D.hte_features
        )

        self.parse_depth = parse_depth
        self.ctree: CausalTreeRegressor = CausalTreeRegressor(
            criterion=criterion,
            groups_cnt=True,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        self.ctree.fit(
            X=self.df[self.features].to_numpy(),
            y=self.df["outcome"].to_numpy(),
            treatment=self.df["treatment"].to_numpy(),
        )
        self.tree = self.ctree.tree_
        self.parse_tree()

    def online(self):
        self.online_time = 0
        # start = time()
        # self.valid_subgroups = self.D.get_valid_subgroups(self.subgroups)
        # self.online_time = round(time() - start, 2)

    def plot_tree(self, max_depth=6):
        plt.figure(figsize=(100, 20))
        plot_causal_tree(self.ctree, max_depth=max_depth, feature_names=self.features)
        plt.show()

    def feature_importances(self):
        return pd.DataFrame(
            {
                "importance": self.ctree.feature_importances_,
                "feature": self.features,
            }
        )

    def parse_tree(
        self,
        node_id=0,
        parent_id=None,
        left=False,
        prev_conditions="",
        prev_combined: dict[str, tuple] = {},
        depth=0,
        parents: list = [],
    ):

        cate = round(self.tree.value[node_id][1][0] - self.tree.value[node_id][0][0], 3)

        full_condition = ""
        combined = prev_combined.copy()

        if parent_id is not None:
            # cond = "<=" if left else ">"
            feature = "feature_" + str(self.tree.feature[parent_id])
            num = round(self.tree.threshold[parent_id], 5)
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

            full_condition = f"{feature} {cond} {num:.5f}"
            if prev_conditions != "":
                full_condition = f"{prev_conditions} AND {full_condition}"

            if parent_id != 0:
                parents.append(parent_id)

            logger.trace(f"{depth * '  '}{full_condition},  CATE: {cate}")
            # logger.trace(f"{depth * '  '}{node_id},  parents: {parents}")

            self.subgroups[node_id] = {
                "id": node_id,
                "condition": full_condition,
                "combined": combined,
                "features": len(combined),
                "t_est": abs(cate),
                # "T0": round(self.tree.value[node_id][0][0], 2),
                # "T1": round(self.tree.value[node_id][1][0], 2),
                "depth": depth,
                "algorithm": self.algorithm,
                "parents": parents[:],
            }

        else:
            logger.trace(rf"Root $\hat{{\tau}}(x)$: {cate}")

        if self.tree.children_left[node_id] != -1 and depth < self.parse_depth:
            self.parse_tree(
                node_id=self.tree.children_left[node_id],
                parent_id=node_id,
                left=True,
                prev_conditions=full_condition,
                prev_combined=combined,
                depth=depth + 1,
                parents=parents[:],
            )
            self.parse_tree(
                node_id=self.tree.children_right[node_id],
                parent_id=node_id,
                left=False,
                prev_conditions=full_condition,
                prev_combined=combined,
                depth=depth + 1,
                parents=parents[:],
            )


class CTP(CT):
    def __init__(self):
        self.subgroups = {}
        self.algorithm = "[Online] CT"
        self.is_online = True

    def fit(
        self,
        D: Data,
        parse_depth=100,
        max_depth=params.CT.MAX_DEPTH,
        min_samples_leaf=params.CT.MIN_SAMPLES_LEAF,
    ):
        self.D = D
        self.parse_depth = parse_depth
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def online(self):
        start = time()
        if self.D.p is None:
            raise ValueError("P is not set")

        self.subgroups = {}
        super().fit(
            D=self.D,
            parse_depth=self.parse_depth,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            on="P",
        )

        self.online_time = round(time() - start, 2)
