import os
from econml.dml import CausalForestDML
import joblib
from loguru import logger
import pandas as pd
from tqdm import tqdm
from trees.data import Data
from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from sklearn.ensemble import RandomForestClassifier
from trees import params


class CF:
    def __init__(self):
        self.subgroups = {}
        self.is_online = False
        self.algorithm = "[Pretrained] CF"

    def fit(
        self,
        D: Data,
        criterion=params.CF.CRITERION,
        n_estimators=params.CF.N_ESTIMATORS,
        tune=False,
        max_depth=params.CF.MAX_DEPTH,
        cv=params.CF.CV,
        min_samples_split=params.CF.MIN_SAMPLES_SPLIT,
        min_samples_leaf=params.CF.MIN_SAMPLES_LEAF,
        max_features=params.CF.MAX_FEATURES,
        train_in_all_features=params.CF.TRAIN_IN_ALL_FEATURES,
    ):
        self.D = D
        self.df = D.df
        self.forest: CausalForestDML = CausalForestDML(
            n_estimators=n_estimators,
            criterion=criterion,
            discrete_treatment=True,
            random_state=123,
            model_t=RandomForestClassifier(),
            model_y=WeightedLassoCVWrapper(),
            max_depth=max_depth,
            cv=cv,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

        self.features = (
            self.D.feature_names if train_in_all_features else self.D.hte_features
        )

        if tune:
            self.forest.tune(
                X=self.df[self.features].to_numpy(),
                Y=self.df["outcome"].to_numpy(),
                T=self.df["treatment"].to_numpy(),
            )

        self.forest.fit(
            X=self.df[self.features].to_numpy(),
            Y=self.df["outcome"].to_numpy(),
            T=self.df["treatment"].to_numpy(),
        )

        for tree in self.forest.model_cate.estimators_[0]:
            self.parse_tree(tree.tree_)

    def online(self):
        self.online_time = 0
        # self.valid_subgroups = self.D.get_valid_subgroups(self.subgroups)

    def parse_tree(
        self,
        tree,
        node_id=0,
        parent_id=None,
        left=False,
        prev_conditions="",
        prev_combined: dict[str, tuple] = {},
        depth=0,
        parents: list = [],
    ):

        cate = round(tree.value[node_id][0][0], 3)

        full_condition = ""
        combined = prev_combined.copy()

        if parent_id is not None:
            # cond = "<=" if left else ">"
            num = round(tree.threshold[parent_id], 5)
            feature = self.features[tree.feature[parent_id]]
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

            self.subgroups[node_id] = {
                "id": node_id,
                "condition": full_condition,
                "combined": combined,
                "Features": len(combined),
                "t_est": abs(cate),
                "Depth": depth,
                "estimator": self.algorithm,
                "parents": parents[:],
            }

        if tree.children_left[node_id] != -1:
            self.parse_tree(
                tree=tree,
                node_id=tree.children_left[node_id],
                parent_id=node_id,
                left=True,
                prev_conditions=full_condition,
                prev_combined=combined,
                depth=depth + 1,
                parents=parents[:],
            )
            self.parse_tree(
                tree=tree,
                node_id=tree.children_right[node_id],
                parent_id=node_id,
                left=False,
                prev_conditions=full_condition,
                prev_combined=combined,
                depth=depth + 1,
                parents=parents[:],
            )

    def get_important_features(self, threshold=0.01):
        important_features = []
        for i, fi in enumerate(self.forest.feature_importances_):
            logger.info(f"{self.features[i]}: {fi}")
            if fi > threshold:
                important_features.append(self.features[i])
        return important_features


class CFT(CF):
    def __init__(self):
        self.subgroups = {}
        self.algorithm = "[Pretrained] CF"
        self.is_online = False

    def fit(
        self,
        D: Data,
        criterion=params.CF.CRITERION,
        n_estimators=params.CF.N_ESTIMATORS,
        tune=False,
        max_depth=params.CF.MAX_DEPTH,
        cv=params.CF.CV,
        min_samples_split=params.CF.MIN_SAMPLES_SPLIT,
        min_samples_leaf=params.CF.MIN_SAMPLES_LEAF,
        max_features=params.CF.MAX_FEATURES,
        train_in_all_features=params.CF.TRAIN_IN_ALL_FEATURES,
        print_tree=False,
    ):
        self.df = D.df
        self.D = D

        self.features = (
            self.D.feature_names if train_in_all_features else self.D.hte_features
        )

        model = os.path.join(
            os.path.dirname(__file__), "..", "models", f"CFT_{self.D.mode}.pkl"
        )

        if params.CF.USE_PRETRAINED and os.path.exists(model):
            final_tree = joblib.load(model)
            logger.info(f"Using pretrained model {model}")

        else:
            if params.CF.USE_PRETRAINED:
                logger.info(f"Pretrained model {model} not found. Training new model.")
            self.forest: CausalForestDML = CausalForestDML(
                n_estimators=n_estimators,
                criterion=criterion,
                discrete_treatment=True,
                random_state=123,
                model_t=RandomForestClassifier(),
                model_y=WeightedLassoCVWrapper(),
                max_depth=max_depth,
                cv=cv,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            )

            if tune:
                self.forest.tune(
                    X=self.df[self.features].to_numpy(),
                    Y=self.df["outcome"].to_numpy(),
                    T=self.df["treatment"].to_numpy(),
                )

            self.forest.fit(
                X=self.df[self.features].to_numpy(),
                Y=self.df["outcome"].to_numpy(),
                T=self.df["treatment"].to_numpy(),
            )

            intrp = SingleTreeCateInterpreter(
                include_model_uncertainty=True,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            )
            intrp.interpret(self.forest, self.df[self.features].to_numpy())

            if print_tree:
                plt.figure(figsize=(25, 5))
                intrp.plot(feature_names=self.features, fontsize=12)

            final_tree = intrp.tree_model_.tree_

            if params.CF.USE_PRETRAINED:
                joblib.dump(final_tree, model)
                logger.info(f"Model {model} saved.")
        self.parse_tree(final_tree)

    def online(self):
        self.online_time = 0
