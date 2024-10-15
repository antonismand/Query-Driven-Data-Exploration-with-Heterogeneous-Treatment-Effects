from econml.dml import CausalForestDML
import pandas as pd
from tqdm import tqdm
from trees.data import Data
from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from sklearn.ensemble import RandomForestClassifier


class CF:
    def __init__(self, D: Data):
        self.options = []
        self.D = D
        self.df = D.df
        self.algorithm = "CF"

    def fit(
        self,
        criterion="mse",
        n_estimators=100,
        tune=False,
        max_depth=100,
        cv=2,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="auto",
    ):

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
                X=self.df[self.D.feature_names].to_numpy(),
                Y=self.df["outcome"].to_numpy(),
                T=self.df["treatment"].to_numpy(),
            )

        self.forest.fit(
            X=self.df[self.D.feature_names].to_numpy(),
            Y=self.df["outcome"].to_numpy(),
            T=self.df["treatment"].to_numpy(),
        )

        for tree in self.forest.model_cate.estimators_[0]:
            self.parse_tree(tree.tree_, max_depth=max_depth)

    def online(self):
        return self.D.get_valid_subgroups(self.options)

    def parse_tree(
        self,
        tree,
        node_id=0,
        parent_id=None,
        left=False,
        prev_conditions="",
        depth=0,
        max_depth=100,
    ):

        cate = round(tree.value[node_id][0][0], 3)

        full_condition = ""

        if parent_id is not None:
            cond = "<=" if left else ">"

            full_condition = f"feature_{tree.feature[parent_id]} {cond} {round(tree.threshold[parent_id],3)}"
            if prev_conditions != "":
                full_condition = f"{prev_conditions} AND {full_condition}"

            self.options.append(
                {
                    "condition": full_condition,
                    "t_est": abs(cate),
                    "depth": depth,
                    "algorithm": self.algorithm,
                }
            )

        if tree.children_left[node_id] != -1 and depth < max_depth:
            self.parse_tree(
                tree=tree,
                node_id=tree.children_left[node_id],
                parent_id=node_id,
                left=True,
                prev_conditions=full_condition,
                depth=depth + 1,
                max_depth=max_depth,
            )
            self.parse_tree(
                tree=tree,
                node_id=tree.children_right[node_id],
                parent_id=node_id,
                left=False,
                prev_conditions=full_condition,
                depth=depth + 1,
                max_depth=max_depth,
            )

    def get_important_features(self, threshold=0.01):
        important_features = []
        for i, fi in enumerate(self.forest.feature_importances_):
            # print(f"{self.D.feature_names[i]}: {fi}")
            if fi > threshold:
                important_features.append(self.D.feature_names[i])
        return important_features


class CFT(CF):
    def __init__(self, D: Data):
        self.options = []
        self.df = D.df
        self.D = D
        self.algorithm = "CF SingleTree"

    def fit(
        self,
        criterion="mse",
        n_estimators=100,
        tune=False,
        max_depth=100,
        cv=2,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="auto",
        print_tree=False,
    ):

        forest: CausalForestDML = CausalForestDML(
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
            forest.tune(
                X=self.df[self.D.feature_names].to_numpy(),
                Y=self.df["outcome"].to_numpy(),
                T=self.df["treatment"].to_numpy(),
            )

        forest.fit(
            X=self.df[self.D.feature_names].to_numpy(),
            Y=self.df["outcome"].to_numpy(),
            T=self.df["treatment"].to_numpy(),
        )

        intrp = SingleTreeCateInterpreter(
            include_model_uncertainty=True,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        intrp.interpret(forest, self.df[self.D.feature_names].to_numpy())

        if print_tree:
            plt.figure(figsize=(25, 5))
            intrp.plot(feature_names=self.D.feature_names, fontsize=12)

        final_tree = intrp.tree_model_.tree_
        self.parse_tree(final_tree)

    def online(self):
        return self.D.get_valid_subgroups(self.options)


def parameter_tuning(param_name, param_values, iterations=10):
    from trees.experiment import plots

    scores = pd.DataFrame()
    for p in param_values:
        print(f"{param_name}: {p}")
        for exp in tqdm(range(iterations)):
            data = Data()
            data.generate(seed=exp)
            data.generate_random_condition()

            cf = CF(data)
            cf.fit(**{param_name: p})

            cft = CFT(cf)
            if param_name in ["max_depth", "min_samples_leaf"]:
                cft.fit(**{param_name: p})
            else:
                cft.fit()

            for alg in [cf, cft]:
                alg.online()
                score = data.get_topK(alg)
                score[param_name] = p
                scores = pd.concat([scores, score], ignore_index=True)

    plots(param_name, scores)
