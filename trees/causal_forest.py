from econml.dml import CausalForestDML
import pandas as pd
from tqdm import tqdm
from trees.data import Data
from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from sklearn.ensemble import RandomForestClassifier
from trees.scanners import Scanner
from trees import params


class CF:
    def __init__(self, scan_method=Scanner):
        self.options = []
        self.scan_method = scan_method
        self.algorithm = "CF" + f" ({scan_method.__name__})"

    def fit(
        self,
        D: Data,
        parse_depth=100,
        criterion=params.CF.CRITERION,
        n_estimators=params.CF.N_ESTIMATORS,
        tune=False,
        max_depth=params.CF.MAX_DEPTH,
        cv=params.CF.CV,
        min_samples_split=params.CF.MIN_SAMPLES_SPLIT,
        min_samples_leaf=params.CF.MIN_SAMPLES_LEAF,
        max_features=params.CF.MAX_FEATURES,
        train_in_all_features=False,
    ):
        self.D = D
        self.df = D.df
        self.parse_depth = parse_depth
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

        if train_in_all_features:
            self.features = self.D.feature_names
        else:
            self.features = self.D.hte_features

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

        self.op_matrix = self.D.compute_overlap_matrix(self.options)

    def online(self):
        self.valid_options, self.max_t = self.D.get_valid_subgroups(self.options)

    def parse_tree(
        self,
        tree,
        node_id=0,
        parent_id=None,
        left=False,
        prev_conditions="",
        prev_combined: dict[str, tuple] = {},
        depth=0,
    ):

        cate = round(tree.value[node_id][0][0], 3)

        full_condition = ""
        combined = prev_combined.copy()

        if parent_id is not None:
            # cond = "<=" if left else ">"
            num = round(tree.threshold[parent_id], 3)
            feature = "feature_" + str(tree.feature[parent_id])
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

            self.options.append(
                {
                    "id": len(self.options),
                    "condition": full_condition,
                    "combined": combined,
                    "features": len(combined),
                    "t_est": abs(cate),
                    "depth": depth,
                    "algorithm": self.algorithm,
                }
            )

        if tree.children_left[node_id] != -1 and depth < self.parse_depth:
            self.parse_tree(
                tree=tree,
                node_id=tree.children_left[node_id],
                parent_id=node_id,
                left=True,
                prev_conditions=full_condition,
                prev_combined=combined,
                depth=depth + 1,
            )
            self.parse_tree(
                tree=tree,
                node_id=tree.children_right[node_id],
                parent_id=node_id,
                left=False,
                prev_conditions=full_condition,
                prev_combined=combined,
                depth=depth + 1,
            )

    def get_important_features(self, threshold=0.01):
        important_features = []
        for i, fi in enumerate(self.forest.feature_importances_):
            print(f"{self.features[i]}: {fi}")
            if fi > threshold:
                important_features.append(self.features[i])
        return important_features


class CFT(CF):
    def __init__(self, scan_method=Scanner):
        self.options = []
        self.scan_method = scan_method
        self.algorithm = "CF SingleTree" + f" ({scan_method.__name__})"

    def fit(
        self,
        D: Data,
        parse_depth=100,
        criterion=params.CF.CRITERION,
        n_estimators=params.CF.N_ESTIMATORS,
        tune=False,
        max_depth=params.CF.MAX_DEPTH,
        cv=params.CF.CV,
        min_samples_split=params.CF.MIN_SAMPLES_SPLIT,
        min_samples_leaf=params.CF.MIN_SAMPLES_LEAF,
        max_features=params.CF.MAX_FEATURES,
        print_tree=False,
    ):
        self.df = D.df
        self.D = D
        self.parse_depth = parse_depth
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

        self.features = D.hte_features

        if tune:
            forest.tune(
                X=self.df[self.features].to_numpy(),
                Y=self.df["outcome"].to_numpy(),
                T=self.df["treatment"].to_numpy(),
            )

        forest.fit(
            X=self.df[self.features].to_numpy(),
            Y=self.df["outcome"].to_numpy(),
            T=self.df["treatment"].to_numpy(),
        )

        intrp = SingleTreeCateInterpreter(
            include_model_uncertainty=True,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        intrp.interpret(forest, self.df[self.features].to_numpy())

        if print_tree:
            plt.figure(figsize=(25, 5))
            intrp.plot(feature_names=self.features, fontsize=12)

        final_tree = intrp.tree_model_.tree_
        self.parse_tree(final_tree)
        self.op_matrix = self.D.compute_overlap_matrix(self.options)

    def online(self):
        self.valid_options, self.max_t = self.D.get_valid_subgroups(self.options)


def parameter_tuning(param_name, param_values, iterations=10, scan_method=Scanner):
    from trees.experiment import plots

    scores = pd.DataFrame()
    for p in param_values:
        print(f"{param_name}: {p}")
        for exp in tqdm(range(iterations)):

            D = Data()
            D.generate(seed=exp)
            D.generate_random_condition(print_condition=False)

            cf = CF(scan_method)
            cf.fit(D=D, **{param_name: p})
            cf.online()

            scanner = scan_method(cf.valid_options, cf.op_matrix, D)
            score = scanner.get_topK()
            score[param_name] = p
            scores = pd.concat([scores, score], ignore_index=True)

            cft = CFT(scan_method)
            cft.fit(D=D, **{param_name: p})
            cft.online()
            scanner = scan_method(cft.valid_options, cft.op_matrix, D)
            score = scanner.get_topK()
            score[param_name] = p
            scores = pd.concat([scores, score], ignore_index=True)

    plots(param_name, scores)
