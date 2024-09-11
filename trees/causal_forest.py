from econml.dml import CausalForestDML
import pandas as pd
from tqdm import tqdm
from trees.data import Data
from time import time
from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from sklearn.ensemble import RandomForestClassifier


class CF:
    def __init__(self, D: Data):
        self.scores = []
        self.D = D
        self.df = D.df
        self.algorithm = "CF"

    def fit(
        self,
        criterion="mse",
        n_estimators=100,
        tune=False,
        max_depth=4,
        cv=2,
        min_samples_split=10,
        min_samples_leaf=5,
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
        )
        # print(
        #     f"n_estimators: {n_estimators}, criterion: {criterion}, max_depth: {max_depth}, cv: {cv}, min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}"
        # )

        if tune:
            self.forest.tune(
                X=self.df[self.D.feature_names].to_numpy(),
                Y=self.df["outcome"].to_numpy(),
                T=self.df["treatment"].to_numpy(),
            )
            self.algorithm += f" (tuned)"

        self.forest.fit(
            X=self.df[self.D.feature_names].to_numpy(),
            Y=self.df["outcome"].to_numpy(),
            T=self.df["treatment"].to_numpy(),
        )

        self.parse_forest()

    def parse_forest(self, max_depth=10):
        start = time()
        self.scores = []
        for tree in self.forest.model_cate.estimators_[0]:
            self.parse_tree(tree.tree_, max_depth=max_depth)
        end = time()
        self.execution_time = round(end - start, 2)

    def parse_tree(
        self,
        tree,
        node_id=0,
        parent_id=None,
        left=False,
        prev_conditions="",
        depth=0,
        max_depth=4,
    ):

        cate = round(tree.value[node_id][0][0], 3)

        full_condition = ""

        if parent_id is not None:
            cond = "<=" if left else ">"

            full_condition = f"feature_{tree.feature[parent_id]} {cond} {round(tree.threshold[parent_id],3)}"
            if prev_conditions != "":
                full_condition = f"{prev_conditions} AND {full_condition}"

            df2 = self.D.execute(full_condition, on="D")
            distance = round(self.D.jaccard_distance(self.D.q_df, df2), 2)

            add_the_new = True

            if "SingleTree" not in self.algorithm:
                for score in self.scores:  # search for duplicates
                    if abs(score["distance"] - distance) < 0.01:  # same distance from Q
                        df_similar = self.D.execute(score["condition"], on="D")

                        distance_between_similar = round(
                            self.D.jaccard_distance(df_similar, df2), 2
                        )  # distance between the conditions
                        if distance_between_similar < 0.01:
                            if score["t_est"] > cate:
                                add_the_new = False
                                break
                            else:
                                self.scores.remove(score)
                                break

            if add_the_new:
                self.scores.append(
                    {
                        "condition": full_condition,
                        "t_est": abs(cate),
                        "distance": distance,
                        "depth": depth,
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

    def get_topK(self, w=0.6, k=5, print_summaries=False):
        top = self.D.get_topK(self.scores, w=w, k=k, print_summaries=print_summaries)

        top["algorithm"] = self.algorithm
        top["execution_time"] = self.execution_time

        return top


class CFT(CF):
    def __init__(self, cf: CF):
        self.scores = []
        self.df = cf.df
        self.D = cf.D
        self.cf = cf

        self.algorithm = "CF SingleTree"

    def fit(self, max_depth=4, min_samples_leaf=10, print_tree=False):

        start = time()
        intrp = SingleTreeCateInterpreter(
            include_model_uncertainty=True,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        intrp.interpret(self.cf.forest, self.df[self.D.feature_names].to_numpy())

        if print_tree:
            plt.figure(figsize=(25, 5))
            intrp.plot(feature_names=self.D.feature_names, fontsize=12)

        self.parse_tree(intrp.tree_model_.tree_)
        end = time()
        self.execution_time = round(end - start, 2)


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
            cft.fit()

            for alg in [cf, cft]:
                score = alg.get_topK()
                score[param_name] = p
                scores = pd.concat([scores, score], ignore_index=True)

    plots(param_name, scores)
