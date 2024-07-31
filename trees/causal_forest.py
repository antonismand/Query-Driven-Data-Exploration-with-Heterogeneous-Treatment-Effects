from econml.dml import CausalForestDML
import pandas as pd
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

        # self.fit()

    def fit(self, criterion="mse", n_estimators=100, tune=False):
        self.label = f"CF ({criterion})"
        self.forest: CausalForestDML = CausalForestDML(
            n_estimators=n_estimators,
            criterion=criterion,
            discrete_treatment=True,
            random_state=123,
            model_t=RandomForestClassifier(),
            model_y=WeightedLassoCVWrapper(),
        )

        start = time()

        if tune:
            self.forest.tune(
                X=self.df[self.D.feature_names].to_numpy(),
                Y=self.df["outcome"].to_numpy(),
                T=self.df["treatment"].to_numpy(),
            )
            self.label += f" (tuned)"

        self.forest.fit(
            X=self.df[self.D.feature_names].to_numpy(),
            Y=self.df["outcome"].to_numpy(),
            T=self.df["treatment"].to_numpy(),
        )
        end = time()
        self.execution_time = round(end - start, 2)

    def parse_forest(self):
        self.scores = []
        for tree in self.forest.model_cate.estimators_[0]:
            self.parse_tree(tree.tree_)

    def single_tree_interpreter(
        self, max_depth=4, min_samples_leaf=10, print_tree=False
    ):
        self.label += " SingleTree"
        intrp = SingleTreeCateInterpreter(
            include_model_uncertainty=True,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        intrp.interpret(self.forest, self.df[self.D.feature_names].to_numpy())

        if print_tree:
            plt.figure(figsize=(25, 5))
            intrp.plot(feature_names=self.D.feature_names, fontsize=12)

        self.scores = []
        self.parse_tree(intrp.tree_model_.tree_)

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

            if "SingleTree" not in self.label:
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

    def get_topK(self, w=0.6, k=5, print_summaries=True):
        scores = pd.DataFrame(self.scores)
        scores["norm_cate"] = scores["t_est"] / scores["t_est"].max()
        scores["score"] = w * scores["norm_cate"] + (1 - w) * scores["distance"]
        scores["algorithm"] = self.label
        scores["execution_time"] = self.execution_time

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
