from itertools import product, combinations
from time import time
import numpy as np
from trees.causal_forest import CF


class BruteForce:
    def __init__(self, cf: CF, oracle=True, min_rows=50, debug=False):

        self.D = cf.D
        self.df = cf.D.df
        self.cf = cf

        self.algorithm = "Brute Force"
        if oracle:
            self.algorithm += " (Oracle)"
        self.oracle = oracle
        self.options = []
        self.debug = debug
        self.min_rows = min_rows

    def acceptable_selectivity(self, cond):
        return self.D.execute(cond).shape[0] > self.min_rows
        # s = self.D.calculate_selectivity(cond)
        # return s > self.min_s and s < self.max_s

    def fit(self):
        important_features = self.cf.get_important_features()
        options = self.generate_combinations_multiple_features(important_features)

        self.scores = []
        for option in options:
            if self.oracle:
                t_est = self.D.CATE(option)
            else:
                t_est = self.cf.forest.effect(
                    self.D.execute(option)[self.D.feature_names]
                ).mean()

            self.scores.append({"condition": option, "t_est": t_est})

        if self.debug:
            print(f"Generated {len(self.scores)} options")

    def scan(self):
        start = time()
        for i, x in enumerate(self.scores):
            df2 = self.D.execute(x["condition"], on="D")
            overlap = round(self.D.overlap_measure(self.D.q_df, df2), 3)
            self.scores[i] = {
                "condition": x["condition"],
                "t_est": x["t_est"],
                "overlap": overlap,
                "depth": x["condition"].count("AND") + 1,
                "rows": df2.shape[0],
                "selectivity": df2.shape[0] / self.D.df.shape[0],
                "selectivity_to_P_ratio": df2.shape[0] / self.D.q_df.shape[0],
            }
        end = time()
        self.scan_time = round(end - start, 2)
        if self.debug:
            print(f"Scan time: {self.scan_time}")

    def generate_combinations_for_feature(
        self,
        feature,
        lower_quantile=0.005,
        upper_quantile=0.999,
        step=0.5,
    ):
        # start = self.df[feature].quantile(lower_quantile)
        # stop = self.df[feature].quantile(upper_quantile)
        start = self.df[feature].min()
        stop = self.df[feature].max()

        gtlt = np.arange(start, stop, step)

        combinations = []

        for x in gtlt:
            cond = f"{feature} > {round(x,2)}"
            if self.acceptable_selectivity(cond):
                combinations.append(cond)

            cond = f"{feature} < {round(x,2)}"
            if self.acceptable_selectivity(cond):
                combinations.append(cond)

        for x, y in list(product(gtlt, gtlt)):
            if x < y:
                cond = f"{feature} > {round(x,2)} AND {feature} < {round(y,2)}"
                if self.acceptable_selectivity(cond):
                    combinations.append(cond)

        return combinations

    def generate_combinations_multiple_features(self, features):
        combs = []
        f_combs = {}
        for feature in features:
            f = self.generate_combinations_for_feature(feature)
            if self.debug:
                print(f"{feature}:{len(f)}")
            f_combs[feature] = f
            combs.extend(f)

        for f1, f2 in list(combinations(features, 2)):
            for x in list(product(*[f_combs[f1], f_combs[f2]])):
                cond = " AND ".join(x)
                if self.acceptable_selectivity(cond):
                    combs.append(cond)

        return combs
