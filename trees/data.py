from causalml.dataset import synthetic_data
import polars as pl
import numpy as np
import pandas as pd


class Data:
    def __init__(self):
        pass

    def generate(self, n=1000, p=10, sigma=3.0, seed=42):
        np.random.seed(seed)

        Y, X, T, tau, _, _ = synthetic_data(
            mode=2, n=n, p=p, sigma=sigma
        )  # Simulate randomized trial: mode=2

        df = pd.DataFrame(X)
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df.columns = self.feature_names
        df["outcome"] = Y
        df["treatment"] = T
        df["ITE"] = tau
        df["id"] = df.index

        self.df = pl.DataFrame(df)
        return df.describe()

    def execute(self, conditions: str, on="D"):
        population = self.df if on == "D" else self.q_df
        ctx = pl.SQLContext(population=population, eager=True)
        return ctx.execute("select * from population where " + conditions)

    def user_condition(self, p: str):
        self.p = p
        self.q_df = self.execute(p, on="D")

    def calculate_selectivity(self, query: str):
        return self.execute(query).shape[0] / self.df.shape[0]

    def generate_random_condition(self, selectivity_threshold=0.1):
        while True:
            p = np.random.choice(self.feature_names)
            threshold = np.random.uniform(self.df[p].min(), self.df[p].max())
            left = np.random.choice([True, False])
            cond = "<=" if left else ">"
            full_cond = f"{p} {cond} {threshold}"

            if self.calculate_selectivity(full_cond) > selectivity_threshold:
                self.user_condition(full_cond)
                # print("User condition:", full_cond)
                return full_cond
