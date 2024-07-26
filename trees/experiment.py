import pandas as pd
from tqdm import tqdm

from trees.causal_tree import CT
from trees.data import Data


class Experiment:
    def __init__(
        self,
        var_name: str,
        var_values: list,
        iterations=None,
        N=None,
        w=None,
        K=None,
        s=None,
    ):
        self.var = var_name
        self.scores = pd.DataFrame(
            columns=[
                "condition",
                "cate",
                "distance",
                "T0",
                "T1",
                "depth",
                "norm_cate",
                "score",
                "algorithm",
                var_name,
            ]
        )
        self.N = N
        self.w = w
        self.K = K
        self.s = s

        for x in var_values:
            print(f"Running experiments for {self.var}={x}")
            self.__dict__[self.var] = x
            for exp in tqdm(range(iterations)):
                data = Data()
                data.generate(n=self.N, seed=exp)
                data.generate_random_condition(selectivity_threshold=self.s)

                ct_D = CT(data, on="D")
                ct_D.parse_tree()
                score = ct_D.get_topK(k=self.K, w=self.w, print_summaries=False)
                score[self.var] = x
                self.scores = pd.concat([self.scores, score], ignore_index=True)

                ct_P = CT(data, on="P")
                ct_P.parse_tree()
                score2 = ct_P.get_topK(k=self.K, w=self.w, print_summaries=False)
                score2[self.var] = x
                self.scores = pd.concat([self.scores, score2], ignore_index=True)
