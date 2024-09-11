from itertools import product
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns

from trees.causal_forest import CF, CFT
from trees.causal_tree import CT
from trees.data import Data


class Experiment:
    def __init__(
        self,
        var_name: str,
        mode=[2],
        n=[10000],
        p=[10],
        sigma=[3.0],
        s_min=[0.3],
        s_max=[0.95],
        w=[0.6],
        k=[5],
        iterations=30,
    ):
        scores = pd.DataFrame()

        data_params = [mode, n, p, sigma, s_min, s_max]
        get_params = [k, w]

        for mode, n, p, sigma, s_min, s_max in product(*data_params):
            print(
                f"Running {iterations} iterations for mode={mode}, n={n}, p={p}, sigma={sigma}, s_min={s_min}, s_max={s_max}"
            )
            for exp in tqdm(range(iterations)):
                data = Data()
                data.generate(n=n, p=p, sigma=sigma, mode=mode, seed=exp)
                data.generate_random_condition(min_s=s_min, max_s=s_max)

                cf = CF(data)
                algorithms = [CT(data, on="D"), CT(data, on="P"), cf, CFT(cf)]
                for alg in algorithms:
                    alg.fit()

                    for k, w in product(*get_params):
                        score = alg.get_topK(k=k, w=w)
                        score[var_name] = locals()[var_name]
                        scores = pd.concat([scores, score], ignore_index=True)

        plots(var_name, scores)
        scores.to_csv(f"../csv/scores_{var_name}.csv", index=False)


def plots(param_name, scores):
    sns.barplot(x=param_name, y="t", hue="algorithm", data=scores)
    plt.title(f"CATE (higher is better)")
    plt.ylabel(r"$\tau(x)$")
    plt.legend(fontsize="x-small")
    plt.show()

    sns.barplot(x=param_name, y="t_est", hue="algorithm", data=scores)
    plt.title(f"Estimated CATE")
    plt.ylabel(r"$\hat{{\tau}}(x)$")
    plt.legend(fontsize="x-small")
    plt.show()

    sns.barplot(x=param_name, y="score", hue="algorithm", data=scores)
    plt.title(f"Score (higher is better)")
    plt.legend(fontsize="x-small")
    plt.show()

    sns.barplot(x=param_name, y="true_score", hue="algorithm", data=scores)
    plt.title(f"True score (higher is better)")
    plt.legend(fontsize="x-small")
    plt.show()

    sns.barplot(x=param_name, y="distance", hue="algorithm", data=scores)
    plt.title(f"Distance (higher is better)")
    plt.legend(fontsize="x-small")
    plt.show()

    sns.barplot(x=param_name, y="depth", hue="algorithm", data=scores)
    plt.title(f"Depth (lower is better)")
    plt.legend(fontsize="x-small")
    plt.show()

    sns.barplot(x=param_name, y="features", hue="algorithm", data=scores)
    plt.title(f"Features")
    plt.legend(fontsize="x-small")
    plt.show()
