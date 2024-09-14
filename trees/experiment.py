from itertools import product
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns

from trees.brute_force import BruteForce
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
        override_p_with_2=[False],
    ):
        scores = pd.DataFrame()

        data_params = [mode, n, p, sigma, s_min, s_max, override_p_with_2]
        get_params = [k, w]

        for mode, n, p, sigma, s_min, s_max, override_p_with_2 in product(*data_params):
            print(
                f"Running {iterations} iterations for mode={mode}, n={n}, p={p}, sigma={sigma}, s_min={s_min}, s_max={s_max}"
            )
            for exp in tqdm(range(iterations)):
                data = Data()
                data.generate(
                    n=n,
                    p=p,
                    sigma=sigma,
                    mode=mode,
                    seed=exp,
                    override_p_with_2=override_p_with_2,
                )
                data.generate_random_condition(min_s=s_min, max_s=s_max)

                cf = CF(data)
                algorithms = [
                    CT(data, on="D"),
                    CT(data, on="P"),
                    cf,
                    CFT(cf),
                    BruteForce(cf),
                ]
                for alg in algorithms:
                    alg.fit()
                    alg.scan()

                    for k, w in product(*get_params):
                        score = data.get_topK(alg=alg, w=w, k=k)
                        score[var_name] = locals()[var_name]
                        scores = pd.concat([scores, score], ignore_index=True)

        plots(var_name, scores)
        scores.to_csv(f"../csv/scores_{var_name}.csv", index=False)


def plots(param_name, scores):

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="t", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"True CATE (higher is better)")
    axs[0].set_ylabel(r"$\tau(x)$")
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="t_est", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Estimated CATE")
    axs[1].set_ylabel(r"$\hat{{\tau}}(x)$")
    # axs[1].legend(fontsize="x-small")
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="true_score", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"True score (higher is better)")
    axs[0].set_ylabel("True Score")
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="score", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Score")
    # axs[1].legend(fontsize="x-small")
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="distance", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"Overlap to P (higher is better)")
    axs[0].set_ylabel("overlap")
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="coverage", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Diversity (higher is better)")
    axs[1].set_ylabel("diversity")
    # plt.legend(fontsize="x-small")
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="depth", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"Depth")
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="features", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Features")
    # axs[1].legend(fontsize="x-small")
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="selectivity", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"Selectivity")
    axs[0].legend(fontsize="x-small")

    sns.barplot(
        x=param_name,
        y="selectivity_to_P_ratio",
        hue="algorithm",
        data=scores,
        ax=axs[1],
    )
    axs[1].set_title(f"Selectivity Ratio compared to P's")
    # axs[1].legend(fontsize="x-small")
    axs[1].legend([], [], frameon=False)

    # sns.barplot(x=param_name, y="rows", hue="algorithm", data=scores, ax=axs[2])
    # axs[2].set_title(f"Rows")
    # axs[2].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    sns.barplot(x=param_name, y="execution_time", hue="algorithm", data=scores)
    plt.ylabel("Execution Time (s)")
    plt.title(f"Execution time (lower is better)")
    plt.legend(fontsize="x-small")
    plt.show()
