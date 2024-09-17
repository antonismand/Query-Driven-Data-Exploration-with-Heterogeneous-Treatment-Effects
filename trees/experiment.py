import inspect
from itertools import product
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from IPython.display import clear_output


from trees.brute_force import BruteForce
from trees.causal_forest import CF, CFT
from trees.causal_tree import CT
from trees.data import Data


class Experiment:
    def __init__(
        self,
        var_name: str,
        data_iterations=5,
        user_iterations=5,
        **kwargs,
    ):
        scores = pd.DataFrame()

        data_params = self.get_signature(Data().generate, kwargs)
        data_keys = list(data_params.keys())

        condition_params = self.get_signature(Data().generate_random_condition, kwargs)
        condition_keys = list(condition_params.keys())

        topK_params = self.get_signature(Data().get_topK, kwargs)
        topK_keys = list(topK_params.keys())

        overlap_params = self.get_signature(Data().set_overlap_measure, kwargs)
        overlap_keys = list(overlap_params.keys())

        for data_values in product(*data_params.values()):
            param_dict = dict(zip(data_keys, data_values))

            for exp in tqdm(range(data_iterations)):
                data = Data()
                param_dict["seed"] = exp
                print("Generating data", param_dict)
                data.generate(**param_dict)

                algorithms = self.get_algorithms(data)

                for alg in algorithms:
                    print("fitting", alg.algorithm)
                    alg.fit()

                for j in range(user_iterations):
                    for condition_values in product(*condition_params.values()):
                        condition_dict = dict(zip(condition_keys, condition_values))
                        print("Generating random condition", condition_dict)
                        data.generate_random_condition(**condition_dict)

                        for overlap_values in product(*overlap_params.values()):
                            overlap_dict = dict(zip(overlap_keys, overlap_values))
                            print("Setting overlap", overlap_dict)
                            data.set_overlap_measure(**overlap_dict)

                            for alg in algorithms:
                                print("scanning", alg.algorithm)
                                alg.scan()

                                for topK_values in product(*topK_params.values()):
                                    topK_dict = dict(zip(topK_keys, topK_values))
                                    print("get topK", alg.algorithm, topK_dict)
                                    score = data.get_topK(alg=alg, **topK_dict)
                                    current_combination = {
                                        **param_dict,
                                        **condition_dict,
                                        **overlap_dict,
                                        **topK_dict,
                                    }
                                    score[var_name] = current_combination[var_name]
                                    scores = pd.concat(
                                        [scores, score], ignore_index=True
                                    )

                clear_output(wait=True)

        print(
            "Finished",
            data_iterations,
            "data iterations with",
            user_iterations,
            "user interactions",
        )
        print("data params", data_params)
        print("condition params", condition_params)
        print("overlap params", overlap_params)
        print("topK params", topK_params)

        plots(var_name, scores)
        scores.to_csv(f"../csv/scores_{var_name}.csv", index=False)

    def get_algorithms(self, data):
        cf = CF(data)
        return [
            CT(data, on="D"),
            CT(data, on="P"),
            cf,
            CFT(cf),
            # BruteForce(cf, oracle=False),
            BruteForce(cf),
        ]

    def get_signature(self, func, kwargs):
        signature = inspect.signature(func)
        parameters = {}
        for param_name, param in signature.parameters.items():
            if param_name in kwargs:
                parameters[param_name] = kwargs[param_name]
            elif param.default is not param.empty:
                parameters[param_name] = [param.default]

        return parameters


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
    sns.barplot(x=param_name, y="overlap", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"Overlap to P (higher is better)")
    axs[0].set_ylabel("overlap")
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="coverage", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Coverage")
    # axs[1].set_ylabel("coverage")
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
