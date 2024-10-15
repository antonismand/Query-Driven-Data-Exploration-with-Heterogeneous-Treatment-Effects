import inspect
from itertools import product
from time import time
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from IPython.display import clear_output


from trees.causal_forest import CF, CFT
from trees.causal_tree import CT, CTP
from trees.data import Data


class Experiment:
    def __init__(
        self,
        var_name: str,
        data_iterations=5,
        user_iterations=5,
        save_csv=True,
        **kwargs,
    ):
        scores = pd.DataFrame()

        data_params = self.get_signature(Data().generate, kwargs)
        data_keys = list(data_params.keys())

        condition_params = self.get_signature(Data().generate_random_condition, kwargs)
        condition_keys = list(condition_params.keys())

        topK_params = self.get_signature(Data().get_topK, kwargs)
        topK_keys = list(topK_params.keys())

        for data_values in product(*data_params.values()):
            param_dict = dict(zip(data_keys, data_values))

            for exp in tqdm(range(data_iterations)):
                data = Data()
                param_dict["seed"] = exp
                print("-" * 20, "New data", "-" * 20)
                print("Generating data", param_dict)
                data.generate(**param_dict)

                algorithms = self.get_algorithms(data)

                for alg in algorithms:
                    start = time()
                    alg.fit()
                    end = round(time() - start, 2)
                    print(f"[{alg.algorithm}] Offline time: {end}s")

                for j in range(user_iterations):
                    for condition_values in product(*condition_params.values()):
                        condition_dict = dict(zip(condition_keys, condition_values))
                        print(
                            "-" * 10,
                            "Condition",
                            j + 1,
                            "out of",
                            user_iterations,
                            "-" * 10,
                        )
                        print("Generating random condition", condition_dict)
                        cond, s = data.generate_random_condition(**condition_dict)
                        print("User condition:", cond, "Selectivity:", s)

                        for alg in algorithms:
                            start = time()
                            valid_options = alg.online()
                            end = round(time() - start, 2)
                            print(
                                f"[{alg.algorithm}] Online time: {end}s, Total Splits: {len(alg.options)}, Valid: {len(valid_options)}"
                            )

                            for topK_values in product(*topK_params.values()):
                                topK_dict = dict(zip(topK_keys, topK_values))
                                start = time()
                                score = data.get_topK(
                                    options=valid_options, **topK_dict
                                )
                                end = round(time() - start, 2)
                                print(
                                    f"[{alg.algorithm}] get topK in {end}s - {str(topK_dict)}"
                                )

                                current_combination = {
                                    **param_dict,
                                    **condition_dict,
                                    **topK_dict,
                                }
                                score[var_name] = current_combination[var_name]
                                scores = pd.concat([scores, score], ignore_index=True)

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
        print("topK params", topK_params)

        plots(var_name, scores)
        if save_csv:
            scores.to_csv(f"../csv/scores_{var_name}.csv", index=False)

    def get_algorithms(self, data):
        return [
            CT(data),
            CTP(data),
            # CF(data),
            CFT(data),
            # BruteForce(cf, oracle=False),
            # BruteForce(cf),
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
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # sns.barplot(x=param_name, y="true_score", hue="algorithm", data=scores, ax=axs[0])
    # axs[0].set_title(f"True score (higher is better)")
    # axs[0].set_ylabel("True Score")
    # axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="score", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Score")
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="overlap", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"Overlap")
    axs[0].set_ylabel("overlap")
    axs[0].legend(fontsize="x-small")

    # sns.barplot(
    #     x=param_name, y="unique_between_K", hue="algorithm", data=scores, ax=axs[1]
    # )
    # axs[1].set_title(f"Unique between K")
    # axs[1].set_ylabel("% Unique")
    # axs[1].legend([], [], frameon=False)
    # plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="depth", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"Depth")
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="features", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Features")
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(
        x=param_name, y="total_options", hue="algorithm", data=scores, ax=axs[0]
    )
    axs[0].set_title(f"Total options")
    axs[0].legend(fontsize="x-small")

    sns.barplot(
        x=param_name, y="valid_options", hue="algorithm", data=scores, ax=axs[1]
    )
    axs[1].set_title(f"Valid options")
    axs[1].legend([], [], frameon=False)

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(
        x=param_name, y="validate_time", hue="algorithm", data=scores, ax=axs[0]
    )
    axs[0].set_title(f"Validation Time")
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="scan_time", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Scan time")
    axs[1].legend([], [], frameon=False)


def debug(cate_model):
    D = Data()
    D.generate()

    alg = cate_model(D)
    start = time()
    alg.fit()
    end = round(time() - start, 2)
    print(f"[{alg.algorithm}] Offline time: {end}")

    cond, s = D.generate_random_condition()
    print("User condition:", cond, "Selectivity:", s)

    start = time()
    valid_options = alg.online()
    end = round(time() - start, 2)
    print(
        f"[{alg.algorithm}] Online time: {end}, Total Splits: {len(alg.options)}, Valid: {len(valid_options)}"
    )

    start = time()
    top = D.get_topK(options=valid_options)
    end = round(time() - start, 2)
    print(f"[{alg.algorithm}] get topK in {end}")

    return top
