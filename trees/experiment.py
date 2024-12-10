import inspect
from itertools import product
import sys
import time
import humanize
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns

from loguru import logger
from trees.data import Data
from trees.topk import *
from trees.causal_tree import CT, CTP
from trees.causal_forest import CFT
from trees.topk import *
from trees.params import DEBUG_LEVEL

logger.configure(handlers=[{"sink": sys.stderr, "level": DEBUG_LEVEL}])


class Experiment:
    def __init__(
        self,
        var_name: str,
        data_iterations=1,
        user_iterations=10,
        cate_models=[CT, CTP, CFT],
        topk_methods=all,
        save_csv=False,
        **kwargs,
    ):
        scores = pd.DataFrame()

        data_params = self.get_signature(Data().generate, kwargs)
        data_keys = list(data_params.keys())

        condition_params = self.get_signature(Data().generate_random_condition, kwargs)
        condition_keys = list(condition_params.keys())

        topK_params = self.get_signature(TopK(None).get_topK, kwargs)
        topK_keys = list(topK_params.keys())

        start_total = time()

        for data_values in product(*data_params.values()):
            param_dict = dict(zip(data_keys, data_values))

            for exp in range(data_iterations):
                data = Data()
                param_dict["seed"] = exp
                logger.info("Generating data: {}", param_dict)
                data.generate(**param_dict)

                cts = [ct() for ct in cate_models]
                for ct in cts:
                    start = time()
                    ct.fit(D=data)
                    logger.info(
                        f"\t {ct.algorithm} - Offline time: {humanize.precisedelta(time() - start)}"
                    )

                for j in range(user_iterations):
                    for condition_values in product(*condition_params.values()):
                        condition_dict = dict(zip(condition_keys, condition_values))
                        logger.info(
                            "\t\t Generating P {}/{}: {}",
                            j + 1,
                            user_iterations,
                            condition_dict,
                        )

                        data.generate_random_condition(**condition_dict)

                        for ct in cts:
                            # logger.info(f"{ct.algorithm} running")
                            start = time()
                            ct.online()
                            logger.info(
                                f"\t\t\t {ct.algorithm} - Online time: {humanize.precisedelta(time() - start)}"
                            )

                            for topK_values in product(*topK_params.values()):
                                topK_dict = dict(zip(topK_keys, topK_values))

                                for topk_method in topk_methods:
                                    start = time()
                                    topk = topk_method(alg=ct)
                                    score = topk.get_topK(**topK_dict)
                                    logger.info(
                                        f"\t\t\t\t {topk.name}: {humanize.precisedelta(time() - start)} - {str(topK_dict)}"
                                    )

                                    current_combination = {
                                        **param_dict,
                                        **condition_dict,
                                        **topK_dict,
                                    }
                                    if var_name in current_combination:
                                        score[var_name] = current_combination[var_name]
                                    scores = pd.concat(
                                        [scores, score], ignore_index=True
                                    )

                # clear_output(wait=True)

        logger.success(
            f"Finished experiment in {humanize.precisedelta(time() - start_total)} with {data_iterations} data iterations and {user_iterations} user iterations"
        )
        logger.success("D params: {}", data_params)
        logger.success("P params: {}", condition_params)
        logger.success(
            "topK params: {}",
            {k: v for k, v in vars(params.TOPK).items() if not k.startswith("__")},
        )
        logger.success(
            "CT Params: {}",
            {k: v for k, v in vars(params.CT).items() if not k.startswith("__")},
        )
        logger.success(
            "CF Params: {}",
            {k: v for k, v in vars(params.CF).items() if not k.startswith("__")},
        )

        plots(var_name, scores)
        if save_csv:
            scores.to_csv(f"../csv/scores_{var_name}.csv", index=False)

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
    axs[0].set_ylabel(r"$\tau(S_i)$")
    axs[0].tick_params(axis="x", labelrotation=30)
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="t_est", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Estimated CATE")
    axs[1].set_ylabel(r"$\hat{\tau}(S_i)$")
    axs[1].tick_params(axis="x", labelrotation=30)
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # sns.barplot(x=param_name, y="true_score", hue="algorithm", data=scores, ax=axs[0])
    # axs[0].set_title(f"True score (higher is better)")
    # axs[0].set_ylabel("True Score")
    # axs[0].legend(fontsize="x-small")

    # sns.barplot(x=param_name, y="score", hue="algorithm", data=scores, ax=axs[1])
    # axs[1].set_title(f"Score")
    # axs[1].legend([], [], frameon=False)
    # plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="t_error", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"Prediction error")
    axs[0].set_ylabel(r"$|\tau(S_i) - \hat{\tau}(S_i)|$")
    axs[0].tick_params(axis="x", labelrotation=30)

    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="t_r_error", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"False Estimation")
    axs[1].set_ylabel(r"$|\tau(R_i) - \tau(S_i)|$")
    axs[1].tick_params(axis="x", labelrotation=30)
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="overlap", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"Overlap")
    axs[0].set_ylabel("overlap")
    axs[0].tick_params(axis="x", labelrotation=30)
    axs[0].legend(fontsize="x-small")

    # sns.barplot(x=param_name, y="score", hue="algorithm", data=scores, ax=axs[1])
    # axs[1].set_title(f"Predicted Score")
    # axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=param_name, y="depth", hue="algorithm", data=scores, ax=axs[0])
    axs[0].set_title(f"Depth")
    axs[0].tick_params(axis="x", labelrotation=30)
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=param_name, y="features", hue="algorithm", data=scores, ax=axs[1])
    axs[1].set_title(f"Features")
    axs[1].tick_params(axis="x", labelrotation=30)
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(
        x=param_name, y="total_subgroups", hue="algorithm", data=scores, ax=axs[0]
    )
    axs[0].set_title(f"Total subgroups")
    axs[0].tick_params(axis="x", labelrotation=30)
    axs[0].legend(fontsize="x-small")

    sns.barplot(
        x=param_name, y="valid_subgroups", hue="algorithm", data=scores, ax=axs[1]
    )
    axs[1].set_title(f"Valid subgroups")
    axs[1].tick_params(axis="x", labelrotation=30)
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # sns.barplot(
    #     x=param_name, y="pruned_min_rows", hue="algorithm", data=scores, ax=axs[0]
    # )
    # axs[0].set_title(f"Pruned due to min rows")
    # axs[0].legend(fontsize="x-small")

    # sns.barplot(
    #     x=param_name, y="pruned_not_subsets", hue="algorithm", data=scores, ax=axs[1]
    # )
    # axs[1].set_title(f"Pruned due to not being subsets")
    # axs[1].legend([], [], frameon=False)
    # plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(
        x=param_name, y="online_execution_time", hue="algorithm", data=scores, ax=axs[0]
    )
    axs[0].set_title(f"Filtering Time")
    axs[0].tick_params(axis="x", labelrotation=30)
    axs[0].set_ylabel("Time (s)")
    axs[0].legend(fontsize="x-small")

    sns.barplot(
        x=param_name, y="topK_execution_time", hue="algorithm", data=scores, ax=axs[1]
    )
    axs[1].set_title(f"TopK time")
    axs[1].tick_params(axis="x", labelrotation=30)
    axs[1].set_ylabel("Time (s)")
    axs[1].legend([], [], frameon=False)
    plt.show()


# def single_run(cate_model=CT, scanner=TopK, seed=42):
#     D = Data()
#     D.generate()

#     alg = cate_model(scan_method=scanner)
#     start = time()
#     alg.fit(D=D, seed=seed)
#     end = round(time() - start, 2)
#     print(f"[{alg.algorithm}] Offline time: {end}")

#     D.generate_random_condition()

#     start = time()
#     alg.online()
#     end = round(time() - start, 2)
#     print(f"[{alg.algorithm}] Online time: {end}")
#     start = time()
#     scan = scanner(valid_subs=alg.valid_options, op_matrix=alg.op_matrix, D=D)
#     top = scan.get_topK()
#     end = round(time() - start, 2)
#     print(f"[{scan.name}] get topK score: {round(top['score'].mean(),2)} in {end}s")

#     return top
