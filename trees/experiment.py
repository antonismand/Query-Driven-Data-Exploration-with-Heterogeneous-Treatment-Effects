import inspect
from itertools import product
import sys
import time
import humanize
import seaborn as sns
import tikzplotlib
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl

from loguru import logger
from trees.data import Data
from trees.topk import *
from trees.causal_tree import CT, CTP
from trees.causal_forest import CFT
from trees.topk import *
from trees.params import DEBUG_LEVEL
from datetime import timedelta

mpl.rcParams.update(mpl.rcParamsDefault)

logger.configure(handlers=[{"sink": sys.stderr, "level": DEBUG_LEVEL}])


class Experiment:
    def __init__(
        self,
        var_name: str,
        exp_name: str = None,
        data_iterations=1,
        user_iterations=20,
        cate_models=[CT, CTP, CFT],
        topk_methods=main_competitors,
        save_csv=True,
        **kwargs,
    ):
        self.scores = pd.DataFrame()
        self.var_name = var_name

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
                        f"\t {ct.algorithm} - Offline time: {timedelta(seconds=(time() - start))}"
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
                                f"\t\t\t {ct.algorithm} - Online time: {timedelta(seconds=(time() - start))}"
                            )

                            for topK_values in product(*topK_params.values()):
                                topK_dict = dict(zip(topK_keys, topK_values))

                                for topk_method in topk_methods:
                                    start = time()
                                    topk = topk_method(alg=ct)
                                    score = topk.get_topK(**topK_dict)
                                    logger.info(
                                        f"\t\t\t\t {topk.name}: {timedelta(seconds=(time() - start))} - {str(topK_dict)}"
                                    )

                                    current_combination = {
                                        **param_dict,
                                        **condition_dict,
                                        **topK_dict,
                                    }
                                    if var_name in current_combination:
                                        score[var_name] = current_combination[var_name]
                                    self.scores = pd.concat(
                                        [self.scores, score], ignore_index=True
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

        if save_csv:
            if exp_name is None:
                exp_name = var_name
            self.scores.to_csv(f"../csv/{exp_name}.csv", index=False)

    def plot(self, x=None, hue="algorithm", rotation=30):
        if x is None:
            x = self.var_name
        plots(x=x, hue=hue, scores=self.scores, rotation=rotation)

    def get_signature(self, func, kwargs):
        signature = inspect.signature(func)
        parameters = {}
        for param_name, param in signature.parameters.items():
            if param_name in kwargs:
                parameters[param_name] = kwargs[param_name]
            elif param.default is not param.empty:
                parameters[param_name] = [param.default]

        return parameters

    def save_tikzplot(self, plots=["t", "overlap"]):
        if "t" in plots:
            plt.figure()
            ax = sns.barplot(
                x=self.var_name, y="t_r", hue="algorithm", data=self.scores
            )
            ax.set_title("True CATE (higher is better)")
            ax.set_ylabel(r"$|\tau(R_i)|$")
            ax.tick_params(axis="x", labelrotation=20)
            ax.set_xlabel(r"top-$K$ Algorithm")

            ax.legend(fontsize="x-small")
            plt.draw()

            # tikzplotlib.clean_figure()
            tikzplotlib.save(
                f"../tex/{self.var_name}-t.tex",
                extra_axis_parameters=[
                    "scaled x ticks=false",
                    "scaled y ticks=false",
                    "yticklabel style={/pgf/number format/precision=3}",
                    "xticklabel style={font=\small}",
                    "legend style={fill opacity=0.8,draw opacity=1, text opacity=1, draw=white!80!black,font=\scriptsize}",
                ],
                axis_height="\\figH",
                axis_width="\\figW",
            )

        if "overlap" in plots:
            plt.figure()
            ax = sns.barplot(
                x=self.var_name, y="overlap", hue="algorithm", data=self.scores
            )
            ax.set_title("Overlap (lower is better)")
            ax.set_ylabel("Overlap")
            ax.tick_params(axis="x", labelrotation=20)
            ax.set_xlabel(r"top-$K$ Algorithm")
            ax.legend(fontsize="x-small")

            plt.draw()

            # tikzplotlib.clean_figure()
            tikzplotlib.save(
                f"../tex/{self.var_name}-overlap.tex",
                extra_axis_parameters=[
                    "scaled x ticks=false",
                    "scaled y ticks=false",
                    "yticklabel style={/pgf/number format/precision=4}",
                    "xticklabel style={font=\small}",
                    "legend style={fill opacity=0.8,draw opacity=1, text opacity=1, draw=white!80!black,font=\scriptsize}",
                ],
                axis_height="\\figH",
                axis_width="\\figW",
            )

        if "time" in plots:
            plt.figure()
            ax = sns.barplot(
                x=self.var_name,
                y="total_execution_time",
                hue="algorithm",
                data=self.scores,
            )
            ax.set_title("Total Execution Time")
            ax.set_ylabel("Time (s)")
            ax.tick_params(axis="x", labelrotation=20)
            ax.set_xlabel(r"top-$K$ Algorithm")
            ax.legend(fontsize="x-small")

            plt.draw()

            # tikzplotlib.clean_figure()
            tikzplotlib.save(
                f"../tex/{self.var_name}-overlap.tex",
                extra_axis_parameters=[
                    "scaled x ticks=false",
                    "scaled y ticks=false",
                    "yticklabel style={/pgf/number format/precision=4}",
                    "xticklabel style={font=\small}",
                    "legend style={fill opacity=0.8,draw opacity=1, text opacity=1, draw=white!80!black,font=\scriptsize}",
                ],
                axis_height="\\figH",
                axis_width="\\figW",
            )


class LoadExperiment(Experiment):
    def __init__(self, var_name):
        self.scores = pd.read_csv(f"../csv/{var_name}.csv")
        self.var_name = var_name
        # plots(x=var_name, scores=self.scores)


def plots(x, hue="algorithm", scores=[], rotation=30):

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=x, y="t_r", hue=hue, data=scores, ax=axs[0])
    axs[0].set_title(f"True CATE (higher is better)")
    axs[0].set_ylabel(r"$\tau(R_i)$")
    axs[0].tick_params(axis="x", labelrotation=rotation)
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=x, y="t_est", hue=hue, data=scores, ax=axs[1])
    axs[1].set_title(f"Estimated CATE")
    axs[1].set_ylabel(r"$\hat{\tau}(S_i)$")
    axs[1].tick_params(axis="x", labelrotation=rotation)
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=x, y="overlap", hue=hue, data=scores, ax=axs[0])
    axs[0].set_title(f"Overlap")
    axs[0].set_ylabel("overlap")
    axs[0].tick_params(axis="x", labelrotation=rotation)
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=x, y="total_execution_time", hue=hue, data=scores, ax=axs[1])
    axs[1].set_title(f"Total Online time")
    axs[1].tick_params(axis="x", labelrotation=rotation)
    axs[1].set_ylabel("Time (s)")
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=x, y="depth", hue=hue, data=scores, ax=axs[0])
    axs[0].set_title(f"Depth")
    axs[0].tick_params(axis="x", labelrotation=rotation)
    axs[0].legend(fontsize="x-small")

    sns.barplot(x=x, y="features", hue=hue, data=scores, ax=axs[1])
    axs[1].set_title(f"Features")
    axs[1].tick_params(axis="x", labelrotation=rotation)
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=x, y="t_error", hue=hue, data=scores, ax=axs[0])
    axs[0].set_title(f"Prediction error")
    axs[0].set_ylabel(r"$|\tau(S_i) - \hat{\tau}(S_i)|$")
    axs[0].tick_params(axis="x", labelrotation=rotation)

    axs[0].legend(fontsize="x-small")

    sns.barplot(x=x, y="t_r_error", hue=hue, data=scores, ax=axs[1])
    axs[1].set_title(f"False Estimation")
    axs[1].set_ylabel(r"$|\tau(R_i) - \tau(S_i)|$")
    axs[1].tick_params(axis="x", labelrotation=rotation)
    axs[1].legend([], [], frameon=False)
    plt.show()

    # ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=x, y="total_subgroups", hue=hue, data=scores, ax=axs[0])
    axs[0].set_title(f"Total subgroups")
    axs[0].tick_params(axis="x", labelrotation=rotation)
    axs[0].legend(fontsize="x-small")

    # sns.barplot(x=x, y="valid_subgroups", hue=hue, data=scores, ax=axs[1])
    # axs[1].set_title(f"Valid subgroups")
    # axs[1].tick_params(axis="x", labelrotation=rotation)
    # axs[1].legend([], [], frameon=False)
    # plt.show()

    # ----------------- #

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # sns.barplot(x=x, y="ct_execution_time", hue=hue, data=scores, ax=axs[0])
    # axs[0].set_title(f"Filtering Time")
    # axs[0].tick_params(axis="x", labelrotation=rotation)
    # axs[0].set_ylabel("Time (s)")
    # axs[0].legend(fontsize="x-small")


def single_run(cate_model=CTP, topk_method=None, seed=0):
    D = Data()
    D.generate(seed=seed)

    alg = cate_model()
    start = time()
    alg.fit(D=D)
    end = round(time() - start, 2)
    logger.info(f"[{alg.algorithm}] Offline time: {end}")

    D.generate_random_condition()

    start = time()
    alg.online()
    end = round(time() - start, 2)
    print(f"[{alg.algorithm}] Online time: {end}")
    start = time()
    topk = topk_method(alg=alg)
    top = topk.get_topK()
    end = round(time() - start, 2)
    print(f"[{alg.algorithm}] {end}s")

    return top
