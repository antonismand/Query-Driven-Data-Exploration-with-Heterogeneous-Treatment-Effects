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
from trees.linear import LinearIV
from trees.topk import *
from trees.params import DEBUG_LEVEL
from datetime import timedelta

mpl.rcParams.update(mpl.rcParamsDefault)

logger.configure(handlers=[{"sink": sys.stderr, "level": DEBUG_LEVEL}])

label_map = {
    "t_r": "Responsiveness",
    # "t": r"$|\tau(S_i)|$",
    "Top-K Algorithm": r"Top-$K$ Algorithm",
    "Time": "Time (s)",
    "t_est": r"$|\hat{\tau}(S_i)|$",
    "w": r"$w$",
    "min_diversity": r"$\theta_D$",
    "n": "Dataset Size",
    "max_depth": "Max Depth",
}


class Experiment:
    def __init__(
        self,
        var_name: str,
        exp_name: str = None,
        data_iterations=1,
        user_iterations=30,
        cate_models=[CFT],
        topk_methods=main_competitors,
        save_csv=True,
        **kwargs,
    ):
        self.scores = pd.DataFrame()
        self.var_name = var_name
        self.exp_name = exp_name

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

                if "mode" in param_dict and param_dict["mode"] == 6:
                    cate_models = [LinearIV]

                cts = [ct() for ct in cate_models]
                for ct in cts:
                    start = time()
                    ct.fit(D=data)
                    logger.info(
                        f"\t {ct.algorithm} - Offline time: {timedelta(seconds=(time() - start))}"
                    )

                for j in tqdm(range(user_iterations)):
                    for condition_values in product(*condition_params.values()):
                        condition_dict = dict(zip(condition_keys, condition_values))
                        logger.debug(
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
                            logger.debug(
                                f"\t\t\t {ct.algorithm} - Online time: {timedelta(seconds=(time() - start))}"
                            )

                            for topK_values in product(*topK_params.values()):
                                topK_dict = dict(zip(topK_keys, topK_values))

                                for topk_method in topk_methods:
                                    start = time()
                                    topk = topk_method(alg=ct)
                                    score = topk.get_topK(**topK_dict)
                                    logger.debug(
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

    def get_signature(self, func, kwargs):
        signature = inspect.signature(func)
        parameters = {}
        for param_name, param in signature.parameters.items():
            if param_name in kwargs:
                parameters[param_name] = kwargs[param_name]
            elif param.default is not param.empty:
                parameters[param_name] = [param.default]

        return parameters

    def save_tikzplot(self, x, y, hue="Top-K Algorithm", rotation=0):
        if x is None:
            x = self.var_name
        if self.exp_name is None:
            self.exp_name = self.var_name

        if x == "Top-K Algorithm":
            ax = sns.barplot(x=x, y=y, hue=hue, data=self.scores)
        else:
            ax = sns.lineplot(x=x, y=y, hue=hue, data=self.scores, errorbar=None)
            if x == "n" and y == "Time":
                # ax.set_yscale("log")
                ax.set_xticks([100000, 500000, 1000000])
                ax.set_xticklabels(["100K", "500K", "1M"])

            for line, label in zip(ax.get_lines(), self.scores[hue].unique()):
                line.set_label(label)

        ax.tick_params(axis="x", labelrotation=rotation)
        ax.set_xlabel(label_map.get(x, x))
        ax.set_ylabel(label_map.get(y, y))
        ax.legend(fontsize="x-small")
        plt.draw()

        # tikzplotlib.clean_figure()
        tikzplotlib.save(
            f"../tex/{self.exp_name}-{y}.tex",
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
        plt.show()

    def two_plots(self, x, y1, y2, hue, rotation):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        if x == "Top-K Algorithm":
            sns.barplot(x=x, y=y1, hue=hue, data=self.scores, ax=axs[0])
        else:
            sns.lineplot(x=x, y=y1, hue=hue, data=self.scores, ax=axs[0])

        axs[0].set_xlabel(label_map.get(x, x))
        axs[0].set_ylabel(label_map.get(y1, y1))
        axs[0].tick_params(axis="x", labelrotation=rotation)
        axs[0].legend(fontsize="x-small")

        if x == "Top-K Algorithm":
            sns.barplot(x=x, y=y2, hue=hue, data=self.scores, ax=axs[1])
        else:
            sns.lineplot(x=x, y=y2, hue=hue, data=self.scores, ax=axs[1])

        axs[1].tick_params(axis="x", labelrotation=rotation)
        axs[1].set_xlabel(label_map.get(x, x))
        axs[1].set_ylabel(label_map.get(y2, y2))
        axs[1].legend([], [], frameon=False)
        plt.show()

    def plots(self, x=None, hue="Top-K Algorithm", rotation=0):

        if x is None:
            x = self.var_name

        self.two_plots(x, y1="t_r", y2="Diversity", hue=hue, rotation=rotation)

        self.two_plots(x, y1="t_est", y2="Max Overlap", hue=hue, rotation=rotation)

        self.two_plots(x, y1="Total Subgroups", y2="Time", hue=hue, rotation=rotation)

        self.two_plots(x, y1="Depth", y2="Features", hue=hue, rotation=rotation)

        self.two_plots(x, y1="t_error", y2="t_r_error", hue=hue, rotation=rotation)


class LoadExperiment(Experiment):
    def __init__(self, exp_name):
        self.scores = pd.read_csv(f"../csv/{exp_name}.csv")
        self.exp_name = exp_name


class CFTuning(Experiment):
    def __init__(
        self,
        var_name,
        var_values,
        n_rows=params.DATA.N_ROWS,
        user_iterations=30,
        topk_methods=main_no_random,
        save_csv=True,
    ):

        self.exp_name = "tune_" + var_name
        self.var_name = var_name
        self.scores = pd.DataFrame()

        D = Data()
        D.generate(seed=34, n=n_rows)

        for p in tqdm(var_values):
            cf = CFT()
            cf.fit(D=D, **{var_name: p})

            for _ in range(user_iterations):
                D.generate_random_condition()
                cf.online()
                for topk_method in topk_methods:
                    topk = topk_method(alg=cf)
                    score = topk.get_topK()
                    score[var_name] = p
                    self.scores = pd.concat([self.scores, score], ignore_index=True)

        if save_csv:
            self.scores.to_csv(f"../csv/{self.exp_name}.csv", index=False)


class TopKExperiment(Experiment):
    def __init__(
        self,
        var_name,
        var_values,
        n_rows=params.DATA.N_ROWS,
        user_iterations=30,
        topk_methods=main_no_random,
        save_csv=False,
    ):

        self.exp_name = "topk_param_" + var_name
        self.var_name = var_name
        self.scores = pd.DataFrame()

        D = Data()
        D.generate(seed=34, n=n_rows)

        for p in tqdm(var_values):
            cf = CFT()
            cf.fit(D=D)

            for _ in range(user_iterations):
                D.generate_random_condition()
                cf.online()
                for topk_method in topk_methods:
                    topk = topk_method(alg=cf, **{var_name: p})
                    score = topk.get_topK()
                    score[var_name] = p
                    self.scores = pd.concat([self.scores, score], ignore_index=True)
        if save_csv:
            self.scores.to_csv(f"../csv/{self.exp_name}.csv", index=False)


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
