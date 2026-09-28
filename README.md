<p align="center">
<a href="https://doi.org/10.1109/ICDE65706.2026.00062"><img alt="Paper: ICDE 2026" src="https://img.shields.io/badge/paper-ICDE%202026-informational"></a>
<a href="#"><img alt="Python version" src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python"></a>
</p>

# Query-Driven Data Exploration with Heterogeneous Treatment Effects

Code and experiments for the ICDE 2026 paper
[**Query-Driven Data Exploration with Heterogeneous Treatment Effects**](https://doi.org/10.1109/ICDE65706.2026.00062)
by Antonis Mandamadiotis, Sihem Amer-Yahia and Georgia Koutrika.

## Abstract

Understanding how changes in actions or policies impact different population segments (users, items, etc.) is central to data-driven decision-making, but it is time-consuming and labor-intensive. We introduce a novel framework that combines user-defined search areas with causal inference to automatically identify responsive subgroups, i.e., subgroups most likely to benefit from a specific action or intervention, by estimating the impact of interventions, while also promoting subgroup diversity to ensure broad and representative insights. To support this, we develop a formal problem definition and an efficient search strategy to navigate the large combinatorial space of possible subgroups. We propose efficient algorithms that use pretrained causal effect estimators to dynamically surface the top-$K$ subgroups. Experiments across multiple datasets and estimation techniques demonstrate the scalability and effectiveness of our framework for large-scale analysis.

## Overview

An analyst defines a search area $P$ with a query (e.g. `WHERE age > 40`), and the system returns the $K$ subgroups inside it that respond most to a treatment (highest CATE), are diverse, and each have at least $\theta_n$ rows.

```
 offline:  dataset ──► CATE estimator ──► single tree of candidate subgroups
 online:   query P ──► keep subgroups valid for P ──► top-K selection
```

The CATE estimator (e.g. a Causal Forest) is trained once, offline. At query time, the subgroups in its tree are matched to $P$ and one of four top-$K$ algorithms balances responsiveness against diversity:

- **DiRe:** weighted sum of responsiveness and diversity.
- **ReCoD:** maximizes diversity among the most responsive subgroups.
- **DiCoR:** maximizes responsiveness with a minimum diversity.
- **LoRe:** picks the most responsive subgroups from a single tree level.

## Key results

Evaluated on synthetic (RCT, UTC), semi-synthetic (OTC) and real-world ([Criteo Uplift](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)) datasets:

- Pretrained Causal Forests find more responsive subgroups than Causal Trees, and training them once offline avoids retraining per query (about 25 minutes for 1M rows).
- Compared to exhaustive search, DiCoR reaches 94% of the optimal responsiveness at the same diversity.
- LoRe and DiCoR stop early and stay fast on up to 30M rows. DiRe and ReCoD scan all (or most) subgroups and slow down on large data.
- The results hold across datasets and CATE estimators (including Linear-ITT-IV on OTC).

## Repository structure

| Path                                                 | Contents                                                                                                            |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [trees/data.py](trees/data.py)                       | `Data`: generates or loads the datasets, runs queries (Polars SQL) and computes subgroup dissimilarity              |
| [trees/causal_forest.py](trees/causal_forest.py)     | `CFT`: EconML Causal Forest merged into a single tree with `SingleTreeCateInterpreter` (`CF` uses the forest's trees directly) |
| [trees/causal_tree.py](trees/causal_tree.py)         | `CT` (pretrained) and `CTP` (retrained per query) CausalML Causal Trees                                             |
| [trees/linear.py](trees/linear.py)                   | `LinearIV`: Linear-ITT-IV estimator, used for the OTC dataset                                                       |
| [trees/topk.py](trees/topk.py)                       | Top-K algorithms `LoRe`, `DiRe`, `DiCoR`, `ReCoD`, the `Random` baseline and the exhaustive-search baselines       |
| [trees/parser.py](trees/parser.py)                   | Parses the query's `WHERE` predicates and checks which subgroups fall inside them                                   |
| [trees/params.py](trees/params.py)                   | Default parameters for data generation, estimators and top-K algorithms                                             |
| [trees/experiment.py](trees/experiment.py)           | Experiment harness: runs estimator × top-K combinations over random queries and plots/exports the results           |
| [experiments/](experiments/)                         | Notebooks for the experiments in the paper                                                                          |

## Installation

The project is managed with [Poetry](https://python-poetry.org/) and requires Python 3.11 or later.

```bash
poetry install
```

This installs the dependencies (including Jupyter) and the `trees` package, so run the notebooks with this environment as their Jupyter kernel.

The code writes to three folders at the repository root that are not in the repository, so create them first:

```bash
mkdir models csv tex    # cached models, experiment results, exported plots
```

## Data

The synthetic datasets (RCT, UTC) are generated on the fly with CausalML's `synthetic_data`, so they need no download. The other two are read from a `datasets/` folder at the repository root, which is gitignored:

| Dataset | File                                | Source                                                                                                                                                                                                                  |
| ------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OTC     | `datasets/ab_sample.csv`            | EconML's [online travel company case study](https://github.com/py-why/EconML/blob/main/notebooks/CustomerScenarios/Case%20Study%20-%20Recommendation%20AB%20Testing%20at%20An%20Online%20Travel%20Company.ipynb) |
| CU      | `datasets/criteo-uplift-v2.1.csv`   | [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)                                                                                                                          |

Criteo distributes the file gzipped (`criteo-uplift-v2.1.csv.gz`), so decompress it first. These paths are set in [trees/params.py](trees/params.py) relative to `experiments/`, so run the notebooks from that folder.

## Usage

A minimal example on the synthetic RCT dataset:

```python
from trees.causal_forest import CFT
from trees.topk import DiRe
from trees.data import Data

# Offline phase
D = Data()
D.generate(n=10000, mode="RCT")    # modes: RCT, UTC, OTC, CU-visit, CU-conversion

CATE_estimator = CFT()              # also CT, CTP, or LinearIV for OTC
CATE_estimator.fit(D=D)

# Online phase
D.user_condition("feature_0 > 1")  # the user's search area P
CATE_estimator.online()

topk = DiRe(alg=CATE_estimator)    # top-K algorithms: LoRe, DiRe, DiCoR, ReCoD
subgroups = topk.get_topK(k=5)
print(subgroups["combined"])        # the subgroups' ranges per feature
print(subgroups["final_condition"]) # the full condition: P AND subgroup
```

`CFT` saves the fitted forest to `models/CFT_{mode}.pkl` and loads it on the next run, and the file name depends only on the dataset mode. Delete the file, or set `CF.USE_PRETRAINED = False` in [trees/params.py](trees/params.py), to retrain after you change the data.

## Experiments

Each notebook in [experiments/](experiments/) runs the evaluation over random user queries, saves the scores to `csv/` and exports the paper's plots to `tex/`.

### Main results

- [Causal Trees vs. Causal Forests](experiments/estimators.ipynb): responsiveness of every top-K algorithm with pretrained CT, online CT and pretrained CF on RCT.
- [Top-K algorithms on RCT](experiments/topk_algorithms.ipynb): responsiveness and diversity of each top-K algorithm.
- [Comparison to exhaustive search](experiments/exhaustive.ipynb): how close each algorithm gets to the exhaustive optimum ($K = 3$), averaged over RCT, UTC, OTC and CU.
- [Other datasets](experiments/datasets.ipynb): UTC, OTC, CU-visit and CU-conversion.

### Scalability

- [Dataset size, 100K–1M](experiments/scalability.ipynb) and [1M–30M](experiments/scalability_30M.ipynb): execution time of each top-K algorithm.
- [Maximum tree depth](experiments/max_depth.ipynb): execution time as the number of candidate subgroups grows.

### Parameters and robustness

- [Varying K](experiments/k.ipynb): responsiveness, diversity and time for $K \in \{3, 5, 7, 10\}$.
- [Top-K parameters](experiments/topk_params.ipynb): the weight $w$ in DiRe and the diversity threshold $\theta_D$ in DiCoR.
- [Feature importance](experiments/missing_HTE_feature.ipynb): UTC with the CATE model trained on all 5 HTE features (default), on 4 of them (one missing), or on all 10 features (HTE and non-HTE).
- [Additional tuning](experiments/other/): Causal Forest parameters, data generation parameters (noise, number of features) and which features the random queries use.

## Citation

If you use this code, please cite our paper:

> A. Mandamadiotis, S. Amer-Yahia and G. Koutrika, "Query-Driven Data Exploration with Heterogeneous Treatment Effects," _2026 IEEE 42nd International Conference on Data Engineering (ICDE)_, 2026, pp. 753-765, doi: [10.1109/ICDE65706.2026.00062](https://doi.org/10.1109/ICDE65706.2026.00062).

```bibtex
@INPROCEEDINGS{11629194,
  author={Mandamadiotis, Antonis and Amer-Yahia, Sihem and Koutrika, Georgia},
  booktitle={2026 IEEE 42nd International Conference on Data Engineering (ICDE)},
  title={Query-Driven Data Exploration with Heterogeneous Treatment Effects},
  year={2026},
  volume={},
  number={},
  pages={753-765},
  keywords={subgroup identification;causal inference;recommendations},
  doi={10.1109/ICDE65706.2026.00062},
  ISSN={2375-026X},
  month={May},}
```
