# Query-Driven Data Exploration with Heterogeneous Treatment Effects

Understanding how changes in actions or policies impact different population segments (users, items, etc.) is central to data-driven decision-making, but it is time-consuming and labor-intensive. We introduce a novel framework that combines user-defined search areas with causal inference to automatically identify responsive subgroups, i.e., subgroups most likely to benefit from a specific action or intervention, by estimating the impact of interventions, while also promoting subgroup diversity to ensure broad and representative insights. To support this, we develop a formal problem definition and an efficient search strategy to navigate the large combinatorial space of possible subgroups. We propose efficient algorithms that use pretrained causal effect estimators to dynamically surface the top-$K$ subgroups. Experiments across multiple datasets and estimation techniques demonstrate the scalability and effectiveness of our framework for large-scale analysis.

## Experiments

### [Evaluation on Causal Forests and Causal Trees](./experiments/estimators.ipynb)

### [Top-K Algorithms in RCT dataset](./experiments/topk_algorithms.ipynb)

Evaluate the performance of each top-K algorithm on the RCT dataset.

### [Evaluation on different datasets](./experiments/datasets.ipynb)

Demonstrate the effectiveness of our framework across multiple datasets.

### [Comparison to Exhaustive Search](./experiments/exhaustive.ipynb)

Evaluate the relative performance of each top-K algorithm against the exhaustive search.

### [Varying Top-K](./experiments/k.ipynb)

Varying the number of subgroup recommendations K, and evaluating the performance of each top-K algorithm.

### [Feature Importance and Inclusion](./experiments/missing_HTE_Feature.ipynb)

Examine the framework's behavior on the UTC dataset, when the CATE model is trained on:

- all 5 HTE features (default)
- 4 HTE features (missing one HTE feature)
- all 10 features (HTE + non-HTE)

### [Scalability](./experiments/scalability_30M.ipynb)

Evaluate the scalability of each top-K algorithm on larger datasets (1M-30M).

### [Parameter Tuning of Top-K Algorithms](./experiments/topk_params.ipynb)

Evaluate the performance of top-K algorithms under different parameter settings.
