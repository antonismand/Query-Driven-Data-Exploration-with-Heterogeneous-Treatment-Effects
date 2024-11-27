class DATA:
    N_MIN_ROWS = 30
    N_ROWS = 10000
    N_FEATURES = 10
    SIGMA = 3.0
    SEED = 42
    MODE = 2


class CT:
    MAX_DEPTH = None
    MIN_SAMPLES_LEAF = 30
    CRITERION = "causal_mse"


class CF:
    CRITERION = "mse"
    N_ESTIMATORS = 64
    MAX_DEPTH = None
    CV = 2
    MIN_SAMPLES_SPLIT = 50
    MIN_SAMPLES_LEAF = 30
    MAX_FEATURES = "auto"


class TOPK:
    W = 0.5
    K = 5
    MAX_PAIRWISE_OVERLAP = 0.5
    PERCENTILE = 0.7
