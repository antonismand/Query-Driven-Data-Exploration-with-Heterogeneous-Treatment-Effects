class DATA:
    N_ROWS = 100000
    N_FEATURES = 10
    SIGMA = 3.0
    SEED = 42
    MODE = 2


class P:
    MIN_S = 0.3
    MAX_S = 0.95
    FEATURES_IN_P = "all"


class CT:
    MAX_DEPTH = 15
    MIN_SAMPLES_LEAF = 20
    CRITERION = "causal_mse"
    TRAIN_IN_ALL_FEATURES = False


class CF:
    CRITERION = "het"
    N_ESTIMATORS = 64
    MAX_DEPTH = 12
    CV = 2
    MIN_SAMPLES_SPLIT = 30
    MIN_SAMPLES_LEAF = 20
    MAX_FEATURES = "auto"
    TRAIN_IN_ALL_FEATURES = False


class TOPK:
    W = 0.7
    K = 5
    MIN_DIVERSITY = 0.2
    PERCENTILE = 0.7
    N_MIN_ROWS = 20
    EXHAUSTIVE_TOPK = 100


# DEBUG_LEVEL = "TRACE"
# DEBUG_LEVEL = "DEBUG"
DEBUG_LEVEL = "INFO"
# DEBUG_LEVEL = "SUCCESS"
