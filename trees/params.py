class DATA:
    N_ROWS = 100000
    N_FEATURES = 10
    SIGMA = 3.0
    SEED = 0
    MODE = "RCT"
    AB_DATA = "../datasets/ab_sample.csv"
    UPLIFT_DATA = "../datasets/criteo-uplift-v2.1.csv"


class P:
    MIN_S = 0.3
    MAX_S = 0.95
    FEATURES_IN_P = "all"
    # FEATURES_IN_P = ["feature_0"]


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
    USE_PRETRAINED = True


class LINEARIV:
    MAX_DEPTH = 12
    MIN_SAMPLES_SPLIT = 30
    MIN_SAMPLES_LEAF = 20
    TRAIN_IN_ALL_FEATURES = False


class TOPK:
    W = 0.5
    K = 5
    MIN_DIVERSITY = 0.2
    PERCENTILE = 0.7
    N_MIN_ROWS = 20
    EXHAUSTIVE_TOPK = 500


# DEBUG_LEVEL = "TRACE"
# DEBUG_LEVEL = "DEBUG"
DEBUG_LEVEL = "INFO"
# DEBUG_LEVEL = "SUCCESS"
