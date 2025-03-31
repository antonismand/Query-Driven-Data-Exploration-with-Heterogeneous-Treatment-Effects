from matplotlib import pyplot as plt

# Generic ML imports
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures

# EconML imports
from econml.iv.dr import LinearIntentToTreatDRIV
from econml.cate_interpreter import SingleTreeCateInterpreter

from trees.causal_forest import CF
from trees.data import Data
from trees import params


class LinearIV(CF):
    def __init__(self):
        self.subgroups = {}
        self.algorithm = "[Pretrained] LinearIV"
        self.is_online = False

    def fit(
        self,
        D: Data,
        max_depth=params.LINEARIV.MAX_DEPTH,
        min_samples_leaf=params.LINEARIV.MIN_SAMPLES_LEAF,
        train_in_all_features=params.LINEARIV.TRAIN_IN_ALL_FEATURES,
        print_tree=False,
    ):
        self.df = D.df
        self.D = D

        self.features = (
            self.D.feature_names if train_in_all_features else self.D.hte_features
        )

        # Define nuissance estimators
        lgb_T_XZ_params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.1,
            "num_leaves": 30,
            "max_depth": 5,
        }

        lgb_Y_X_params = {
            "metric": "rmse",
            "learning_rate": 0.1,
            "num_leaves": 30,
            "max_depth": 5,
        }
        model_T_XZ = lgb.LGBMClassifier(**lgb_T_XZ_params)
        model_Y_X = lgb.LGBMRegressor(**lgb_Y_X_params)
        flexible_model_effect = lgb.LGBMRegressor(**lgb_Y_X_params)

        model = LinearIntentToTreatDRIV(
            model_y_xw=model_Y_X,
            model_t_xwz=model_T_XZ,
            flexible_model_effect=flexible_model_effect,
            featurizer=PolynomialFeatures(degree=1, include_bias=False),
        )

        model.fit(
            Y=self.D.Y,
            T=self.D.T,
            Z=self.D.Z,
            X=self.D.X[self.features],
            inference="statsmodels",
        )

        print(model.summary())

        intrp = SingleTreeCateInterpreter(
            include_model_uncertainty=True,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        intrp.interpret(model, self.D.X[self.features].to_numpy())

        if print_tree:
            plt.figure(figsize=(25, 5))
            intrp.plot(feature_names=self.features, fontsize=12)

        final_tree = intrp.tree_model_.tree_
        self.parse_tree(final_tree)

    def online(self):
        self.online_time = 0
