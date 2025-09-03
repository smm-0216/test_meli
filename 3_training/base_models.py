import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from base_model_optimizer import Optimizer


class XGBoost(Optimizer):
    """Optimizer for XGBoost model using Optuna for hyperparameter tuning."""

    def __init__(self, model_name):
        super().__init__(model_name)

    def _get_hyperparameter_space(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2, 10),
        }

    def _create_model(self, params: dict) -> xgb.XGBClassifier:
        model_params = params.copy()
        return xgb.XGBClassifier(
            **model_params,
            tree_method="hist",
            random_state=42,
        )


class LightGBM(Optimizer):
    """Optimizer for LightGBM model using Optuna for hyperparameter tuning."""

    def __init__(self, model_name):
        super().__init__(model_name)

    def _get_hyperparameter_space(self, trial: optuna.Trial) -> dict:

        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.01),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
        }

    def _create_model(self, params: dict) -> lgb.LGBMClassifier:
        model_params = params.copy()
        return lgb.LGBMClassifier(
            **model_params,
            verbose=-1,
            random_state=42,
            is_unbalance=True,
        )


class CatBoost(Optimizer):
    """Optimizer for CatBoost model using Optuna for hyperparameter tuning."""

    def __init__(self, model_name):
        super().__init__(model_name)

    def _get_hyperparameter_space(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 3, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 0.9),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
            "class_weights": trial.suggest_float("class_weights", 2, 10),
        }

    def _create_model(self, params: dict) -> cb.CatBoostClassifier:
        model_params = params.copy()
        class_weight = model_params.pop("class_weights")
        return cb.CatBoostClassifier(
            **model_params,
            verbose=False,
            allow_writing_files=False,
            random_state=42,
            class_weights=[1, class_weight],
        )
