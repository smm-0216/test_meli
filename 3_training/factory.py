from typing import Dict, Type

from base_model_optimizer import Optimizer
from base_models import XGBoost, LightGBM, CatBoost
from superlearner import SuperLearner


class TrainingFactory:
    """
    Factory class for creating model trainer instances.

    This factory provides a centralized way to create instances of different
    model optimizers without directly coupling the code to
    specific implementations.
    """

    _trainers: Dict[str, Type[Optimizer]] = {
        "xgboost": XGBoost,
        "lightgbm": LightGBM,
        "catboost": CatBoost,
        "superlearner": SuperLearner
    }

    @classmethod
    def create_trainer(cls, model_name: str, **kwargs) -> Optimizer:
        """
        Create and return an instance of the requested model trainer.

        Args:
            model_name: Name of the model trainer to create
            **kwargs: Additional arguments to pass to the trainer constructor

        Returns:
            An instance of a Optimizer subclass

        Raises:
            ValueError: If the requested model name is not registered
        """
        if model_name not in cls._trainers:
            available = list(cls._trainers.keys())
            raise ValueError(
                f"Model '{model_name}' not supported. Available: {available}")
        model_specific_params = {
            "xgboost": {"model_name": "xgboost"},
            "lightgbm": {"model_name": "lightgbm"},
            "catboost": {"model_name": "catboost"},
            "superlearner": {
                "base_models": kwargs.get("base_models"),
                "best_score_base_models": kwargs.get("best_score_base_models"),
                "type_ensemble": kwargs.get("type_ensemble")
            },
        }
        params = {**model_specific_params[model_name]}
        return cls._trainers[model_name](**params)
