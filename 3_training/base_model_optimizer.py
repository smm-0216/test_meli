import warnings
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    message=r'^Failed to optimize method "evaluate".*'
)

from abc import ABC, abstractmethod

import optuna
import mlflow
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, average_precision_score

from evaluation import metric


class Optimizer(ABC):
    """Base class for training different ML models with Optuna optimization"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.train_features_key = "train_features"
        self.train_target_key = "train_target"

    @abstractmethod
    def _get_hyperparameter_space(self, trial: optuna.Trial) -> dict:
        """Define hyperparameter search space for each model"""
        pass

    @abstractmethod
    def _create_model(self, params: dict) -> object:
        """Create model instance with given parameters"""
        pass

    def _perform_cross_validation(self, model: object, data: dict) -> tuple:
        """Perform stratified cross-validation and return scores and early stopping rounds.
        Args:
            model (object): The machine learning model to train and evaluate.
            data (dict): Dictionary containing training features and targets.
        Returns:
            tuple: A tuple containing (scores, early_stopping_rounds) where:
                - scores: List of metric scores from each fold
                - early_stopping_rounds: List of actual rounds used in early stopping (if applicable)
        """
        skf = StratifiedKFold(10, shuffle=True, random_state=42)
        scores = []
        f1_scores = []
        # recall_scores = []
        # avg_scores = []
        for train_idx, val_idx in skf.split(
            data[self.train_features_key],
            data[self.train_target_key]
        ):
            X_train = data[self.train_features_key].iloc[train_idx]
            X_val = data[self.train_features_key].iloc[val_idx]
            y_train = data[self.train_target_key].iloc[train_idx]
            y_val = data[self.train_target_key].iloc[val_idx]
            monto_val = data[self.train_features_key]["Monto"].iloc[val_idx]
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X_val)
            # probs = fold_model.predict_proba(X_val)[:, 1]
            score = metric(y_val, preds, monto_val)
            scores.append(score)
            # avg = average_precision_score(y_val, probs)
            # avg_scores.append(avg)
            # recall = recall_score(y_val, preds)
            f1 = f1_score(y_val, preds)
            f1_scores.append(f1)
            # recall_scores.append(recall)
        return np.mean(scores), np.mean(f1_scores)
        # return np.mean(scores)

    def __objective(self, trial: optuna.Trial, data: dict, parent_run_id: str) -> float:
        """Objective function for Optuna optimization.
        This function is called for each trial and performs the following:
        1. Starts a new MLflow run for the trial.
        2. Logs parameters and tags.
        3. Creates a model with the suggested hyperparameters.
        4. Performs time series cross-validation.
        5. Logs the ultra metric score.
        6. Returns the mean score across all folds.
        Args:
            trial (optuna.Trial): The trial object containing hyperparameters.
            data (dict): Dictionary containing training and validation data.
            parent_run_id (str): ID of the parent MLflow run.
        Returns:
            float: The mean ultra metric score for the trial.
        """
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True, parent_run_id=parent_run_id) as _:
            mlflow.set_tag("model_name", self.model_name)
            mlflow.set_tag("trial_number", trial.number)
            params = self._get_hyperparameter_space(trial)
            print(f"Starting trial {trial.number} for {self.model_name} with parameters: {params}")
            for key, value in params.items():
                mlflow.log_param(key, value)
            model = self._create_model(params)
            mean_scores, f1_scores = self._perform_cross_validation(model, data)
            # mean_scores = self._perform_cross_validation(model, data)
            mlflow.log_metric("metric", mean_scores)
            print(f"Trial {trial.number} completed with score: {mean_scores}")
            return mean_scores, f1_scores
            # return mean_scores

    def __pick_by_knee(self, pareto_trials):
        vals = np.array([t.values for t in pareto_trials], dtype=float)
        vmin = vals.min(axis=0)
        vmax = vals.max(axis=0)
        denom = np.where(vmax > vmin, vmax - vmin, 1.0)
        norm = (vals - vmin) / denom
        utopia = np.ones(norm.shape[1])
        dists = np.linalg.norm(norm - utopia, axis=1)
        idx = int(dists.argmin())
        return pareto_trials[idx]

    def __optimize_model(self, data: dict, parent_run_id: str, n_trials: int = 10, n_jobs: int = 2) -> tuple:
        """Optimize model parameters using Optuna.
        This function creates a study, optimizes the model parameters, and returns the best model and study object.
        Args:
            data (dict): Dictionary containing training and validation data.
            parent_run_id (str): ID of the parent MLflow run.
            n_trials (int): Number of trials for hyperparameter optimization.
            n_jobs (int): Number of parallel jobs to run.
        Returns:
            tuple: (best_model, best_params, best_value) - The best model, its parameters, and the best score.
        """
        connection_string = "sqlite:///meli.db"
        study = optuna.create_study(directions=["maximize", "maximize"], storage=connection_string, study_name=parent_run_id, load_if_exists=True)
        # study = optuna.create_study(direction="maximize", storage=connection_string, study_name=parent_run_id, load_if_exists=True)
        print(f"Starting hyperparameter optimization for {self.model_name} with {n_trials} trials")
        study.optimize(lambda trial: self.__objective(trial, data, parent_run_id), n_trials=n_trials, n_jobs=n_jobs)
        best_trial = self.__pick_by_knee(study.best_trials)
        final_params = best_trial.params
        # best_trial = study.best_trial
        # final_params = best_trial.params
        print(f"Optimization completed for {self.model_name} with parameters: {final_params}")
        best_model = self._create_model(final_params)
        best_model.fit(data["train_features"], data["train_target"])
        best_value = best_trial.values[0]
        # best_value = study.best_value
        return best_model, final_params, best_value

    def fit(self, data: dict, parent_run_id: str = None) -> dict:
        """Fit the model using the provided data.
        This function performs the following steps:
        1. Initializes the number of rows based on the training data.
        2. Optimizes the model parameters using Optuna.
        3. Fits the model with the best parameters on the training data.
        4. Evaluates the model on the test data and returns the results.
        Args:
            data (dict): Dictionary containing training and validation data.
            parent_run_id (str): ID of the parent MLflow run for logging.
        Returns:
            dict: Dictionary containing the fitted model, its name, best parameters, and final scores.
        """

        model, best_params, best_value = self.__optimize_model(data, parent_run_id=parent_run_id)

        y_train_pred = model.predict(data[self.train_features_key])
        y_test_pred = model.predict(data["test_features"])
        train_monto = data['train_monto']
        test_monto = data['test_monto']

        return {
            "model_name": self.model_name,
            "model": model,
            "best_params": best_params,
            "cv_score": best_value,
            "train_score": metric(data[self.train_target_key], y_train_pred, train_monto),
            "test_score": metric(data["test_target"], y_test_pred, test_monto),
            "train_f1_score": f1_score(data[self.train_target_key], y_train_pred),
            "train_precision": precision_score(data[self.train_target_key], y_train_pred),
            "test_f1_score": f1_score(data["test_target"], y_test_pred),
            "test_precision": precision_score(data["test_target"], y_test_pred),
            "train_recall": recall_score(data[self.train_target_key], y_train_pred),
            "test_recall": recall_score(data["test_target"], y_test_pred),
            "train_accuracy": accuracy_score(data[self.train_target_key], y_train_pred),
            "test_accuracy": accuracy_score(data["test_target"], y_test_pred)
        }
