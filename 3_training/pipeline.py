
import os
import joblib
import datetime

import mlflow

from factory import TrainingFactory
from data_handler import DataHandler
from models_registry import models_registry
from train_test_plots import train_test_plots


def log_model_as_artifact(model: object, model_name: str, delete_after=True) -> None:
    """Logs the trained model as an artifact in MLflow.
    Args:
        model (object): The trained model to log.
        model_name (str): The name of the model to be used in the artifact.
        delete_after (bool): Whether to delete the local file after logging.
    """
    pkl_path = f"{model_name}.pkl"
    joblib.dump(model, pkl_path)
    mlflow.log_artifact(pkl_path)
    if delete_after and os.path.exists(pkl_path):
        os.remove(pkl_path)


def log_time_diff(start_time: datetime.datetime, operation_name: str) -> datetime.datetime:
    """Logs the time difference from the start time to the current time for a specific operation.
    Args:
        start_time (datetime.datetime): The time when the operation started.
        operation_name (str): The name of the operation being timed.
    Returns:
        datetime.datetime: The current time after logging the time difference.
    """
    time_diff = round((datetime.datetime.now() - start_time).total_seconds() / 60, 4)
    mlflow.log_param(f"{operation_name.lower().replace(' ', '_')}_time", time_diff)
    return datetime.datetime.now()


def train_base_model(model_name: str, data: dict, parent_run_id: str) -> dict:
    """Trains a base model and logs the results in MLflow.
    Args:
        model_name (str): Name of the model to train.
        data (dict): Dictionary containing training and testing data.
        parent_run_id (str): ID of the parent MLflow run.
    Returns:
        dict: Dictionary containing the results of the training, including model, score, and best parameters.
    """
    run_name = f"train_{model_name}"
    with mlflow.start_run(run_name=run_name.lower(), nested=True, parent_run_id=parent_run_id) as child_run:
        child_run_id = child_run.info.run_id
        mlflow.log_params({"model_name": model_name})
        print(f"Starting training for {model_name}...")
        start_time = datetime.datetime.now()
        trainer = TrainingFactory.create_trainer(model_name=model_name)
        result = trainer.fit(data, parent_run_id=child_run_id)
        training_time = (datetime.datetime.now() - start_time).total_seconds() / 60
        print(f"Training completed for {model_name} in {training_time:.2f} minutes")
        mlflow.log_param("training_time", training_time)
        metrics_to_log = [
            "cv_score", "train_score", "test_score", "train_f1_score",
            "train_precision", "test_f1_score", "test_precision",
            "train_recall", "test_recall", "train_accuracy", "test_accuracy"
        ]
        for metric in metrics_to_log:
            mlflow.log_metric(metric, result[metric])
        mlflow.log_param("best_params", str(result["best_params"]))
        log_model_as_artifact(result["model"], f"{model_name}")
        path_plots = train_test_plots(result["model"], data, model_name)
        for plot_type, path_plot in path_plots.items():
            mlflow.log_artifact(path_plot)
            os.remove(path_plot)
            print(f"Created {plot_type} plot for {model_name}")
        result["id_experiment_child"] = child_run_id
        print(f"Completed training for {model_name} with score: {result['cv_score']}")
        print(f"Train score {result['train_score']} - Test score {result['test_score']}")
        print(f"Train Recall {result['train_recall']} - Test Recall {result['test_recall']}")
        print(f"Train Precision {result['train_precision']} - Test Precision {result['test_precision']}")
        print(f"Train f1 {result['train_f1_score']} - Test f1 {result['test_f1_score']}")
        return result


def train_superlearner(
    data: dict,
    base_models: list,
    best_score_base_models: float,
    type_ensemble: str,
    parent_run_id: str
) -> dict:
    """Trains a SuperLearner model using the provided base models and logs the results in MLflow.
    Args:
        data (dict): Dictionary containing training and testing data.
        base_models (list): List of dictionaries containing base model results.
        best_score_base_models (float): Best score from the full fit models.
        type_ensemble (str): Type of ensemble to create ('stacking' or 'voting').
        parent_run_id (str): ID of the parent MLflow run.
    Returns:
        dict: Dictionary containing the results of the SuperLearner training, including model, score,
              estimator names, and best parameters.
    """
    with mlflow.start_run(
            run_name=f"train_{type_ensemble}".lower(), nested=True, parent_run_id=parent_run_id) as child_run:
        child_run_id = child_run.info.run_id
        mlflow.log_params({"model_name": type_ensemble})
        start_time = datetime.datetime.now()
        superlearner = TrainingFactory.create_trainer(
            model_name="superlearner",
            base_models=base_models,
            best_score_base_models=best_score_base_models,
            type_ensemble=type_ensemble
        )
        result = superlearner.fit(data)
        training_time = (datetime.datetime.now() - start_time).total_seconds() / 60
        mlflow.log_param("training_time", training_time)
        mlflow.log_param("training_time", training_time)
        metrics_to_log = [
            "cv_score", "train_score", "test_score", "train_f1_score",
            "train_precision", "test_f1_score", "test_precision",
            "train_recall", "test_recall", "train_accuracy", "test_accuracy"
        ]
        for metric in metrics_to_log:
            mlflow.log_metric(metric, result[metric])
        mlflow.log_param("best_params", str(result["best_params"]))
        log_model_as_artifact(result["model"], f"{type_ensemble}")
        path_plots = train_test_plots(result["model"], data, type_ensemble)
        for plot_type, path_plot in path_plots.items():
            mlflow.log_artifact(path_plot)
            os.remove(path_plot)
            print(f"Created {plot_type} plot for {type_ensemble}")
        result["id_experiment_child"] = child_run_id
        print(f"Completed training for {type_ensemble} with score: {result['cv_score']}")
        print(f"Train score {result['train_score']} - Test score {result['test_score']}")
        print(f"Train Recall {result['train_recall']} - Test Recall {result['test_recall']}")
        print(f"Train Precision {result['train_precision']} - Test Precision {result['test_precision']}")
        print(f"Train f1 {result['train_f1_score']} - Test f1 {result['test_f1_score']}")
        return result


def run():
    with mlflow.start_run() as parent_run:
        parent_run_id = parent_run.info.run_id
        current_time = datetime.datetime.now()

        data_handler = DataHandler()
        data = data_handler.fetch_data()
        print("Data fetched successfully.")

        current_time = log_time_diff(current_time, "Data fetch")
        models_names = set(TrainingFactory._trainers.keys()) - {"superlearner"}
        base_models = []
        for model_name in models_names:
            model_result = train_base_model(model_name, data, parent_run_id)
            base_models.append(model_result)
        current_time = log_time_diff(current_time, "Base models training")
        print("Base models trained successfully.")
        mlflow.set_tag("base_models_training_status", "completed")

        best_score_base_models = float(max(base_models, key=lambda x: x["cv_score"])["cv_score"])

        ensemble_models = []
        for ensemble_type in ["stacking", "voting"]:
            super_learner_result = train_superlearner(
                data,
                base_models,
                best_score_base_models,
                ensemble_type,
                parent_run_id
            )
            current_time = log_time_diff(current_time, f"SuperLearner - {ensemble_type.capitalize()} training")
            ensemble_models.append(super_learner_result)
        print("SuperLearner training completed.")
        mlflow.set_tag("superlearners_training_status", "completed")

        models = data_handler.merge_results(ensemble_models, base_models)
        best_model = models[-1]
        mlflow.log_params({"best_model_name": best_model["name"], "best_params": str(best_model["model_data"]["best_params"])})
        mlflow.log_metric("best_model_score", best_model["score"])
        print(f"Best model: {best_model['name']} with score: {best_model['score']}")
        print(f"Best model parameters: {best_model['model_data']['best_params']}")
        best_model_data = best_model["model_data"]
        log_model_as_artifact(
            best_model_data["model"],
            f"{best_model['name']}",
        )
        mlflow.set_tag("best_model_status", "completed")

        models_registry(models, data["test_features"].iloc[0:1].copy())
        current_time = log_time_diff(current_time, "Model registration")
        mlflow.set_tag("model_registration_status", "completed")

        data_handler.save_results(
            id_experiment_parent=parent_run_id,
            models=models,
        )
        current_time = log_time_diff(current_time, "Results saving")
        mlflow.set_tag("results_saving_status", "completed")


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    print(f"MLflow tracking URI after set: {mlflow.get_tracking_uri()}")
    experiment_name = "meli_training"
    mlflow.set_experiment(experiment_name)
    run()