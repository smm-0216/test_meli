import os
import joblib

import mlflow


def models_registry(models: list, input_example: dict) -> None:
    """    Registers trained models in MLflow with appropriate metadata and input examples.
    Args:
        models (list): List of dictionaries containing model information.
        id_stock (str): Stock identifier.
        forecast_period (str): Forecast period.
        input_example (dict): Example input data for the model.
        logger (object): Logger object for logging information.
    """
    client = mlflow.tracking.MlflowClient()
    for model in models:
        model_name = model['name']
        model_data = model['model_data']['model']
        model_score = model['score']
        model_filename = f"{model_name}_meli.pkl".lower()
        model_path = os.path.abspath(model_filename)
        joblib.dump(model_data, model_path)
        registered_model_name = f"{model_name}".lower()
        print(f"Registering model: {registered_model_name} with score: {model_score}")
        if model_name == 'xgboost':
            mlflow.xgboost.log_model(
                model_data,
                artifact_path=model_filename,
                registered_model_name=registered_model_name,
                input_example=input_example
            )
        elif model_name == 'lightgbm':
            mlflow.lightgbm.log_model(
                model_data,
                artifact_path=model_filename,
                registered_model_name=registered_model_name,
                input_example=input_example
            )
        elif model_name == 'catboost':
            mlflow.catboost.log_model(
                model_data,
                artifact_path=model_filename,
                registered_model_name=registered_model_name,
                input_example=input_example
            )
        elif model_name in ['voting', 'stacking']:
            mlflow.sklearn.log_model(
                model_data,
                artifact_path=model_filename,
                registered_model_name=registered_model_name,
                input_example=input_example
            )
        latest_versions = client.get_latest_versions(registered_model_name)
        if latest_versions:
            model['version'] = latest_versions[0].version
        else:
            model['version'] = 1
        print(f"Model {model_name} registered as {registered_model_name} with version {model['version']}")
        if os.path.exists(model_path):
            os.remove(model_path)
