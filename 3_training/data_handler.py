from typing import Dict, Any

import pandas as pd
from sqlalchemy import create_engine
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


class DataHandler:

    def __init__(self, path: str = '../data/processed/data.parquet', oversampling: bool = False):
        self.path = path
        self.oversampling = oversampling

    def fetch_data(self) -> Dict[str, Any]:
        self.data = pd.read_parquet(self.path)
        self.__validate_data(self.data)
        dataset = self.__split_data(self.data)
        return dataset

    def __validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the input DataFrame to ensure it contains the required columns
        and does not contain null values.
        Args:
            df (pd.DataFrame): Input DataFrame to validate.
            selected_features (list): List of selected features to validate.
        Raises:
            KeyError: If any required column is not found in the DataFrame.
            ValueError: If the DataFrame contains null values.
        """
        required_columns = set(['A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Monto', 'Fraude', 'J_BR', 'J_MX', 'J_OTHERS'])
        # Validate that the DataFrame contains the required columns
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}.")
            raise KeyError(f"Missing required columns: {missing_columns}")
        # Validate that the DataFrame does not contain null values
        null_columns = df.columns[df.isna().any()].tolist()
        if null_columns:
            self.logger.error(f"DataFrame contains null values in columns: {null_columns}")
            raise ValueError(f"Null values found in columns: {null_columns}")

    def __split_data(
        self,
        df: pd.DataFrame,
        random_state: int = 42
    ) -> Dict[str, Any]:

        X = df.drop(columns=['Fraude'])
        y = df['Fraude']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=random_state
        )
        data_test = X_test.copy()
        data_test['Fraude'] = y_test
        total_monto = data_test[['Monto','Fraude']].apply(lambda x: x.Monto if x.Fraude == 1 else 0.25*x.Monto, axis=1).sum()
        print(f"Total monto: {total_monto}")
        target_counts = y_train.value_counts()
        n_minority = target_counts.min()
        minority_class = target_counts.idxmin()
        new_minority_size = int(n_minority * 1.25)
        sampling_strategy = {cls: count for cls, count in target_counts.items()}
        sampling_strategy[minority_class] = new_minority_size
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        return {
            'train_features': X_train_res,
            'train_target': y_train_res,
            'test_features': X_test,
            'test_target': y_test,
            'train_monto': X_train_res['Monto'],
            'test_monto': X_test['Monto']
        }

    def merge_results(self, ensemble_models: dict, models_full_fit: list) -> list:
        """
        Merge results from the ensemble models and full fit models into a single list.
        Args:
            ensemble_models (dict): Dictionary containing the ensemble model results.
            models_full_fit (list): List of dictionaries containing full fit model results.
        Returns:
            list: Merged list of model results, sorted by score.
        """
        all_models = []
        for model in ensemble_models:
            all_models.append({"name": model["model_name"], "score": model["test_score"], "model_data": model})
        for model in models_full_fit:
            all_models.append({"name": model["model_name"], "score": model["test_score"], "model_data": model})
        all_models.sort(key=lambda x: x["score"])
        return all_models

    def save_results(self, id_experiment_parent: str, models: list) -> None:
        """
        Save model results to the database with proper ranking and selection.

        Args:
            id_experiment (str): The experiment identifier
            models (list): List of model dictionaries containing results
        """
        df = pd.DataFrame(
            [
                {
                    "id_experiment_parent": id_experiment_parent,
                    "id_experiment_child": model["model_data"]["id_experiment_child"],
                    "model": model["name"],
                    "metric": model["score"],
                }
                for model in models
            ]
        )
        df["rank"] = df["metric"].rank(method="min", ascending=False)

        engine = create_engine("sqlite:///meli.db", echo=False)
        df.to_sql('training', con=engine, if_exists='append', index=False)
        print(f"Saved {len(models)} model results to database.")
