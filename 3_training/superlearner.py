from typing import Union

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score


from evaluation import metric
from ensemble_classifiers import VotingClassifier, StackingClassifier


class SuperLearner:
    """SuperLearner class for stacking multiple base models with a meta model."""

    def __init__(self, base_models: list, best_score_base_models: float, type_ensemble: str):
        self.base_models = base_models
        self.best_score_base_models = best_score_base_models
        self.type_ensemble = type_ensemble
        self.train_features_key = "train_features"
        self.train_target_key = "train_target"

    def __evaluate_ensemble(self, estimators: list) -> float:
        """Evaluate an ensemble model using time series cross-validation
        Args:
            estimators (list): List of trained models to use as base estimators.
        Returns:
            float: Mean evaluation score for the ensemble model.
        """
        X = self.data[self.train_features_key]
        y = self.data[self.train_target_key]
        skf = StratifiedKFold(10, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            monto_val = X["Monto"].iloc[val_idx]
            model_objects = [est['model'] for est in estimators]
            if self.type_ensemble == "voting":
                model = VotingClassifier(estimators=model_objects)
            else:
                model = StackingClassifier(estimators=model_objects)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = metric(y_val, preds, monto_val)
            scores.append(score)
        mean_score = float(np.mean(scores))
        print(f"Ensemble model {[estimator['model_name'] for estimator in estimators]} evaluated with mean score: {mean_score}")
        return mean_score

    def __greedy_ensemble(self) -> tuple:
        """Select the best subset of models using a greedy approach with specific combinations
        Returns:
            tuple: (best_estimators_list, best_score) - The best combination and its score.
        """
        top_models = self.base_models.copy()
        top_models.sort(key=lambda d: d['test_score'], reverse=True)
        print(f"Base models sorted by test_score: {[float(model['test_score']) for model in top_models]}")
        print(f"Best score from base models: {self.best_score_base_models}")
        combinations_to_test = [
            [0, 1],        # Top 2 models
            [0, 2],        # 1st and 3rd models
            [0, 1, 2],     # Top 3 models
            [1, 2],        # 2nd and 3rd models
        ]
        combos_and_scores = {}
        for i, indices in enumerate(combinations_to_test):
            combo_list = [top_models[idx] for idx in indices]
            combo_name = " + ".join([model['model_name'] for model in combo_list])
            print(f"Evaluating {combo_name} in iteration {i + 1}")
            score = self.__evaluate_ensemble(combo_list)
            combos_and_scores[score] = combo_list
            if score > self.best_score_base_models:
                print(f"{combo_name} improves over best score {self.best_score_base_models}, returning early")
                break
        best_score = max(combos_and_scores.keys())
        best_estimators = combos_and_scores[best_score]
        print(f"Best ensemble found: {[model['model_name'] for model in best_estimators]} with score: {best_score}")
        return best_estimators, best_score

    def __train_ensemble_model(self, estimators: list) -> Union[StackingClassifier, VotingClassifier]:
        """Train the ensemble model using the selected estimators
        Args:
            estimators (list): List of selected base estimators.
        Returns:
            Union[StackingClassifier, VotingClassifier]: The trained model.
        """
        X_train = self.data[self.train_features_key]
        y_train = self.data[self.train_target_key]
        model_objects = [est['model'] for est in estimators]
        if self.type_ensemble == "voting":
            model = VotingClassifier(estimators=model_objects)
        else:
            model = StackingClassifier(estimators=model_objects)
        model.fit(X_train, y_train)
        return model

    def fit(self, data: dict):
        """Fit the superlearner model using the provided data
        Args:
            data (dict): Dictionary containing training and validation data.
        Returns:
            dict: Dictionary containing the fitted model, its name, best parameters, and final score.
        """
        self.data = data
        print(f"Training {self.type_ensemble} using greedy ensemble selection")
        best_estimators, score = self.__greedy_ensemble()
        trained_model = self.__train_ensemble_model(estimators=best_estimators)
        print(f"Trained {self.type_ensemble} model with best estimators: {[model['model_name'] for model in best_estimators]}")
        y_train_pred = trained_model.predict(data[self.train_features_key])
        y_test_pred = trained_model.predict(data["test_features"])
        train_monto = data['train_monto']
        test_monto = data['test_monto']
        return {
            "model_name": self.type_ensemble,
            "model": trained_model,
            "best_params": [model['model_name'] for model in best_estimators],
            "cv_score": score,
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
