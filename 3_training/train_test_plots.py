import os

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def train_test_plots(model, data: dict, model_name: str) -> dict:
    """
    Generate confusion matrices for training and test data.

    Args:
        model: Trained model object with predict method
        data (dict): Dictionary containing train/test features and targets
        model_name (str): Name of the model for plot titles

    Returns:
        dict: Dictionary with 'train' and 'test' as keys and file paths as values
    """
    y_train_pred = model.predict(data['train_features'])
    y_test_pred = model.predict(data['test_features'])

    cm_train = confusion_matrix(data['train_target'], y_train_pred)
    cm_test = confusion_matrix(data['test_target'], y_test_pred)

    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fraud', 'Fraud'],
                yticklabels=['No Fraud', 'Fraud'])
    plt.title(f'{model_name} - Training Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    train_path = os.path.join(plots_dir, f'{model_name}_train_confusion_matrix.png')
    plt.savefig(train_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fraud', 'Fraud'],
                yticklabels=['No Fraud', 'Fraud'])
    plt.title(f'{model_name} - Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    test_path = os.path.join(plots_dir, f'{model_name}_test_confusion_matrix.png')
    plt.savefig(test_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'train': train_path,
        'test': test_path
    }
