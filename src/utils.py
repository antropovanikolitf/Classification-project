"""
Utility functions for wine classification project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve
import pickle


def load_wine_data(red_path, white_path):
    """
    Load and combine red and white wine datasets.

    Parameters:
    -----------
    red_path : str
        Path to red wine CSV file
    white_path : str
        Path to white wine CSV file

    Returns:
    --------
    pd.DataFrame
        Combined dataset with wine_type column (0=red, 1=white)
    """
    red_wine = pd.read_csv(red_path, delimiter=';')
    white_wine = pd.read_csv(white_path, delimiter=';')

    red_wine['wine_type'] = 0
    white_wine['wine_type'] = 1

    combined = pd.concat([red_wine, white_wine], ignore_index=True)
    return combined


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation with metrics and visualizations.

    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    model_name : str
        Name for display

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    report = classification_report(y_test, y_pred, output_dict=True)

    # ROC-AUC if model supports probability
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
    except:
        auc_score = None
        y_proba = None

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'auc': auc_score,
        'predictions': y_pred,
        'probabilities': y_proba,
        'classification_report': report
    }

    return results


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                         labels=['Red', 'White'], ax=None):
    """
    Plot confusion matrix with annotations.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str
        Plot title
    labels : list
        Class labels
    ax : matplotlib axis
        Axis to plot on (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    return ax


def plot_feature_importance(model, feature_names, top_n=10, ax=None):
    """
    Plot feature importance for tree-based models.

    Parameters:
    -----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int
        Number of top features to display
    ax : matplotlib axis
        Axis to plot on
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)

    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.invert_yaxis()

    return ax


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plot ROC curves for multiple models.

    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model}
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    """
    plt.figure(figsize=(10, 8))

    for name, model in models_dict.items():
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        except:
            print(f"Could not compute ROC curve for {name}")

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_learning_curves(model, X, y, cv=5, scoring='accuracy'):
    """
    Plot learning curves to diagnose overfitting/underfitting.

    Parameters:
    -----------
    model : sklearn estimator
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Labels
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation score')

    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.1)
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.1)

    plt.xlabel('Training Set Size')
    plt.ylabel(f'{scoring.capitalize()} Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def save_model(model, filepath):
    """
    Save model to pickle file.

    Parameters:
    -----------
    model : object
        Model to save
    filepath : str
        Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load model from pickle file.

    Parameters:
    -----------
    filepath : str
        Path to model file

    Returns:
    --------
    object
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def create_model_summary_table(results_list):
    """
    Create summary table comparing multiple models.

    Parameters:
    -----------
    results_list : list
        List of results dictionaries from evaluate_model()

    Returns:
    --------
    pd.DataFrame
        Summary comparison table
    """
    summary = pd.DataFrame(results_list)
    summary = summary[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'auc']]
    summary = summary.round(4)
    summary = summary.sort_values('accuracy', ascending=False)
    return summary


def identify_misclassified_samples(X, y_true, y_pred, feature_names):
    """
    Analyze characteristics of misclassified samples.

    Parameters:
    -----------
    X : array-like
        Features
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    feature_names : list
        Names of features

    Returns:
    --------
    pd.DataFrame
        Analysis of misclassified samples
    """
    misclassified_idx = np.where(y_true != y_pred)[0]

    if len(misclassified_idx) == 0:
        print("No misclassified samples!")
        return None

    X_df = pd.DataFrame(X, columns=feature_names)

    misclassified = X_df.iloc[misclassified_idx]
    correctly_classified = X_df.drop(misclassified_idx)

    analysis = pd.DataFrame({
        'Feature': feature_names,
        'Misclassified_Mean': misclassified.mean(),
        'Correct_Mean': correctly_classified.mean(),
        'Difference': misclassified.mean() - correctly_classified.mean(),
        'Abs_Difference': abs(misclassified.mean() - correctly_classified.mean())
    })

    analysis = analysis.sort_values('Abs_Difference', ascending=False)

    return analysis