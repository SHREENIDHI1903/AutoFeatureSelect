import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import shap


def tree_importance_filter(X, y, model=None, importance_threshold=0.01, top_n=None, verbose=True):
    """
    Selects features based on importance scores from a tree-based model (e.g., RandomForest).
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series or np.array): Target values
        model (sklearn model): Tree-based model to use (optional)
        importance_threshold (float): Minimum importance to retain
        top_n (int or None): If set, keeps only top N features by importance
        verbose (bool): Print removed features
    
    Returns:
        pd.DataFrame: Filtered DataFrame with selected features
    """

    # Choose default model if none provided
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42) if is_classifier(y) else RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model.fit(X, y)

    # Get importance scores
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    })

    if top_n is not None:
        # Keep only top N
        selected_features = importance_df.sort_values(by='importance', ascending=False).head(top_n)['feature'].tolist()
    else:
        # Keep all above the threshold
        selected_features = importance_df[importance_df['importance'] >= importance_threshold]['feature'].tolist()

    if verbose:
        removed = set(X.columns) - set(selected_features)
        print(f"Tree Importance Filter removed {len(removed)} features: {removed}")

    return X[selected_features]




def rfe_filter(X, y, model=None, n_features_to_select=10, verbose=True):
    """
    Selects features using Recursive Feature Elimination (RFE).
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series or np.array): Target values
        model (sklearn estimator): Estimator to use (if None, uses LogisticRegression or RandomForest)
        n_features_to_select (int): Number of top features to keep
        verbose (bool): Print removed features or not
        
    Returns:
        pd.DataFrame: Filtered DataFrame with selected features
    """
    if model is None:
        if is_classifier(y):
            model = LogisticRegression(solver='liblinear', max_iter=1000)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

    selector = RFE(estimator=model, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)

    selected_features = X.columns[selector.support_].tolist()
    removed_features = list(set(X.columns) - set(selected_features))

    if verbose:
        print(f"RFE removed {len(removed_features)} features: {removed_features}")

    return X[selected_features]

def shap_filter(X, y, model=None, top_n=10, verbose=True):
    """
    Selects top N features based on SHAP values.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series or array): Target values
        model (sklearn estimator): Model to compute SHAP values. Defaults to RandomForest.
        top_n (int): Number of top features to keep
        verbose (bool): If True, prints removed features

    Returns:
        pd.DataFrame: DataFrame with top_n most important features
    """
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42) if is_classifier(y) \
            else RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Compute mean absolute SHAP value per feature
    mean_shap = pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().sort_values(ascending=False)

    top_features = mean_shap.head(top_n).index.tolist()
    removed_features = list(set(X.columns) - set(top_features))

    if verbose:
        print(f"SHAP filter removed {len(removed_features)} features: {removed_features}")

    return X[top_features]


