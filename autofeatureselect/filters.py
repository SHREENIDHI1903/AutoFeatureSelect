import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor


def correlation_filter(df, target, threshold=0.9):
    """
    Removes highly correlated features from the dataset.

    Parameters:
    - df: pandas DataFrame containing features and target column.
    - target: string, name of the target column.
    - threshold: float, correlation level above which to drop features.

    Returns:
    - df_filtered: DataFrame with selected features.
    """
    df_features = df.drop(columns=[target])  # exclude target

    #Pearson correlation coefficient
    corr_matrix = df_features.corr().abs()   # absolute correlation

    # Upper triangle of the correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find columns with correlation above the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    df_filtered = df.drop(columns=to_drop)

    return df_filtered

def low_variance_filter(df, target=None, threshold=0.01):
    """
    Removes features with low variance (near-constant columns).

    Parameters:
    - df: pandas DataFrame (including target optionally)
    - target: name of the target column to exclude from filtering
    - threshold: minimum variance a feature must have to be kept

    Returns:
    - df_filtered: DataFrame with low-variance features removed
    """
    if target:
        features = df.drop(columns=[target])
    else:
        features = df.copy()

    #sets up the tool that will do the filtering
    selector = VarianceThreshold(threshold=threshold)

    #calculates the variance for each column
    selector.fit(features)

    # Get names of features that pass the threshold
    #compares each calculated variance to the threshold of 0.01
    kept_columns = features.columns[selector.get_support()]

    df_filtered = df[kept_columns]

    # Add target column back if needed
    if target:
        df_filtered[target] = df[target]

    return df_filtered



def mutual_info_filter(df, target, problem_type="classification", threshold=0.01):
    """
    Removes features with low mutual information with the target.

    Parameters:
    - df: pandas DataFrame
    - target: name of the target column
    - problem_type: "classification" or "regression"
    - threshold: minimum mutual information score to keep a feature

    Returns:
    - df_filtered: DataFrame with low-MI features removed
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    X = df.drop(columns=[target])
    y = df[target]

    if problem_type == "classification":
        mi_scores = mutual_info_classif(X, y, discrete_features="auto")
    elif problem_type == "regression":
        mi_scores = mutual_info_regression(X, y)
    else:
        raise ValueError("problem_type must be either 'classification' or 'regression'.")

    mi_series = pd.Series(mi_scores, index=X.columns)

    # Keep only features with MI score above the threshold
    selected_features = mi_series[mi_series > threshold].index.tolist()
    df_filtered = df[selected_features + [target]]

    return df_filtered


def vif_filter(df, target=None, threshold=5.0, verbose=True):
    """
    Removes features with high Variance Inflation Factor (multicollinearity).

    Parameters:
    - df: pandas DataFrame
    - target: name of the target column to exclude from VIF calculation
    - threshold: maximum acceptable VIF value (default is 5.0)
    - verbose: whether to print info during filtering

    Returns:
    - df_filtered: DataFrame with low-VIF features only
    """

    # Step 1: Remove target column if given
    if target:
        features = df.drop(columns=[target])
    else:
        features = df.copy()

    # Step 2: Drop any non-numeric columns
    features = features.select_dtypes(include=[np.number])

    dropped = True
    while dropped:
        dropped = False
        vif = pd.Series(
            [variance_inflation_factor(features.values, i) for i in range(features.shape[1])],
            index=features.columns
        )

        max_vif = vif.max()
        if max_vif > threshold:
            drop_feature = vif.idxmax()
            features = features.drop(columns=[drop_feature])
            dropped = True
            if verbose:
                print(f"Dropped '{drop_feature}' with VIF: {max_vif:.2f}")

    # Step 3: Add target column back if needed
    if target:
        features[target] = df[target]

    return features