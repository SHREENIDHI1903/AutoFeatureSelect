import sys
import os
# Get the path to the root of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to Python's search path
sys.path.insert(0, project_root)

from sklearn.datasets import load_breast_cancer
import pandas as pd
from autofeatureselect.model_wrappers import rfe_filter

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Original shape:", X.shape)

# Apply RFE to select top 10 features
X_rfe = rfe_filter(X, y, n_features_to_select=10)

print("Filtered shape:", X_rfe.shape)
print("Remaining features:\n", X_rfe.columns.tolist())
