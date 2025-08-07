import sys
import os
# Get the path to the root of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to Python's search path
sys.path.insert(0, project_root)
from sklearn.datasets import load_breast_cancer
import pandas as pd
from autofeatureselect.model_wrappers import shap_filter

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Original shape:", X.shape)

# Apply SHAP-based filtering
X_shap = shap_filter(X, y, top_n=10)

print("Filtered shape:", X_shap.shape)
print("Remaining features:\n", X_shap.columns.tolist())
