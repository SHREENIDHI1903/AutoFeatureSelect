import sys
import os
# Get the path to the root of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to Python's search path
sys.path.insert(0, project_root)

from autofeatureselect.model_wrappers import tree_importance_filter
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Original shape:", X.shape)

#X_filtered = tree_importance_filter(X, y, importance_threshold=0.02)

#print("Filtered shape:", X_filtered.shape)
#print("Remaining features:\n", X_filtered.columns.tolist())

X_top10 = tree_importance_filter(X, y, top_n=10)

print("Top 10 features:\n", X_top10.columns.tolist())


