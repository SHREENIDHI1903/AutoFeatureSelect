# test_script.py
import sys
import os
# Get the path to the root of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to Python's search path
sys.path.insert(0, project_root)

from sklearn.datasets import load_diabetes
import pandas as pd
from autofeatureselect.filters import vif_filter

# Load sample dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Combine features and target for testing
df = X.copy()
df['target'] = y

# Apply VIF filter
print("Before VIF filter:", df.shape)
df_filtered = vif_filter(df, target='target', threshold=5.0, verbose=True)
print("After VIF filter:", df_filtered.shape)

# Check remaining features
print("Remaining features:", df_filtered.columns.tolist())
