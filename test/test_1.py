import sys
import os
# Get the path to the root of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to Python's search path
sys.path.insert(0, project_root)
from autofeatureselect.filters import low_variance_filter
import pandas as pd
# Create dataset with a constant and low-variance feature
data = {
    'feature1': [1, 1, 1, 1, 1],  # constant
    'feature2': [0, 0, 1, 0, 0],  # low variance
    'feature3': [10, 20, 30, 40, 50],  # useful
    'target': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("\nBefore low_variance_filter:")
print(df)

df_filtered = low_variance_filter(df, target='target', threshold=0.1)

print("\nAfter low_variance_filter:")
print(df_filtered)
