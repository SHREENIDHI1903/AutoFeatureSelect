import sys
import os
# Get the path to the root of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to Python's search path
sys.path.insert(0, project_root)
import pandas as pd
from autofeatureselect.filters import correlation_filter

# Create a sample dataset
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 4, 6, 8, 10],  # perfectly correlated with feature1
    'feature3': [5, 4, 3, 2, 1],   # negatively correlated
    'target': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("Before filtering:")
print(df)

df_filtered = correlation_filter(df, target='target', threshold=0.9)

print("\nAfter filtering:")
print(df_filtered)
