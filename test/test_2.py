import sys
import os
# Get the path to the root of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to Python's search path
sys.path.insert(0, project_root)
from autofeatureselect.filters import mutual_info_filter
import pandas as pd
import numpy as np

data = {
    'feature1': [1, 2, 3, 4, 5],         # Strongly related to target
    'feature2': [0, 0, 0, 0, 0],         # Useless
    'feature3': [10, 10, 10, 10, 10],    # Useless (constant)
    'target':   [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)

print("\nBefore mutual_info_filter:")
print(df)

df_filtered = mutual_info_filter(df, target='target', problem_type='regression', threshold=0.01)

print("\nAfter mutual_info_filter:")
print(df_filtered)
