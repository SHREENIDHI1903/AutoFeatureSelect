
# 🔍 AutoFeatureSelect

A lightweight, modular Python package for **automatic feature selection** in machine learning workflows.  
It helps data scientists and ML engineers **remove irrelevant, redundant, or noisy features** to improve model performance, reduce overfitting, and simplify datasets — without manual guesswork.

---

## ✨ Why AutoFeatureSelect?

Most automated ML tools focus on **feature generation**, not feature elimination.

But **throwing more features** at a model often hurts performance. What if you want to **cut the noise**?

That's where `AutoFeatureSelect` shines:
- ✅ Selects only the most **relevant features**
- ✅ Works on any **tabular dataset**
- ✅ No need to generate new features
- ✅ Transparent, modular, and customizable
- ✅ Compatible with any ML framework (`scikit-learn`, `XGBoost`, `LightGBM`, etc.)

---

## 🚀 Installation

```bash
pip install autofeatureselect
```

---

## 📦 What This Package Does

`AutoFeatureSelect` provides a growing set of **feature selection tools**, including:

| Method                    | Type     | Description                                                        |
|---------------------------|----------|--------------------------------------------------------------------|
| `low_variance_filter`     | Filter   | Removes features with very low variance (near-constant columns)   |
| `correlation_filter`      | Filter   | Drops features that are highly correlated with others              |
| `mutual_info_filter`      | Filter   | Selects features with high mutual information with target          |
| `vif_filter`              | Filter   | Removes multicollinear features using Variance Inflation Factor    |
| `rfe_filter`              | Wrapper  | Uses model-based recursive elimination of weakest features         |
| `tree_importance_filter`  | Embedded | Uses tree model importance scores for feature selection            |
| `shap_filter`             | Embedded | Uses SHAP values to rank and select top features                   |
---
## 📌 Functions and Their Parameters

### 🔹 filters.py

| Function               | Parameters                                                                 |
|------------------------|----------------------------------------------------------------------------|
| `correlation_filter`   | `df`, `target`, `threshold=0.9`                                            |
| `low_variance_filter`  | `df`, `target=None`, `threshold=0.01`                                      |
| `mutual_info_filter`   | `df`, `target`, `problem_type="classification"`, `threshold=0.01`          |
| `vif_filter`           | `df`, `target=None`, `threshold=5.0`, `verbose=True`                       |

---

### 🔹 model_wrappers.py

| Function               | Parameters                                                                 |
|------------------------|----------------------------------------------------------------------------|
| `tree_importance_filter` | `X`, `y`, `model=None`, `importance_threshold=0.01`, `top_n=None`, `verbose=True` |
| `rfe_filter`           | `X`, `y`, `model=None`, `n_features_to_select=10`, `verbose=True`          |
| `shap_filter`          | `X`, `y`, `model=None`, `top_n=10`, `verbose=True`                         |

---

### ✅ Notes

- Only `tree_importance_filter` uses a threshold-like parameter, named `importance_threshold`.
- `rfe_filter` uses `n_features_to_select` to define how many features to retain.
- `shap_filter` uses `top_n` to keep the most impactful features based on SHAP values.


---

## 🔧 Example Usage

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from autofeatureselect.filters import (
    low_variance_filter,
    correlation_filter,
    mutual_info_filter,
    vif_filter
)
from autofeatureselect.model_wrappers import (
    tree_importance_filter,
    rfe_filter,
    shap_filter
)

# Load your dataset
df = pd.read_csv("data.csv")
target = "target_column"

# 1. Apply Low Variance Filter
df = low_variance_filter(df, threshold=0.01)

# 2. Apply Correlation Filter
df = correlation_filter(df, target=target, threshold=0.9)

# 3. Apply Mutual Information Filter
df = mutual_info_filter(df, target=target, problem_type="classification", threshold=0.01)

# 4. Apply VIF Filter
df = vif_filter(df, threshold=5.0)

# Separate features and target for model-based methods
X = df.drop(columns=[target])
y = df[target]

# 5. Apply Tree-Based Feature Importance
X = tree_importance_filter(X, y, model=RandomForestClassifier())

# 6. Apply RFE (Recursive Feature Elimination)
X = rfe_filter(X, y, model=RandomForestClassifier())

# 7. Apply SHAP-Based Selection
X = shap_filter(X, y, model=RandomForestClassifier())

# Combine X and y back
df_selected = pd.concat([X, y], axis=1)

# Final selected features
print(df_selected.head())

```

---

## 💡 When to Use Which Method?

| Method                  | Best For                                | Notes                                     |
|-------------------------|------------------------------------------|-------------------------------------------|
| `low_variance_filter`   | Removing constant or near-constant columns | Fast and simple                           |
| `correlation_filter`    | Redundant features (highly correlated)   | Pairwise comparison                        |
| `mutual_info_filter`    | Non-linear relevance to target           | Requires `problem_type` parameter         |
| `vif_filter`            | Multicollinearity                        | Good before linear models                 |
| `rfe_filter`            | Model-driven selection                   | Slower but accurate                       |
| `tree_importance_filter`| Tree models (e.g. XGBoost, RandomForest) | Uses feature importances from trees       |
| `shap_filter`           | Global interpretability, model-agnostic | Uses SHAP values, slower but insightful   |
---
## 🔍 How to Explore Available Functions

To explore the available feature selection functions and understand their parameters, you can use built-in Python tools in any Python environment (terminal, IDE, or Jupyter Notebook).

---

### 📘 1. List All Functions in a Module

You can list all functions within a module using `dir()`:

```python
import autofeatureselect.filters as filters
import autofeatureselect.model_wrappers as models

# List functions in the filters module
print(dir(filters))

# List functions in the model_wrappers module
print(dir(models))
```
---
### 📑 2. Get Help on a Specific Function
Use the help() function to view the docstring of a specific function, including description, parameters, and return values:
```python
from autofeatureselect.filters import mutual_info_filter

help(mutual_info_filter)
```
---


## 🔍 How It's Different from Other Tools

| Feature | AutoFeatureSelect | Featuretools | tsfresh | feature-engine | scikit-learn |
|--------|--------------------|--------------|---------|----------------|--------------|
| Focus | Feature **Selection** | Feature **Generation** | Feature **Extraction** | Manual Engineering | General ML |
| Auto-generate new features | ❌ | ✅ | ✅ | ⚠️ | ⚠️ |
| Remove irrelevant features | ✅ | ❌ | ⚠️ Partial | ✅ | ✅ |
| Targeted data | Tabular | Relational | Time-series | Tabular | All |
| Easy to use | ✅ | ❌ | ❌ | ⚠️ | ✅ |
| Beginner-friendly | ✅ | ❌ | ❌ | ⚠️ | ⚠️ |

---

## 🧪 Testing the Package

```bash
# Run from the root of the project directory
pytest test/
```

Ensure `pytest` is installed:
```bash
pip install pytest
```

---

## 📁 Project Structure

```
autofeatureselect/
│
├── filters.py # Statistical methods
│ ├── low_variance_filter()
│ ├── correlation_filter()
│ └── mutual_info_filter()
│
├── model_wrappers.py # Model-based methods
│ ├── tree_importance_filter()
│ ├── rfe_filter()
│ └── shap_filter()
│
└── init.py # Exposes selected functions
```

---

## 📘 Documentation

Every function is self-contained and comes with:

- Docstrings
- Clear parameters
- Defaults set to typical values
- Returns filtered `DataFrame` (with target column)

You can use Python’s built-in help:

```python
from autofeatureselect.filters import mutual_info_filter
help(mutual_info_filter)
```

Or read the source — it’s clean and readable!

---

## 🤔 What It Does *Not* Do

- ❌ It does not create new features (use `Featuretools`, `tsfresh`, or `scikit-learn` for that)
- ❌ It does not automate the entire pipeline (no one-size-fits-all — feature selection must be task-specific!)
- ❌ It does not hide logic in “black boxes” — everything is transparent and user-controllable

---

## 🙌 Contributing

Want to add your own feature selection method?  
PRs and discussions welcome! Just follow the modular style, and document everything clearly.

---

## 👤 Author

**Shreenidhi TH**  
Developer passionate about building tools for applied ML and automation.

---
## License

This project is licensed under the [MIT License](./LICENSE).

----
