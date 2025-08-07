
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

## 🔧 Example Usage

```python
import pandas as pd
from autofeatureselect.filters import (
    low_variance_filter,
    correlation_filter,
    mutual_info_filter,
    vif_filter
)
from autofeatureselect.wrappers import recursive_feature_elimination
from autofeatureselect.embedded import tree_based_selector, lasso_selector

# Load your dataset
df = pd.read_csv("data.csv")
target = "target_column"

# 1. Apply Low Variance Filter
df = low_variance_filter(df, target=target, threshold=0.01)

# 2. Apply Correlation Filter
df = correlation_filter(df, target=target, threshold=0.9)

# 3. Apply Mutual Information Filter (classification or regression)
df = mutual_info_filter(df, target=target, task="classification", top_k=15)

# 4. Apply VIF Filter (multicollinearity)
df = vif_filter(df, target=target, threshold=5.0)

# 5. Apply Recursive Feature Elimination
df = recursive_feature_elimination(df, target=target, task="classification", n_features_to_select=10)

# 6. Apply Tree-Based Feature Selection
df = tree_based_selector(df, target=target, task="classification", top_k=10)

# 7. Apply Lasso-Based Feature Selection
df = lasso_selector(df, target=target, task="regression", alpha=0.01)

# The resulting df will contain only selected features + target column
print(df.head())
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

## 🛠️ Future Plans

- [ ] Add support for PCA and statistical tests
- [ ] Add logging and visual reports
- [ ] Add command-line interface (CLI)
- [ ] Jupyter notebook demo and interactive docs

---

## 🙌 Contributing

Want to add your own feature selection method?  
PRs and discussions welcome! Just follow the modular style, and document everything clearly.

---

## 📄 License

MIT License – use freely for personal and commercial projects.

---

## 👤 Author

**Shreenidhi TH**  
Developer passionate about building tools for applied ML and automation.

---
