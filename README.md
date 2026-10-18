# 🚀 ShopSmart: Online Shoppers Purchasing Behavior Predictor

[![License: MIT](https://shields.io)](https://opensource.org)
[![Python: 3.13+](https://shields.io)](https://python.org)
[![Scikit-Learn](https://shields.io)](https://scikit-learn.org)

## 📌 Problem Statement
An e-commerce company, **ShopSmart**, aims to better understand the behavior of visitors browsing its online store to optimize marketing strategies and capture lost revenue. This project implements a **Decision Tree-based classification model** built to accurately analyze session telemetry and predict whether a unique website visitor will complete a purchase (`Revenue`).

The target variable is highly imbalanced, making the **F1-score** our primary evaluation metric, with an initial project benchmark of **0.55**.

## ✨ Features & Architecture
* ⚙️ **Automated Preprocessing Pipeline:** Utilizes a `ColumnTransformer` to seamlessly scale continuous session metrics (`StandardScaler`) and encode categorical visitor attributes (`OneHotEncoder`).
* 📊 **Imbalance Handling:** Controls class weights dynamically within the `DecisionTreeClassifier` logic via `class_weight="balanced"`.
* ✂️ **Pruning & Regularization:** Incorporates hyperparameter tuning via `GridSearchCV` to determine optimal path constraints (`max_depth`, `min_samples_leaf`) to combat overfitting.

## 🛠️ Dataset Description
The model trains on **12,330 individual user sessions** tracking metrics across several distinct features:
* **Numerical Metrics:** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, and `SpecialDay`.
* **Categorical Metrics:** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, and `Weekend`.
* **Target Label:** `Revenue` (Binary classification: `1` for purchase, `0` otherwise).

## 🚀 Getting Started

### Prerequisites
Ensure you have a Python environment set up with necessary package managers:
```bash
pip install pandas numpy scikit-learn
```

### Quick Start & Execution
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd shopsmart-purchase-predictor
   ```
2. **Run your notebook or driver script:**
   Place `online_shoppers.csv` in the root folder and run your pipeline script or open the notebook matching your file contents.

## 📈 Model Performance & Tuning

### Baseline Decision Tree Metrics
* **F1-Score (Purchases):** `0.6278` (Surpasses project benchmark of 0.55)
* **Overall Accuracy:** `85%`

```text
Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.85      0.90      2084
           1       0.50      0.83      0.63       382
```

### Hyperparameter Grid Optimization
The final iteration searches an optimization matrix via cross-validation to isolate maximum scoring structures:
```python
param_grid = {
    "model__max_depth":,
    "model__min_samples_leaf": [20, 30, 50]
}
```
* **Optimized F1 Score:** `0.6343`
* **Best Identified Parameters:** `{'model__max_depth': 4, 'model__min_samples_leaf': 50}`

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
