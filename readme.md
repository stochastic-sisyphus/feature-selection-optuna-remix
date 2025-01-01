# Feature Selection with Optuna

A Python implementation of feature selection methods combining traditional approaches (PCA, LASSO) with Optuna-based optimization. This project implements the methodology described in our research paper "Expanding Optuna's Optimization Principles: Advanced Feature Engineering and Selection Strategies."

## Overview

This project implements an interaction-aware feature selection framework that combines:
- Traditional dimensionality reduction (PCA)
- Sparse feature selection (LASSO)
- Optuna-based optimization with interaction awareness
- Comprehensive visualization suite

## Key Features

- **Interaction-Aware Feature Selection:**
  ```python
  I(f_i) = α⋅MI(f_i; y) + β⋅∑(j≠i) I_interaction(f_i, f_j)
  ```
  where:
  - α = 0.7 (mutual information weight)
  - β = 0.3 (interaction weight)
  - MI = mutual information score
  - I_interaction = feature interaction score

- **Multiple Selection Methods:**
  - PCA with automated component selection
  - LASSO with L1 regularization
  - Optuna with TPE sampler and pruning

## Datasets

### Included Datasets

1. **Iris Dataset** (`data/Iris Dataset/bezdekIris.data`)
   - 150 samples with 4 features
   - Features: sepal length, sepal width, petal length, petal width
   - Target: 3 different iris species
   - Source: UCI Machine Learning Repository

2. **Adult Dataset** (`data/Adult Dataset/adult.data`)
   - 32,561 samples with 14 features
   - Features: age, workclass, education, etc.
   - Target: income >50K or <=50K
   - Source: UCI Machine Learning Repository

### Using the Datasets

```python
# Using Iris Dataset
iris_data = pd.read_csv('data/Iris Dataset/bezdekIris.data', header=None)
X_iris = iris_data.iloc[:, :-1].values  # Features
y_iris = iris_data.iloc[:, -1].values   # Target
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Using Adult Dataset
adult_data = pd.read_csv('data/Adult Dataset/adult.data', header=None)
X_adult = adult_data.iloc[:, :-1].values  # Features
y_adult = adult_data.iloc[:, -1].values   # Target
```

## Installation

```bash
git clone https://github.com/stochastic-sisyphus/feature-selection-optuna-remix.git
cd feature-selection-optuna-remix
pip install -r requirements.txt
```

## Quick Start

```python
from feature_selection_optuna import FeatureSelector
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5)
feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

# Initialize and run feature selection
selector = FeatureSelector(X, y, feature_names, "My Dataset")
results = selector.run_all_methods()

# Print results
print("\nFeature Selection Results:")
for method, score in results.items():
    print(f"{method}: {score:.4f}")
```

## Example Output

When running the feature selection on the Iris dataset, the optimization process shows:

```
[I 2024-12-31 00:31:24,361] A new study created in memory with name: no-name-299a70ae-5d53-48e3-884c-7c10c26d4837
[I 2024-12-31 00:31:24,365] Trial 0 finished with value: 0.7853027705488896 and parameters: {'selected_features': '1,2,3'}
[I 2024-12-31 00:31:24,432] Trial 24 finished with value: 1.0025102205623477 and parameters: {'selected_features': '2'}
Best is trial 24 with value: 1.0025102205623477
```

### Analysis Results

#### 1. Feature Importance
- **Petal Length** (Feature 2): Emerged as the most informative feature with MI score > 1.0
- **Petal Width** (Feature 3): Second most important, highly correlated with species classification
- **Sepal** measurements showed lower but complementary importance

#### 2. Method Comparison
```
Feature Selection Results:
PCA: 0.8245
LASSO: 0.9132
Optuna: 1.0025
```

#### 3. Runtime Performance
```
Method Runtimes (seconds):
PCA: 0.0023
LASSO: 0.0156
Optuna: 0.2341
```

## Methods

### PCA Baseline
- Implements Principal Component Analysis for dimensionality reduction
- Returns transformed data and component importance

### LASSO Baseline
- Uses L1 regularization for feature selection
- Returns selected features and their coefficients

### Optuna Selection
- Optimizes feature combinations using Optuna
- Uses mutual information scoring for evaluation
- Returns optimal feature subset and importance scores

## Requirements

- Python 3.8+
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=0.24.2
- optuna>=3.0.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- tqdm>=4.65.0
- joblib>=1.2.0
- plotly>=5.13.0

## Citation

```bibtex
@software{feature_selection_optuna_remix,
  author = {Stochastic Sisyphus},
  title = {Feature Selection with Optuna},
  year = {2024},
  url = {https://github.com/stochastic-sisyphus/feature-selection-optuna-remix}
}
```

## License

MIT License

## Acknowledgments

- Optuna team for the optimization framework
- scikit-learn team for machine learning tools
- The open-source community for inspiration and support