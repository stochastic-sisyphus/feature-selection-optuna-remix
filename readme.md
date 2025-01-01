# Feature Selection with Optuna

A Python implementation that combines PCA, LASSO, and Optuna-based feature selection with comprehensive visualization capabilities.

## Overview

This project provides a `FeatureSelector` class that implements:
- Feature selection using multiple methods (PCA, LASSO, Optuna)
- Automated data preprocessing and encoding
- Interaction-aware feature importance scoring
- Comprehensive visualization suite

## Implementation Details

### Feature Selection Methods

1. **PCA Baseline**
   - Uses `sklearn.decomposition.PCA`
   - Fixed 2 components (`n_components=2`)
   - Returns transformed data and component importance

2. **LASSO Baseline**
   - Uses `sklearn.linear_model.Lasso`
   - Alpha value of 0.01
   - Uses `SelectFromModel` for feature selection
   - Returns selected features and coefficients

3. **Optuna Selection**
   - Uses TPE sampler with median pruning
   - 50 optimization trials
   - Dynamic feature masking
   - Early pruning configuration:
     - 5 startup trials
     - 10 warmup steps
     - 1 interval step

### Interaction-Aware Feature Importance

Calculates feature importance using the formula:
```python
I(f_i) = α⋅MI(f_i; y) + β⋅∑(j≠i) I_interaction(f_i, f_j)
```
where:
- α = 0.7 (mutual information weight)
- β = 0.3 (interaction weight)
- MI(f_i; y) = mutual information score between feature i and target y
- I_interaction(f_i, f_j) = Spearman correlation between features i and j

### Supported Datasets

1. **Iris Dataset**
   ```python
   columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
   target = 'class'
   path = "data/Iris Dataset/bezdekIris.data"
   ```

2. **Adult Dataset**
   ```python
   columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 
             'native-country', 'income']
   target = 'income'
   path = "data/Adult Dataset/adult.data"
   ```

### Preprocessing

The implementation automatically:
1. Handles categorical features using `LabelEncoder`
2. Scales numerical features using `StandardScaler`
3. Encodes categorical target variables
4. Suppresses warnings for:
   - UserWarning from sklearn.metrics
   - DataConversionWarning

### Visualization Suite

Creates a 2x2 plot (15x12 inches) showing:
1. Method Comparison
   - Bar plot of mutual information scores
   - X-axis: methods
   - Y-axis: MI score

2. Runtime Analysis
   - Bar plot of execution times
   - X-axis: methods
   - Y-axis: seconds

3. Feature Importance Distribution
   - Heatmap using 'YlOrRd' colormap
   - Rows: methods
   - Columns: features

4. Feature Interaction Heatmap
   - Heatmap using 'coolwarm' colormap
   - Shows pairwise Spearman correlations
   - Symmetric matrix of feature interactions

## Dependencies

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mutual_info_score
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
```

## Usage

```python
# Initialize selector
selector = FeatureSelector("Iris")  # or "Adult"

# Load and preprocess data
selector.load_data()

# Run all methods
results = selector.run_all_methods()

# Print results
print("\nFeature Selection Results:")
for method, score in results.items():
    print(f"{method}: {score:.4f}")
```

## Output Format

The `run_all_methods()` function returns a dictionary with:
- Keys: 'PCA', 'LASSO', 'Optuna'
- Values: Mean mutual information scores for selected features

Example output:
```python
{
    'PCA': 0.8245,
    'LASSO': 0.9132,
    'Optuna': 1.0025
}
```

## License

MIT License

## Citation

```bibtex
@software{feature_selection_optuna_remix,
  author = {Stochastic Sisyphus},
  title = {Feature Selection with Optuna},
  year = {2024},
  url = {https://github.com/stochastic-sisyphus/feature-selection-optuna-remix}
}
```