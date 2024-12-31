# Feature Selection with Optuna

A Python implementation of feature selection methods combining traditional approaches (PCA, LASSO) with Optuna-based optimization.

## Overview

This project provides a comprehensive feature selection toolkit that combines classical dimensionality reduction techniques with modern hyperparameter optimization. It's particularly useful for:
- Automated feature selection in machine learning pipelines
- Comparing different feature selection methods
- Visualizing feature importance across methods
- Optimizing feature combinations using Optuna

## Features

- Multiple feature selection methods:
  - PCA (Principal Component Analysis)
  - LASSO (Least Absolute Shrinkage and Selection Operator)
  - Optuna-optimized feature selection
- Comprehensive visualization suite:
  - Method comparison plots
  - Runtime analysis
  - Feature importance heatmaps
- Automated preprocessing for both numerical and categorical data
- Support for custom datasets and feature names

## Installation

```bash
# Clone the repository
git clone https://github.com/stochastic-sisyphus/feature-selection-optuna-remix.git
cd feature-selection-optuna-remix

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
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

When running the feature selection on sample data, the optimization process shows:

```
[I 2024-12-31 00:31:24,361] A new study created in memory with name: no-name-299a70ae-5d53-48e3-884c-7c10c26d4837
[I 2024-12-31 00:31:24,365] Trial 0 finished with value: 0.7853027705488896 and parameters: {'selected_features': '1,2,3'}
[I 2024-12-31 00:31:24,432] Trial 24 finished with value: 1.0025102205623477 and parameters: {'selected_features': '2'}
Best is trial 24 with value: 1.0025102205623477
```

### Visualization Results

![Feature Selection Results](/assets/Figure_1.png)

The visualization shows:
1. Method comparison based on mutual information scores
2. Runtime comparison between different methods
3. Feature importance heatmap showing the relative importance of each feature across methods

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
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 0.24.2
- optuna >= 3.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- tqdm >= 4.65.0
- joblib >= 1.2.0
- plotly >= 5.13.0

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{feature_selection_optuna_remix,
  author = {Stochastic Sisyphus},
  title = {Feature Selection with Optuna},
  year = {2024},
  url = {https://github.com/stochastic-sisyphus/feature-selection-optuna-remix}
}
```

## Acknowledgments

- Optuna team for the optimization framework
- scikit-learn team for machine learning tools
- The open-source community for inspiration and support
