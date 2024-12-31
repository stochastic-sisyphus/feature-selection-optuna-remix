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
from itertools import combinations
import time
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')
warnings.filterwarnings('ignore', category=DataConversionWarning)

class FeatureSelector:
    """Feature selection implementation with multiple methods and visualizations"""
    
    def __init__(self, X, y, feature_names=None, dataset_name="Dataset"):
        self.dataset_name = dataset_name
        self.X = X
        self.y = y
        self.feature_names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(X.shape[1])]
        self.results = {}
        self.runtimes = {}
        self.feature_importance = {}
        
    def preprocess_data(self):
        """Preprocess the data by scaling features and encoding labels"""
        # Handle categorical features if X is a DataFrame
        X_processed = self.X.copy() if isinstance(self.X, pd.DataFrame) else self.X
        if isinstance(X_processed, pd.DataFrame):
            for column in X_processed.columns:
                if X_processed[column].dtype == 'object':
                    X_processed[column] = LabelEncoder().fit_transform(X_processed[column].astype(str))
        
        # Scale features and handle target
        self.X_scaled = StandardScaler().fit_transform(X_processed)
        self.y_processed = LabelEncoder().fit_transform(self.y.astype(str)) if self.y.dtype == 'object' else self.y
        self.y_processed = np.ravel(self.y_processed)
    
    def run_all_methods(self):
        """Run all feature selection methods and collect results"""
        self.preprocess_data()
        
        methods = {
            'PCA': self.pca_baseline,
            'LASSO': self.lasso_baseline,
            'Optuna': self.optuna_selection
        }
        
        for name, method in methods.items():
            start_time = time.time()
            X_selected, importance = method()
            self.results[name] = np.mean(self.calculate_mi_scores(X_selected))
            self.runtimes[name] = time.time() - start_time
            self.feature_importance[name] = importance
            
        self.visualize_all_results()
        return self.results
    
    def pca_baseline(self, n_components=2):
        """PCA-based feature selection"""
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.X_scaled)
        return X_pca, pca.components_
    
    def lasso_baseline(self, alpha=0.01):
        """LASSO-based feature selection"""
        lasso = Lasso(alpha=alpha)
        lasso.fit(self.X_scaled, self.y_processed)
        X_lasso = SelectFromModel(lasso, prefit=True).transform(self.X_scaled)
        return X_lasso, lasso.coef_
    
    def optuna_selection(self, n_trials=50):
        """Optuna-based feature selection"""
        n_features_range = range(1, min(5, self.X_scaled.shape[1] + 1))
        feature_combinations = [
            ','.join(map(str, comb))
            for n in n_features_range
            for comb in combinations(range(self.X_scaled.shape[1]), n)
        ]
        
        def objective(trial):
            selected_str = trial.suggest_categorical("selected_features", feature_combinations)
            selected_features = tuple(map(int, selected_str.split(',')))
            X_selected = self.X_scaled[:, list(selected_features)]
            return np.mean(self.calculate_mi_scores(X_selected))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        best_features = tuple(map(int, study.best_params["selected_features"].split(',')))
        return self.X_scaled[:, list(best_features)], best_features
    
    def calculate_mi_scores(self, X):
        """Calculate mutual information scores for each feature"""
        return [mutual_info_score(self.y_processed, X[:, i]) for i in range(X.shape[1])]
    
    def visualize_all_results(self):
        """Create comprehensive visualizations of all results"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        methods = list(self.results.keys())
        
        # Method Comparison
        ax1.bar(methods, list(self.results.values()))
        ax1.set_title(f'Method Comparison - {self.dataset_name}')
        ax1.set_ylabel('Mutual Information Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Runtime Comparison
        ax2.bar(methods, list(self.runtimes.values()))
        ax2.set_title('Runtime Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Feature Importance Heatmap
        importance_data = []
        for method in methods:
            if isinstance(self.feature_importance[method], np.ndarray):
                feat_importance = self.feature_importance[method]
                if len(feat_importance.shape) == 1:
                    feat_importance = feat_importance.reshape(1, -1)
                n_features = len(self.feature_names)
                if feat_importance.shape[1] > n_features:
                    feat_importance = feat_importance[:, :n_features]
                elif feat_importance.shape[1] < n_features:
                    padding = np.zeros((feat_importance.shape[0], n_features - feat_importance.shape[1]))
                    feat_importance = np.hstack([feat_importance, padding])
                importance_data.append(feat_importance)
        
        if importance_data:
            importance_matrix = np.vstack(importance_data)
            sns.heatmap(importance_matrix, ax=ax3,
                       xticklabels=self.feature_names,
                       yticklabels=methods,
                       cmap='YlOrRd')
            ax3.set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()

def example_usage():
    """Example usage of the FeatureSelector class"""
    # Generate sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Initialize and run feature selection
    selector = FeatureSelector(X, y, feature_names, "Example Dataset")
    results = selector.run_all_methods()
    
    # Print results
    print("\nFeature Selection Results:")
    for method, score in results.items():
        print(f"{method}: {score:.4f}")

if __name__ == "__main__":
    example_usage()
