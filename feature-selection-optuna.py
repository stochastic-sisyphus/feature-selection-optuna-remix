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
from scipy.stats import spearmanr
from sklearn.datasets import load_iris, fetch_openml

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')
warnings.filterwarnings('ignore', category=DataConversionWarning)

class FeatureSelector:
    """Feature selection implementation with multiple methods and visualizations"""
    
    def __init__(self, dataset_name="iris", alpha=0.7, beta=0.3):
        """
        Initialize with weighting parameters for interaction-aware importance
        
        Args:
            dataset_name (str): Name of the dataset to analyze ('iris' or 'adult')
            alpha (float): Weight for mutual information score (default: 0.7)
            beta (float): Weight for feature interaction score (default: 0.3)
        """
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.beta = beta
        self.results = {}
        self.runtimes = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load data from scikit-learn datasets"""
        if self.dataset_name.lower() == "iris":
            data = load_iris()
            self.X = pd.DataFrame(data.data, columns=data.feature_names)
            self.y = pd.Series(data.target)
        else:  # adult dataset
            data = fetch_openml(name='adult', version=2, as_frame=True)
            self.X = data.data
            self.y = data.target
            
        self.feature_names = self.X.columns.tolist()
        self.X_scaled, self.y = self.preprocess_data(self.X, self.y)
        
    def preprocess_data(self, X, y):
        """Preprocess the data by scaling features and encoding labels"""
        # Handle categorical features
        X_processed = X.copy()
        for column in X_processed.columns:
            if X_processed[column].dtype == 'object':
                X_processed[column] = LabelEncoder().fit_transform(X_processed[column].astype(str))
        
        # Scale features and handle target
        X_scaled = StandardScaler().fit_transform(X_processed)
        y_processed = LabelEncoder().fit_transform(y.astype(str)) if y.dtype == 'object' else y
        return X_scaled, np.ravel(y_processed)
    
    def calculate_interaction_importance(self, feature_idx):
        """Calculate interaction importance between features using Spearman correlation"""
        interactions = []
        for j in range(self.X_scaled.shape[1]):
            if j != feature_idx:
                correlation = abs(spearmanr(
                    self.X_scaled[:, feature_idx], 
                    self.X_scaled[:, j]
                )[0])
                interactions.append(correlation)
        return np.mean(interactions) if interactions else 0
    
    def calculate_feature_importance(self, feature_idx):
        """
        Calculate interaction-aware importance:
        I(f_i) = α⋅MI(f_i; y) + β⋅∑(j≠i) I_interaction(f_i, f_j)
        """
        mi_score = mutual_info_score(self.y, self.X_scaled[:, feature_idx])
        interaction_score = self.calculate_interaction_importance(feature_idx)
        return self.alpha * mi_score + self.beta * interaction_score

    def run_all_methods(self):
        """Run all feature selection methods and collect results"""
        self.load_data()
        
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
        lasso.fit(self.X_scaled, self.y)
        X_lasso = SelectFromModel(lasso, prefit=True).transform(self.X_scaled)
        return X_lasso, lasso.coef_
    
    def optuna_selection(self, n_trials=50):
        """Enhanced Optuna-based feature selection with TPE sampler"""
        def objective(trial):
            # Dynamic feature masking with pruning
            n_features = self.X_scaled.shape[1]
            feature_mask = [
                trial.suggest_int(f"feature_{i}", 0, 1) 
                for i in range(n_features)
            ]
            
            if sum(feature_mask) == 0:
                return float('-inf')
            
            selected_features = [i for i, mask in enumerate(feature_mask) if mask]
            X_selected = self.X_scaled[:, selected_features]
            importance_scores = [
                self.calculate_feature_importance(i) 
                for i in selected_features
            ]
            
            mean_score = np.mean(importance_scores)
            
            if trial.number > 0:
                try:
                    if mean_score < trial.study.best_value:
                        raise optuna.TrialPruned()
                except ValueError:
                    pass
            
            return mean_score

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        try:
            study.optimize(objective, n_trials=n_trials)
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return self.X_scaled, np.zeros(self.X_scaled.shape[1])
        
        best_params = study.best_params
        feature_mask = [
            best_params.get(f"feature_{i}", 0)
            for i in range(self.X_scaled.shape[1])
        ]
        selected_features = [i for i, mask in enumerate(feature_mask) if mask]
        
        if not selected_features:
            selected_features = [0]
        
        importance = np.zeros(self.X_scaled.shape[1])
        for idx in selected_features:
            importance[idx] = self.calculate_feature_importance(idx)
        
        return self.X_scaled[:, selected_features], importance

    def calculate_mi_scores(self, X):
        """Calculate mutual information scores for each feature"""
        return [mutual_info_score(self.y, X[:, i]) for i in range(X.shape[1])]

    def visualize_all_results(self):
        """Create comprehensive visualizations of all results"""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        methods = list(self.results.keys())
        
        # Method Comparison
        ax1.bar(methods, list(self.results.values()))
        ax1.set_title('Method Comparison')
        ax1.set_ylabel('Mutual Information Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Runtime Comparison
        ax2.bar(methods, list(self.runtimes.values()))
        ax2.set_title('Runtime Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Feature Importance Heatmap
        importance_matrix = self._prepare_importance_matrix()
        if importance_matrix is not None:
            sns.heatmap(
                importance_matrix, 
                ax=ax3,
                xticklabels=self.feature_names,
                yticklabels=methods,
                cmap='YlOrRd'
            )
            ax3.set_title('Feature Importance Distribution')
        
        # Feature Interaction Heatmap
        interaction_matrix = self._calculate_interaction_matrix()
        sns.heatmap(
            interaction_matrix,
            ax=ax4,
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            cmap='coolwarm'
        )
        ax4.set_title('Feature Interaction Heatmap')
        
        plt.tight_layout()
        plt.show()

    def _calculate_interaction_matrix(self):
        """Calculate pairwise feature interactions"""
        n_features = len(self.feature_names)
        matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    correlation = abs(spearmanr(
                        self.X_scaled[:, i],
                        self.X_scaled[:, j]
                    )[0])
                    matrix[i, j] = correlation
        
        return matrix

    def _prepare_importance_matrix(self):
        """Prepare feature importance matrix for visualization"""
        if not self.feature_importance:
            return None
            
        n_methods = len(self.feature_importance)
        n_features = len(self.feature_names)
        matrix = np.zeros((n_methods, n_features))
        
        for i, (method, importance) in enumerate(self.feature_importance.items()):
            if isinstance(importance, (list, np.ndarray)):
                if len(np.array(importance).shape) > 1:
                    importance = np.mean(importance, axis=0)
                if len(importance) != n_features:
                    temp = np.zeros(n_features)
                    temp[:len(importance)] = importance[:n_features]
                    importance = temp
            else:
                importance = np.array([importance] * n_features)
            
            matrix[i] = importance
            
        return matrix

def main():
    datasets = ["iris", "adult"]
    results = {}
    
    for dataset in datasets:
        print(f"\nProcessing {dataset} dataset...")
        selector = FeatureSelector(dataset)
        results[dataset] = selector.run_all_methods()
    
    print("\nFinal Results:")
    for dataset, scores in results.items():
        print(f"\n{dataset} Dataset:")
        for method, score in scores.items():
            print(f"{method}: {score:.4f}")

if __name__ == "__main__":
    main()
