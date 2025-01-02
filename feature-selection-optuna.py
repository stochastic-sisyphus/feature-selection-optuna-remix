"""
Feature Selection with Optuna

This script implements feature selection using Optuna optimization framework,
with comparison to PCA and LASSO baselines. It includes visualization of results
and performance metrics.

Features:
- Optuna-based feature selection with convergence monitoring
- Dynamic weight adjustment for mutual information and feature interaction
- Comparison with PCA and LASSO baselines
- Comprehensive visualization of results
- Support for multiple datasets

Author: Stochastic Sisyphus
"""

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
from itertools import combinations, product
import time
import warnings
from sklearn.exceptions import DataConversionWarning
from scipy.stats import spearmanr
import os
from contextlib import suppress

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')
warnings.filterwarnings('ignore', category=DataConversionWarning)

class FeatureSelector:
    """Feature selection implementation with multiple methods and visualizations"""
    
    def __init__(self, dataset_name, alpha=0.7, beta=0.3):
        """
        Initialize the FeatureSelector.
        
        Args:
            dataset_name (str): Name of the dataset to analyze
            alpha (float): Initial weight for mutual information score (default: 0.7)
            beta (float): Initial weight for interaction score (default: 0.3)
        """
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.beta = beta
        self.results = {}
        self.runtimes = {}
        self.importances = {}  # Store feature importances
        self.load_data()
        
    def load_data(self):
        """
        Load and preprocess data. Override this method to load your own dataset.
        Example implementation for the Iris dataset is provided.
        """
        # Example using iris dataset
        from sklearn.datasets import load_iris
        iris = load_iris()
        self.X = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.y = iris.target
        
        print(f"\nLoaded {self.dataset_name} dataset:")
        print(f"Shape: {self.X.shape}")
        print("Features:", ", ".join(self.X.columns))
        
        # Extract features and preprocess
        self.feature_names = self.X.columns.tolist()
        self.X_scaled, self.y = self.preprocess_data(self.X, self.y)
        
        print(f"Preprocessed {len(self.feature_names)} features")
        print(f"Target classes: {len(np.unique(self.y))}")
    
    def preprocess_data(self, X, y):
        """
        Preprocess the data by scaling features and encoding labels.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (array-like): Target variable
            
        Returns:
            tuple: (scaled features, encoded labels)
        """
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
        """
        Calculate interaction importance between features using Spearman correlation.
        
        Args:
            feature_idx (int): Index of the feature to analyze
            
        Returns:
            float: Mean interaction importance
        """
        interactions = []
        for j in range(self.X_scaled.shape[1]):
            if j != feature_idx:
                correlation = abs(spearmanr(
                    self.X_scaled[:, feature_idx], 
                    self.X_scaled[:, j]
                )[0])
                interactions.append(correlation)
        
        if not interactions:
            return 0
        return np.mean(interactions)

    def calculate_feature_importance(self, feature_idx):
        """
        Calculate interaction-aware importance for a feature.
        
        Args:
            feature_idx (int): Index of the feature to analyze
            
        Returns:
            float: Combined importance score
        """
        mi_score = mutual_info_score(self.y, self.X_scaled[:, feature_idx])
        interaction_score = self.calculate_interaction_importance(feature_idx)
        return self.alpha * mi_score + self.beta * interaction_score

    def adjust_weights(self, mi_scores, spearman_scores):
        """
        Dynamically adjust alpha and beta weights based on score trends.
        
        Args:
            mi_scores (list): History of mutual information scores
            spearman_scores (list): History of Spearman correlation scores
        """
        if len(mi_scores) <= 1:
            return

        mi_trend = (mi_scores[-1] - mi_scores[-2]) / max(abs(mi_scores[-2]), 1e-5)
        spearman_trend = (spearman_scores[-1] - spearman_scores[-2]) / max(abs(spearman_scores[-2]), 1e-5)
        
        if mi_trend > spearman_trend:
            self.alpha += 0.1 * (1 - self.alpha)
            self.beta -= 0.1 * self.beta
        else:
            self.alpha -= 0.1 * self.alpha
            self.beta += 0.1 * (1 - self.beta)
        
        # Normalize weights
        total = self.alpha + self.beta
        self.alpha /= total
        self.beta /= total
    
    def optuna_selection(self, n_trials=50, min_improvement=1e-4, patience=8):
        """
        Perform feature selection using Optuna with convergence monitoring.
        
        Args:
            n_trials (int): Maximum number of trials (default: 50)
            min_improvement (float): Minimum improvement threshold (default: 1e-4)
            patience (int): Number of trials without improvement before stopping (default: 8)
            
        Returns:
            tuple: (selected features matrix, feature importance scores)
        """
        if 'optuna' in self.results:
            return (
                self.X_scaled[:, self.selected_features['optuna']],
                self.importances['optuna']
            )
            
        start_time = time.time()
        state = {
            'mi_scores': [],
            'spearman_scores': [],
            'best_score': -float("inf"),
            'no_improvement_count': 0,
            'min_trials': max(20, n_trials // 3),
            'best_features': None
        }
        
        def objective(trial):
            n_features = self.X_scaled.shape[1]
            feature_mask = [
                trial.suggest_int(f"feature_{i}", 0, 1)
                for i in range(n_features)
            ]
            
            if sum(feature_mask) < 2:
                return float('-inf')
            
            selected_features = [i for i, mask in enumerate(feature_mask) if mask]
            X_selected = self.X_scaled[:, selected_features]
            
            # Calculate scores
            mi_scores = [
                mutual_info_score(self.y, X_selected[:, i])
                for i in range(X_selected.shape[1])
            ]
            spearman_scores = []
            for i, j in combinations(range(len(selected_features)), 2):
                corr = abs(spearmanr(X_selected[:, i], X_selected[:, j])[0])
                spearman_scores.append(corr)
            
            mi_score = np.mean(mi_scores)
            spearman_score = np.mean(spearman_scores) if spearman_scores else 0
            
            state['mi_scores'].append(mi_score)
            state['spearman_scores'].append(spearman_score)
            
            if len(state['mi_scores']) >= 3:
                self.adjust_weights(state['mi_scores'], state['spearman_scores'])
            
            # Calculate final score with feature count penalty
            n_selected = len(selected_features)
            feature_penalty = 0.1 * (abs(n_selected - n_features/2) / (n_features/2))
            current_score = (
                (self.alpha * mi_score + self.beta * spearman_score) *
                (1 - feature_penalty)
            )
            
            # Check for improvement
            if current_score > state['best_score']:
                state['best_score'] = current_score
                state['best_features'] = selected_features
                state['no_improvement_count'] = 0
            else:
                state['no_improvement_count'] += 1
            
            # Convergence check
            if (trial.number >= state['min_trials'] and 
                state['no_improvement_count'] >= patience):
                print(f"\nConvergence reached after {trial.number + 1} trials:")
                print(f"Best score: {state['best_score']:.4f}")
                print(
                    f"Selected features: "
                    f"{[self.feature_names[i] for i in state['best_features']]}"
                )
                raise optuna.exceptions.TrialPruned("Convergence reached")
            
            return current_score

        # Setup and run optimization
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                catch=(optuna.exceptions.TrialPruned,)
            )
        except optuna.exceptions.TrialPruned as e:
            print(f"Optimization stopped: {str(e)}")
        
        runtime = time.time() - start_time
        print(f"\nTotal runtime: {runtime:.2f} seconds")
        print(f"Final weights - Alpha: {self.alpha:.3f}, Beta: {self.beta:.3f}")
        
        selected_features = state['best_features'] or [0, 1]
        importance = np.zeros(self.X_scaled.shape[1])
        for idx in selected_features:
            importance[idx] = self.calculate_feature_importance(idx)
        
        self.results['optuna'] = state['best_score']
        self.runtimes['optuna'] = runtime
        self.importances['optuna'] = importance
        self.selected_features = {'optuna': selected_features}
        
        return self.X_scaled[:, selected_features], importance
    
    def pca_baseline(self):
        """
        PCA-based feature selection baseline.
        
        Returns:
            tuple: (transformed data, feature importance scores)
        """
        if 'pca' in self.results:
            return (
                self.transformed_data['pca'],
                self.importances['pca']
            )
            
        start_time = time.time()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        runtime = time.time() - start_time
        
        score = np.mean([
            mutual_info_score(self.y, X_pca[:, i])
            for i in range(X_pca.shape[1])
        ])
        
        self.results['pca'] = score
        self.runtimes['pca'] = runtime
        self.importances['pca'] = np.mean(np.abs(pca.components_), axis=0)
        self.transformed_data = {'pca': X_pca}
        
        return X_pca, pca.components_
    
    def lasso_baseline(self):
        """
        LASSO-based feature selection baseline.
        
        Returns:
            tuple: (transformed data, feature importance scores)
        """
        if 'lasso' in self.results:
            return (
                self.transformed_data['lasso'],
                self.importances['lasso']
            )
            
        start_time = time.time()
        lasso = Lasso(alpha=0.01)
        lasso.fit(self.X_scaled, self.y)
        selector = SelectFromModel(lasso, prefit=True)
        X_lasso = selector.transform(self.X_scaled)
        runtime = time.time() - start_time
        
        score = np.mean([
            mutual_info_score(self.y, X_lasso[:, i])
            for i in range(X_lasso.shape[1])
        ])
        
        self.results['lasso'] = score
        self.runtimes['lasso'] = runtime
        self.importances['lasso'] = np.abs(lasso.coef_)
        self.transformed_data = {'lasso': X_lasso}
        
        return X_lasso, lasso.coef_
    
    def visualize_results(self):
        """Create comprehensive visualizations of feature selection results"""
        # Setup the plot grid
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)
        
        # 1. Method Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        methods = list(self.results.keys())
        scores = list(self.results.values())
        ax1.bar(methods, scores)
        ax1.set_title('Method Comparison')
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Runtime Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        runtimes = list(self.runtimes.values())
        ax2.bar(methods, runtimes)
        ax2.set_title('Runtime Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Feature Importance Heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        importance_matrix = np.zeros((len(methods), len(self.feature_names)))
        for i, method in enumerate(methods):
            importance_matrix[i] = self.importances[method]
        
        sns.heatmap(
            importance_matrix,
            ax=ax3,
            xticklabels=self.feature_names,
            yticklabels=methods,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f'
        )
        ax3.set_title('Feature Importance by Method')
        
        # 4. Feature Interaction Heatmap
        ax4 = fig.add_subplot(gs[1, 1])
        interaction_matrix = np.zeros((len(self.feature_names), len(self.feature_names)))
        for i, j in combinations(range(len(self.feature_names)), 2):
            corr = abs(spearmanr(self.X_scaled[:, i], self.X_scaled[:, j])[0])
            interaction_matrix[i, j] = corr
            interaction_matrix[j, i] = corr
        
        sns.heatmap(
            interaction_matrix,
            ax=ax4,
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            cmap='coolwarm',
            annot=True,
            fmt='.2f'
        )
        ax4.set_title('Feature Interaction Heatmap')
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = "results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"{output_dir}/{self.dataset_name}_analysis.png")
        plt.close()

def main():
    """Example usage with the Iris dataset"""
    selector = FeatureSelector("iris")
    
    # Run all methods
    selector.optuna_selection()
    selector.pca_baseline()
    selector.lasso_baseline()
    
    # Create visualizations
    selector.visualize_results()
    
    # Print summary
    print("\nResults Summary:")
    print("-" * 40)
    for method, score in selector.results.items():
        runtime = selector.runtimes[method]
        print(f"{method.upper()}:")
        print(f"Score: {score:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")
    print("-" * 40)

if __name__ == "__main__":
    main()
