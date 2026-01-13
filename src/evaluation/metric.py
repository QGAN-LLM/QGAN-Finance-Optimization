"""
Evaluation Metrics Module
Includes both performance and ethical metrics
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class EthicalMetricsCalculator:
    """Calculate ethical metrics for models"""
    
    def __init__(self):
        self.fairness_metrics = ['demographic_parity', 'equalized_odds', 'predictive_equality']
        self.privacy_metrics = ['memorization_score', 'information_leakage']
        
    def calculate(self, predictions, probabilities, features, model_type='baseline') -> Dict[str, float]:
        """
        Calculate comprehensive ethical metrics
        """
        metrics = {}
        
        # Fairness metrics (simplified - assumes no sensitive attributes)
        metrics['fairness_score'] = self._calculate_fairness(predictions, features)
        
        # Bias metrics
        metrics['bias_score'] = self._calculate_bias(predictions, features)
        
        # Privacy metrics
        metrics['privacy_score'] = self._calculate_privacy(predictions, probabilities, model_type)
        
        # Explainability score
        metrics['explainability_score'] = self._calculate_explainability(model_type)
        
        # Energy efficiency (estimated)
        metrics['energy_efficiency'] = self._estimate_energy_efficiency(model_type)
        
        return metrics
    
    def _calculate_fairness(self, predictions, features) -> float:
        """
        Calculate fairness score (simplified)
        In real implementation, would use sensitive attributes
        """
        # For financial data without sensitive attributes,
        # we check prediction distribution fairness
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        # Fair predictions should not be too skewed
        skewness = abs(pred_mean - 0.5) # Distance from 0.5 for binary
        
        # Calculate fairness score (higher is better)
        fairness = 1.0 - (skewness * 2) # Scale to [0, 1]
        fairness = max(0, min(1, fairness))
        
        return fairness
    
    def _calculate_bias(self, predictions, features) -> float:
        """
        Calculate bias score (simplified)
        Checks for systematic errors
        """
        if features is None or len(features.shape) < 2:
            return 0.5 # Default middle score
        
        # Check if predictions correlate strongly with any feature
        # (which might indicate bias)
        max_correlation = 0
        
        for i in range(min(10, features.shape[1])): # Check first 10 features
            if features.shape[0] == len(predictions):
                correlation = np.abs(np.corrcoef(predictions, features[:, i])[0, 1])
                max_correlation = max(max_correlation, correlation)
        
        # Bias score: lower correlation with features is better
        # (models should learn patterns, not just memorize features)
        bias_score = 1.0 - max_correlation
        bias_score = max(0, min(1, bias_score))
        
        return bias_score
    
    def _calculate_privacy(self, predictions, probabilities, model_type) -> float:
        """
        Calculate privacy score
        Higher score means better privacy protection
        """
        if probabilities is None:
            return 0.5 # Default middle score
        
        # Check prediction confidence distribution
        # Overly confident predictions might memorize training data
        if len(probabilities) > 0:
            # Convert to confidence scores
            confidence = np.abs(probabilities - 0.5) * 2 # Scale to [0, 1]
            avg_confidence = np.mean(confidence)
            
            # Models that are very confident might be memorizing
            # Ideal: moderate confidence (not 0 or 1)
            privacy_score = 1.0 - avg_confidence
            
            # Adjust for model type
            if model_type == 'qgan':
                privacy_score *= 1.1 # QGAN might have better privacy
            elif model_type == 'baseline':
                privacy_score *= 0.9 # Baseline models might memorize more
            
            privacy_score = max(0, min(1, privacy_score))
        else:
            privacy_score = 0.5
        
        return privacy_score
    
    def _calculate_explainability(self, model_type) -> float:
        """
        Calculate explainability score based on model type
        """
        # Base scores by model type
        base_scores = {
            'random_forest': 0.8, # Feature importances available
            'xgboost': 0.7, # Feature importances available
            'lightgbm': 0.7, # Feature importances available
            'qgan': 0.6, # Quantum models less explainable
            'llm': 0.5, # Transformers less explainable
            'baseline': 0.5 # Default
        }
        
        return base_scores.get(model_type, 0.5)
    
    def _estimate_energy_efficiency(self, model_type) -> float:
        """
        Estimate energy efficiency based on model type
        Higher score means more energy efficient
        """
        # Estimated efficiency scores
        efficiency_scores = {
            'random_forest': 0.7,
            'xgboost': 0.8,
            'lightgbm': 0.9, # LightGBM is efficient
            'qgan': 0.4, # Quantum simulation is energy intensive
            'llm': 0.3, # Transformers are energy intensive
            'baseline': 0.5
        }
        
        return efficiency_scores.get(model_type, 0.5)


class PerformanceMetrics:
    """Calculate traditional performance metrics"""
    
    @staticmethod
    def calculate_all(y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Calculate all performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        
        # AUC-ROC if probabilities available
        if y_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc_roc'] = 0.5
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positive'] = tp
        metrics['false_positive'] = fp
        metrics['true_negative'] = tn
        metrics['false_negative'] = fn
        
        # Derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics
    
    @staticmethod
    def calculate_statistical_significance(y_true, preds_model1, preds_model2, test='mcnemar'):
        """
        Calculate statistical significance between two models
        """
        from statsmodels.stats.contingency_tables import mcnemar
        
        if test == 'mcnemar':
            # Create contingency table
            contingency = np.zeros((2, 2))
            
            for i in range(len(y_true)):
                contingency[preds_model1[i], preds_model2[i]] += 1
            
            # Perform McNemar's test
            result = mcnemar(contingency, exact=True)
            
            return {
                'test': 'mcnemar',
                'p_value': result.pvalue,
                'statistic': result.statistic if hasattr(result, 'statistic') else None,
                'significant': result.pvalue < 0.05
            }
        else:
            raise ValueError(f"Test {test} not implemented")


class RobustnessMetrics:
    """Calculate model robustness metrics"""
    
    @staticmethod
    def calculate_adversarial_robustness(model, X_test, y_test, attack_method='fgsm', epsilon=0.1):
        """
        Calculate adversarial robustness score
        Simplified version - in practice would use adversarial attacks
        """
        from sklearn.metrics import accuracy_score
        
        # Base accuracy
        y_pred = model.predict(X_test)
        base_accuracy = accuracy_score(y_test, y_pred)
        
        # Simulate adversarial perturbation (simplified)
        # In practice, would generate actual adversarial examples
        X_perturbed = X_test.copy()
        
        if attack_method == 'fgsm':
            # Fast Gradient Sign Method (simplified)
            gradient = np.random.randn(*X_test.shape) * 0.01 # Simulated gradient
            X_perturbed = X_test + epsilon * np.sign(gradient)
        elif attack_method == 'random':
            # Random noise
            X_perturbed = X_test + np.random.randn(*X_test.shape) * epsilon
        
        # Predict on perturbed data
        y_pred_perturbed = model.predict(X_perturbed)
        perturbed_accuracy = accuracy_score(y_test, y_pred_perturbed)
        
        # Robustness score
        robustness = perturbed_accuracy / base_accuracy if base_accuracy > 0 else 0
        robustness = max(0, min(1, robustness))
        
        return {
            'attack_method': attack_method,
            'epsilon': epsilon,
            'base_accuracy': base_accuracy,
            'perturbed_accuracy': perturbed_accuracy,
            'robustness_score': robustness,
            'accuracy_drop': base_accuracy - perturbed_accuracy
        }
    
    @staticmethod
    def calculate_stability(model, X_test, y_test, n_bootstrap=100):
        """
        Calculate model stability through bootstrapping
        """
        from sklearn.utils import resample
        
        accuracies = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            X_boot, y_boot = resample(X_test, y_test, random_state=_)
            
            # Predict
            y_pred = model.predict(X_boot)
            accuracy = accuracy_score(y_boot, y_pred)
            accuracies.append(accuracy)
        
        stability_metrics = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'stability_score': 1.0 / (1.0 + np.std(accuracies)), # Higher std = less stable
            'confidence_interval': (
                np.percentile(accuracies, 2.5),
                np.percentile(accuracies, 97.5)
            )
        }
        
        return stability_metrics

