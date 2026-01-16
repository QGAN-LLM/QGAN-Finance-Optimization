"""
Visualization tools for research results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """Creates visualizations for research results."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = save_dir or Path("figures")
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, 
                             history: Dict[str, List[float]],
                             title: str = "Training History") -> plt.Figure:
        """Plot training history metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        metrics = [
            ('g_loss', 'Generator Loss'),
            ('d_loss', 'Discriminator Loss'),
            ('d_real_acc', 'Discriminator Real Accuracy'),
            ('d_fake_acc', 'Discriminator Fake Accuracy')
        ]
        
        for idx, (metric_key, metric_name) in enumerate(metrics):
            if metric_key in history:
                ax = axes[idx]
                ax.plot(history[metric_key])
                ax.set_title(metric_name)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / f"training_history_{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
        
        return fig
    
    def plot_adversarial_results(self,
                                attack_results: Dict[str, Dict[str, Any]],
                                title: str = "Adversarial Attack Results") -> plt.Figure:
        """Visualize adversarial attack results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. Attack Success Rates
        ax = axes[0]
        attack_names = []
        success_rates = []
        
        for attack_name, results in attack_results.items():
            if 'success_rate' in results:
                attack_names.append(attack_name.upper())
                success_rates.append(results['success_rate'])
        
        if attack_names:
            bars = ax.bar(attack_names, success_rates)
            ax.set_title('Attack Success Rates')
            ax.set_ylabel('Success Rate')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        # 2. Prediction Changes
        ax = axes[1]
        sample_attack = next(iter(attack_results.values()))
        if 'original_predictions' in sample_attack and 'adversarial_predictions' in sample_attack:
            n_samples = min(50, len(sample_attack['original_predictions']))
            
            original_preds = sample_attack['original_predictions'][:n_samples].flatten()
            adversarial_preds = sample_attack['adversarial_predictions'][:n_samples].flatten()
            
            x = np.arange(n_samples)
            width = 0.35
            
            ax.bar(x - width/2, original_preds, width, label='Original', alpha=0.7)
            ax.bar(x + width/2, adversarial_preds, width, label='Adversarial', alpha=0.7)
            ax.set_title('Prediction Changes')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Prediction')
            ax.legend()
        
        # 3. Perturbation Magnitude
        ax = axes[2]
        if 'perturbation' in sample_attack:
            perturbations = sample_attack['perturbation']
            if perturbations.ndim > 1:
                perturbation_norms = np.linalg.norm(perturbations, axis=1)
            else:
                perturbation_norms = np.abs(perturbations)
            
            ax.hist(perturbation_norms, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title('Perturbation Magnitude Distribution')
            ax.set_xlabel('Perturbation Norm')
            ax.set_ylabel('Frequency')
            ax.axvline(np.mean(perturbation_norms), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(perturbation_norms):.4f}')
            ax.legend()
        
        # 4. Feature-wise Perturbation
        ax = axes[3]
        if 'perturbation' in sample_attack and sample_attack['perturbation'].ndim > 1:
            feature_perturbations = np.mean(np.abs(sample_attack['perturbation']), axis=0)
            n_features = len(feature_perturbations)
            
            ax.bar(range(n_features), feature_perturbations)
            ax.set_title('Average Perturbation per Feature')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Average |Perturbation|')
            ax.set_xticks(range(n_features))
        
        # 5. Success Rate vs Epsilon (if multiple epsilons)
        ax = axes[4]
        epsilons = []
        rates = []
        
        for attack_name, results in attack_results.items():
            if 'epsilon' in results and 'success_rate' in results:
                epsilons.append(results['epsilon'])
                rates.append(results['success_rate'])
        
        if len(set(epsilons)) > 1:
            ax.scatter(epsilons, rates, s=100, alpha=0.7)
            ax.set_title('Success Rate vs Perturbation Magnitude')
            ax.set_xlabel('Epsilon')
            ax.set_ylabel('Success Rate')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(epsilons) > 2:
                z = np.polyfit(epsilons, rates, 1)
                p = np.poly1d(z)
                ax.plot(sorted(epsilons), p(sorted(epsilons)), 'r--', alpha=0.7)
        
        # 6. Confusion Matrix (for classification)
        ax = axes[5]
        # Placeholder for classification results
        ax.text(0.5, 0.5, 'Classification Metrics\n(If applicable)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Classification Performance')
        ax.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / f"adversarial_results_{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved adversarial results plot to {save_path}")
        
        return fig
    
    def plot_quantum_analysis(self,
                            quantum_metrics: Dict[str, Any],
                            title: str = "Quantum State Analysis") -> plt.Figure:
        """Visualize quantum state analysis results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. State Fidelity over iterations
        ax = axes[0]
        if 'fidelity_history' in quantum_metrics:
            fidelity_history = quantum_metrics['fidelity_history']
            ax.plot(fidelity_history)
            ax.set_title('State Fidelity Evolution')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Fidelity')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # 2. Entanglement Entropy
        ax = axes[1]
        if 'entanglement_entropy' in quantum_metrics:
            entropy = quantum_metrics['entanglement_entropy']
            if isinstance(entropy, (list, np.ndarray)):
                ax.bar(range(len(entropy)), entropy)
                ax.set_title('Entanglement Entropy by Subsystem')
                ax.set_xlabel('Subsystem')
                ax.set_ylabel('Entropy')
            else:
                ax.text(0.5, 0.5, f'Entanglement Entropy: {entropy:.4f}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Entanglement Entropy')
                ax.axis('off')
        
        # 3. State Purity
        ax = axes[2]
        if 'purity' in quantum_metrics:
            purity = quantum_metrics['purity']
            ax.bar(['Purity'], [purity])
            ax.set_title('State Purity')
            ax.set_ylabel('Purity')
            ax.set_ylim(0, 1)
            ax.text(0, purity, f'{purity:.4f}', ha='center', va='bottom')
        
        # 4. Density Matrix Heatmap
        ax = axes[3]
        if 'density_matrix' in quantum_metrics:
            density_matrix = quantum_metrics['density_matrix']
            im = ax.imshow(np.abs(density_matrix), cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title('Density Matrix (Magnitude)')
            ax.set_xlabel('Basis State')
            ax.set_ylabel('Basis State')
        
        # 5. Measurement Probabilities
        ax = axes[4]
        if 'measurement_probabilities' in quantum_metrics:
            probs = quantum_metrics['measurement_probabilities']
            n_states = len(probs)
            ax.bar(range(n_states), probs)
            ax.set_title('Measurement Probabilities')
            ax.set_xlabel('Basis State')
            ax.set_ylabel('Probability')
            ax.set_xticks(range(n_states))
        
        # 6. Quantum Volume/Coherence
        ax = axes[5]
        quantum_vol = quantum_metrics.get('quantum_volume', 0)
        coherence = quantum_metrics.get('coherence_estimate', 0)
        
        metrics = ['Quantum Volume', 'Coherence']
        values = [quantum_vol, coherence]
        
        bars = ax.bar(metrics, values)
        ax.set_title('Quantum Performance Metrics')
        ax.set_ylabel('Value')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / f"quantum_analysis_{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved quantum analysis plot to {save_path}")
        
        return fig
    
    def plot_comparative_analysis(self,
                                 results_qgan: Dict[str, Any],
                                 results_baseline: Dict[str, Any],
                                 title: str = "QGAN vs Baseline Comparison") -> plt.Figure:
        """Create comparative analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Metrics to compare
        comparison_metrics = [
            ('rmse', 'RMSE', 'Lower is better'),
            ('mae', 'MAE', 'Lower is better'),
            ('asr', 'Attack Success Rate', 'Lower is better'),
            ('fid', 'Fréchet Distance', 'Lower is better'),
            ('cqr', 'CQR Score', 'Higher is better'),
            ('ari', 'ARI', 'Higher is better')
        ]
        
        for idx, (metric_key, metric_name, direction) in enumerate(comparison_metrics[:6]):
            ax = axes[idx]
            
            # Extract metric values
            qgan_value = self._extract_metric(results_qgan, metric_key)
            baseline_value = self._extract_metric(results_baseline, metric_key)
            
            if qgan_value is not None and baseline_value is not None:
                models = ['Baseline', 'QGAN']
                values = [baseline_value, qgan_value]
                colors = ['skyblue', 'lightcoral']
                
                bars = ax.bar(models, values, color=colors, alpha=0.8)
                ax.set_title(f'{metric_name}\n({direction})')
                ax.set_ylabel(metric_name)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')
                
                # Highlight better performance
                if direction == 'Lower is better':
                    better_idx = np.argmin(values)
                else:
                    better_idx = np.argmax(values)
                
                bars[better_idx].set_edgecolor('green')
                bars[better_idx].set_linewidth(3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / f"comparative_analysis_{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparative analysis plot to {save_path}")
        
        return fig
    
    def _extract_metric(self, results: Dict[str, Any], metric_key: str) -> Optional[float]:
        """Extract metric value from results dictionary."""
        # Try direct key
        if metric_key in results:
            value = results[metric_key]
            if isinstance(value, (int, float, np.number)):
                return float(value)
        
        # Try keys starting with metric
        for key, value in results.items():
            if key.startswith(metric_key) and isinstance(value, (int, float, np.number)):
                return float(value)
        
        # Try to find any metric containing the key
        for key, value in results.items():
            if metric_key in key and isinstance(value, (int, float, np.number)):
                return float(value)
        
        return None
    
    def create_dashboard(self,
                        all_results: Dict[str, Dict[str, Any]],
                        save_path: Optional[Path] = None) -> Path:
        """Create comprehensive results dashboard."""
        from matplotlib.backends.backend_pdf import PdfPages
        
        if save_path is None:
            save_path = self.save_dir / "results_dashboard.pdf"
        
        with PdfPages(save_path) as pdf:
            # 1. Title page
            fig = plt.figure(figsize=(8.5, 11))
            plt.suptitle("QGAN-LLM Cybersecurity Research\nResults Dashboard", 
                        fontsize=20, y=0.85)
            plt.figtext(0.5, 0.6, 
                       f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                       f"Total Experiments: {len(all_results)}",
                       ha='center', fontsize=14)
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 2. Summary statistics
            fig = self._create_summary_page(all_results)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 3. Individual experiment results
            for exp_name, results in all_results.items():
                if 'training_history' in results:
                    fig = self.plot_training_history(
                        results['training_history'],
                        title=f"{exp_name} - Training History"
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                
                if 'adversarial_results' in results:
                    fig = self.plot_adversarial_results(
                        results['adversarial_results'],
                        title=f"{exp_name} - Adversarial Results"
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                
                if 'quantum_metrics' in results:
                    fig = self.plot_quantum_analysis(
                        results['quantum_metrics'],
                        title=f"{exp_name} - Quantum Analysis"
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        logger.info(f"Saved results dashboard to {save_path}")
        return save_path
    
    def _create_summary_page(self, all_results: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """Create summary statistics page."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        axes = axes.flatten()
        
        # Prepare summary data
        summary_data = []
        
        for exp_name, results in all_results.items():
            row = {'Experiment': exp_name}
            
            # Extract key metrics
            metrics_to_extract = ['rmse', 'asr', 'fid', 'cqr']
            for metric in metrics_to_extract:
                value = self._extract_metric(results, metric)
                row[metric.upper()] = value if value is not None else np.nan
            
            summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # 1. Metric comparison table
            ax = axes[0]
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.round(4).values,
                           colLabels=df.columns,
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            ax.set_title('Experiment Results Summary')
            
            # 2. Best performers
            ax = axes[1]
            best_performers = []
            
            for metric in ['RMSE', 'ASR', 'FID']:
                if metric in df.columns and not df[metric].isna().all():
                    best_idx = df[metric].idxmin()
                    best_value = df.loc[best_idx, metric]
                    best_name = df.loc[best_idx, 'Experiment']
                    best_performers.append(f"{metric}: {best_name} ({best_value:.4f})")
            
            for metric in ['CQR']:
                if metric in df.columns and not df[metric].isna().all():
                    best_idx = df[metric].idxmax()
                    best_value = df.loc[best_idx, metric]
                    best_name = df.loc[best_idx, 'Experiment']
                    best_performers.append(f"{metric}: {best_name} ({best_value:.4f})")
            
            ax.text(0.1, 0.9, "Best Performers:", fontsize=12, fontweight='bold')
            for i, performer in enumerate(best_performers):
                ax.text(0.1, 0.8 - i*0.1, performer, fontsize=10)
            ax.axis('off')
            ax.set_title('Top Performers by Metric')
            
            # 3. Metric distributions
            ax = axes[2]
            metrics_to_plot = [col for col in df.columns if col != 'Experiment']
            if metrics_to_plot:
                df[metrics_to_plot].boxplot(ax=ax)
                ax.set_title('Metric Distributions')
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)
            
            # 4. Correlation heatmap
            ax = axes[3]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax)
                
                # Add correlation values
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
                
                ax.set_xticks(range(len(corr_matrix)))
                ax.set_yticks(range(len(corr_matrix)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45)
                ax.set_yticklabels(corr_matrix.columns)
                ax.set_title('Metric Correlations')
        
        plt.suptitle('Results Summary Dashboard', fontsize=16)
        plt.tight_layout()
        
        return fig