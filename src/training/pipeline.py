"""
Training Pipeline Module
Manages complete QGAN training process with ethical monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class QGANTrainingPipeline:
    """Complete QGAN training pipeline with ethical monitoring"""
    
    def __init__(self, model, trainer, config, device='cpu'):
        self.model = model
        self.trainer = trainer
        self.config = config
        self.device = device
        
        # Training state
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 10)
        
        # Monitoring
        self.history = {
            'train_losses': [],
            'discriminator_losses': [],
            'val_losses': [],
            'ethical_scores': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Ethical compliance tracking
        self.ethical_violations = []
        
    def train(self, train_loader, test_loader, epochs=100):
        """Execute complete training pipeline"""
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = datetime.now()
            
            # Train one epoch
            epoch_metrics = self.trainer.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self._validate(test_loader)
            
            # Check ethical compliance
            ethical_check = self._check_ethical_compliance(epoch)
            
            # Record metrics
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            self._record_metrics(epoch, epoch_metrics, val_metrics, ethical_check, epoch_time)
            
            # Check for improvement
            current_val_loss = val_metrics.get('val_loss', float('inf'))
            improved = current_val_loss < self.best_loss
            
            if improved:
                self.best_loss = current_val_loss
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                
                # Save best model
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.early_stopping_counter += 1
            
            # Save regular checkpoint
            if epoch % self.config['training'].get('checkpoint_frequency', 10) == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Print progress
            self._print_epoch_progress(epoch, epoch_metrics, val_metrics, improved)
            
            # Check early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                self.history['early_stopped'] = True
                break
        
        # Finalize training
        return self._finalize_training()
    
    def _validate(self, test_loader):
        """Validate model on test set"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data[0].to(self.device)
                batch_size = batch_data.size(0)
                
                # Generate synthetic data
                z = torch.randn(batch_size, 1, device=self.device)
                generated_data = self.model.generate_ethical_samples(batch_size, self.device)
                
                # Calculate discriminator loss on real data
                real_authenticity, real_ethical = self.model.discriminator(batch_data)
                real_loss = nn.BCELoss()(real_authenticity, torch.ones_like(real_authenticity))
                
                # Calculate discriminator loss on fake data
                fake_authenticity, fake_ethical = self.model.discriminator(generated_data)
                fake_loss = nn.BCELoss()(fake_authenticity, torch.zeros_like(fake_authenticity))
                
                val_loss = (real_loss + fake_loss) / 2
                val_losses.append(val_loss.item())
        
        return {
            'val_loss': np.mean(val_losses) if val_losses else float('inf')
        }
    
    def _check_ethical_compliance(self, epoch):
        """Check ethical compliance for current epoch"""
        compliance_report = {}
        
        # Generate samples for ethical checking
        with torch.no_grad():
            test_samples = self.model.generate_ethical_samples(1000, self.device)
            ethical_report = self.model.get_ethical_report(test_samples)
        
        # Check key ethical constraints
        compliance_report['within_bounds'] = ethical_report.get('within_bounds', False)
        compliance_report['no_pii_detected'] = not ethical_report.get('pii_pattern_detected', False)
        compliance_report['no_proprietary_detected'] = not ethical_report.get('proprietary_pattern_detected', False)
        
        # Calculate ethical score (0-1)
        ethical_score = 0.0
        if compliance_report['within_bounds']:
            ethical_score += 0.4
        if compliance_report['no_pii_detected']:
            ethical_score += 0.3
        if compliance_report['no_proprietary_detected']:
            ethical_score += 0.3
        
        compliance_report['ethical_score'] = ethical_score
        
        # Record violations
        if ethical_score < 0.7: # Threshold
            violation = {
                'epoch': epoch,
                'ethical_score': ethical_score,
                'report': ethical_report
            }
            self.ethical_violations.append(violation)
        
        return compliance_report
    
    def _record_metrics(self, epoch, train_metrics, val_metrics, ethical_check, epoch_time):
        """Record all metrics for the epoch"""
        self.history['train_losses'].append(train_metrics.get('g_loss', 0))
        self.history['discriminator_losses'].append(train_metrics.get('d_loss', 0))
        self.history['val_losses'].append(val_metrics.get('val_loss', 0))
        self.history['ethical_scores'].append(ethical_check.get('ethical_score', 0))
        self.history['epoch_times'].append(epoch_time)
        
        # Record learning rate if available
        if hasattr(self.trainer, 'g_optimizer'):
            lr = self.trainer.g_optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(lr)
    
    def _print_epoch_progress(self, epoch, train_metrics, val_metrics, improved):
        """Print progress for the current epoch"""
        marker = "*" if improved else " "
        
        print(f"Epoch {epoch:3d} {marker} | "
              f"G Loss: {train_metrics.get('g_loss', 0):.4f} | "
              f"D Loss: {train_metrics.get('d_loss', 0):.4f} | "
              f"Val Loss: {val_metrics.get('val_loss', 0):.4f} | "
              f"Ethical: {self.history['ethical_scores'][-1]:.3f} | "
              f"Best: {self.best_loss:.4f}@{self.best_epoch}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['logging'].get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'config': self.config,
            'ethical_violations': self.ethical_violations
        }
        
        if is_best:
            filename = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, filename)
        else:
            filename = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, filename)
    
    def _finalize_training(self):
        """Finalize training and prepare results"""
        # Ensure history has all required keys
        if 'early_stopped' not in self.history:
            self.history['early_stopped'] = False
        
        # Create comprehensive results dictionary
        results = {
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'final_epoch': len(self.history['train_losses']),
            'early_stopped': self.history['early_stopped'],
            'train_losses': self.history['train_losses'],
            'discriminator_losses': self.history['discriminator_losses'],
            'val_losses': self.history['val_losses'],
            'ethical_scores': self.history['ethical_scores'],
            'epoch_times': self.history['epoch_times'],
            'ethical_violations_count': len(self.ethical_violations),
            'average_epoch_time': np.mean(self.history['epoch_times']) if self.history['epoch_times'] else 0,
            'total_training_time': np.sum(self.history['epoch_times']) if self.history['epoch_times'] else 0
        }
        
        # Save final training report
        self._save_training_report(results)
        
        return results
    
    def _save_training_report(self, results):
        """Save comprehensive training report"""
        report_dir = Path("../results/training")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'training_summary': {
                'start_time': datetime.now().isoformat(),
                'total_epochs': results['final_epoch'],
                'best_epoch': results['best_epoch'],
                'best_loss': float(results['best_loss']),
                'early_stopped': results['early_stopped'],
                'total_training_time': results['total_training_time'],
                'average_epoch_time': results['average_epoch_time']
            },
            'performance_metrics': {
                'final_train_loss': results['train_losses'][-1] if results['train_losses'] else 0,
                'final_val_loss': results['val_losses'][-1] if results['val_losses'] else 0,
                'min_train_loss': min(results['train_losses']) if results['train_losses'] else 0,
                'min_val_loss': min(results['val_losses']) if results['val_losses'] else 0
            },
            'ethical_compliance': {
                'average_ethical_score': np.mean(results['ethical_scores']) if results['ethical_scores'] else 0,
                'min_ethical_score': min(results['ethical_scores']) if results['ethical_scores'] else 0,
                'ethical_violations': results['ethical_violations_count'],
                'compliance_rate': 1 - (results['ethical_violations_count'] / results['final_epoch']) if results['final_epoch'] > 0 else 1
            },
            'config_used': self.config,
            'ethical_violations_details': self.ethical_violations
        }
        
        # Save as JSON
        report_path = report_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✓ Training report saved to: {report_path}")
        
        # Save metrics as CSV for easier analysis
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(results['train_losses']) + 1),
            'train_loss': results['train_losses'],
            'discriminator_loss': results['discriminator_losses'],
            'val_loss': results['val_losses'],
            'ethical_score': results['ethical_scores'],
            'epoch_time': results['epoch_times']
        })
        
        csv_path = report_dir / "training_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        print(f"✓ Training metrics saved to: {csv_path}")
