"""
Classical LSTM Baseline for EUR/USD Price Forecasting.

This module implements the standalone Classical LSTM baseline used as a
control condition in the QGAN-LLM dissertation study. It represents
traditional time-series forecasting without GAN augmentation or LLM integration.

Purpose:
    - Establish baseline forecasting accuracy (expected RMSE ~0.523)
    - Provide comparison point for QGAN-LLM framework
    - Serve as ablation reference for isolating quantum contributions

Architecture:
    - Input: 32 PCA components (or raw features)
    - LSTM: 2 layers, 128 hidden units
    - Output: 1 value (next-day price or return)
    
Reference:
    Chi, D. T. K., Kien, H. N. T., & Nguyen, T. Q. (2025). Enhancing forex
    market forecasting with feature-augmented multivariate LSTM models using
    real-time data. Knowledge-Based Systems, 330, 114500.
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class LSTMConfig:
    """Configuration for the Classical LSTM baseline."""
    
    # Architecture
    input_size: int = 32
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 1
    dropout: float = 0.2
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Sequence
    sequence_length: int = 50
    prediction_horizon: int = 1
    
    # Reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    checkpoint_dir: str = "models/classical_lstm/"
    log_dir: str = "results/classical_lstm/"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "LSTMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})
    
    @classmethod
    def from_json(cls, path: str) -> "LSTMConfig":
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# REPRODUCIBILITY UTILITIES
# ============================================================================

def set_global_seed(seed: int) -> None:
    """Set random seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LSTMForecaster(nn.Module):
    """
    LSTM-based forecaster for financial time series.
    
    Architecture:
        Input (batch, seq_len, input_size)
            ↓
        LSTM Layer 1 (hidden_size=128, return_sequences=True)
            ↓
        Dropout (p=0.2)
            ↓
        LSTM Layer 2 (hidden_size=128, return_sequences=False)
            ↓
        Dropout (p=0.2)
            ↓
        Fully Connected (hidden_size → output_size)
            ↓
        Output (batch, output_size)
    """
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Xavier initialization for stable training."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch, output_size)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Dropout + FC
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out


# ============================================================================
# DATASET
# ============================================================================

class FinancialTimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset for financial time series with sliding window sequences.
    
    Given a time series of features X and targets y, creates overlapping
    windows of length `sequence_length` to predict the target at the next
    timestep (or `prediction_horizon` steps ahead).
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 50,
        prediction_horizon: int = 1
    ):
        """
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            sequence_length: Length of input window
            prediction_horizon: Steps ahead to predict
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Validate
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have same length. Got {len(self.X)} and {len(self.y)}"
            )
        
        # Compute number of valid windows
        self.n_windows = max(0, len(X) - sequence_length - prediction_horizon + 1)
        
        if self.n_windows == 0:
            raise ValueError(
                f"Not enough samples ({len(X)}) for sequence_length "
                f"({sequence_length}) + prediction_horizon ({prediction_horizon})"
            )
    
    def __len__(self) -> int:
        return self.n_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (x_seq, y_target) where:
                x_seq: (sequence_length, n_features)
                y_target: (1,) - target at prediction_horizon
        """
        x_seq = self.X[idx : idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length + self.prediction_horizon - 1]
        
        return x_seq, y_target.unsqueeze(0)


# ============================================================================
# TRAINER
# ============================================================================

class LSTMTrainer:
    """
    Trainer for the Classical LSTM baseline.
    
    Handles:
        - Training loop with early stopping
        - Validation monitoring
        - Checkpoint saving
        - Metric logging
    """
    
    def __init__(self, model: LSTMForecaster, config: LSTMConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Loss & optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': []
        }
        
        # Best checkpoint tracking
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.patience_counter = 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history dict
        """
        # Create checkpoint dir
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        for epoch in range(self.config.epochs):
            # ---- Training ----
            train_loss = self._train_epoch(train_loader)
            
            # ---- Validation ----
            val_loss = self._validate(val_loader)
            
            # ---- LR scheduler ----
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # ---- Log history ----
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"Train: {train_loss:.6f} | "
                    f"Val: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # ---- Early stopping ----
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Best epoch: {self.best_epoch+1} (val_loss={self.best_val_loss:.6f})"
                    )
                    break
        
        # Load best checkpoint
        self._load_best_checkpoint()
        logger.info(f"Training complete. Best val_loss: {self.best_val_loss:.6f}")
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, path)
    
    def _load_best_checkpoint(self) -> None:
        """Load best checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, 'best.pt')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']+1}")
    
    def save_history(self, path: str) -> None:
        """Save training history as JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


# ============================================================================
# EVALUATOR
# ============================================================================

class LSTMEvaluator:
    """
    Evaluator for the Classical LSTM baseline.
    
    Computes:
        - RMSE (primary metric)
        - MAE
        - MAPE
        - R²
        - Directional accuracy
    """
    
    def __init__(self, model: LSTMForecaster, config: LSTMConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input sequences.
        
        Args:
            X: Input array of shape (n_samples, sequence_length, n_features)
               OR (n_samples, n_features) which will be reshaped.
               
        Returns:
            Predictions of shape (n_samples,)
        """
        self.model.eval()
        
        # Handle 2D input by creating sequences
        if X.ndim == 2:
            X = self._make_sequences(X)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        predictions = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            preds = self.model(batch)
            predictions.append(preds.cpu().numpy())
        
        return np.vstack(predictions).flatten()
    
    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert 2D array to 3D sequences."""
        seq_len = self.config.sequence_length
        n_samples = len(X) - seq_len + 1
        
        if n_samples <= 0:
            raise ValueError(
                f"Input has {len(X)} samples, need at least {seq_len}"
            )
        
        sequences = np.stack([
            X[i:i+seq_len] for i in range(n_samples)
        ])
        
        return sequences
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict of metrics
        """
        predictions = self.predict(X_test)
        
        # Align y_test with predictions (account for sequence offset)
        if X_test.ndim == 2:
            y_test_aligned = y_test[self.config.sequence_length - 1:]
        else:
            y_test_aligned = y_test
        
        # Ensure same length
        min_len = min(len(predictions), len(y_test_aligned))
        predictions = predictions[:min_len]
        y_test_aligned = y_test_aligned[:min_len]
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, y_test_aligned)
        metrics['n_samples'] = len(predictions)
        metrics['model_type'] = 'Classical LSTM'
        
        return metrics
    
    def _compute_metrics(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Compute all regression metrics."""
        residuals = y_pred - y_true
        
        # MSE / RMSE
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        
        # MAE
        mae = float(np.mean(np.abs(residuals)))
        
        # MAPE (avoid division by zero)
        nonzero_mask = np.abs(y_true) > 1e-8
        if nonzero_mask.sum() > 0:
            mape = float(
                np.mean(np.abs(residuals[nonzero_mask] / y_true[nonzero_mask])) * 100
            )
        else:
            mape = float('nan')
        
        # R²
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1 - ss_res / (ss_tot + 1e-10))
        
        # Directional accuracy
        if len(y_true) > 1:
            true_dir = np.sign(np.diff(y_true))
            pred_dir = np.sign(np.diff(y_pred))
            dir_acc = float(np.mean(true_dir == pred_dir) * 100)
        else:
            dir_acc = float('nan')
        
        return {
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': dir_acc,
        }


# ============================================================================
# MAIN BASELINE CLASS
# ============================================================================

class ClassicalLSTMBaseline:
    """
    High-level interface for the Classical LSTM baseline.
    
    Usage:
        baseline = ClassicalLSTMBaseline(config)
        baseline.fit(X_train, y_train, X_val, y_val)
        results = baseline.evaluate(X_test, y_test)
        baseline.save_results("results/classical_lstm/")
    """
    
    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
        config_path: Optional[str] = None,
        name: str = "Classical LSTM"
    ):
        if config_path is not None:
            self.config = LSTMConfig.from_json(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = LSTMConfig()
        
        self.name = name
        
        # Set seed for reproducibility
        set_global_seed(self.config.seed)
        
        # Build model
        self.model = LSTMForecaster(self.config)
        self.trainer = LSTMTrainer(self.model, self.config)
        self.evaluator = LSTMEvaluator(self.model, self.config)
        
        # Results storage
        self.history: Dict = {}
        self.results: Dict = {}
        self.is_fitted: bool = False
        
        # Setup directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.name}")
        logger.info(f"Config: {self.config.to_dict()}")
    
    def _build_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """Build DataLoaders for training and validation."""
        train_dataset = FinancialTimeSeriesDataset(
            X_train, y_train,
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon
        )
        val_dataset = FinancialTimeSeriesDataset(
            X_val, y_val,
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.config.device == 'cuda'),
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.config.device == 'cuda')
        )
        
        logger.info(f"Train windows: {len(train_dataset)}")
        logger.info(f"Val windows: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM baseline.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training history
        """
        logger.info(f"Fitting {self.name}...")
        
        # Reshape targets to 1D if needed
        y_train = np.asarray(y_train).flatten()
        y_val = np.asarray(y_val).flatten()
        
        # Build loaders
        train_loader, val_loader = self._build_loaders(
            X_train, y_train, X_val, y_val
        )
        
        # Train
        self.history = self.trainer.train(train_loader, val_loader)
        
        # Save history
        self.trainer.save_history(
            os.path.join(self.config.log_dir, 'training_history.json')
        )
        
        self.is_fitted = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.evaluator.predict(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict of evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")
        
        logger.info(f"Evaluating {self.name}...")
        
        y_test = np.asarray(y_test).flatten()
        self.results = self.evaluator.evaluate(X_test, y_test)
        
        # Log results
        logger.info(f"{self.name} results:")
        for metric, value in self.results.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.6f}")
            else:
                logger.info(f"  {metric}: {value}")
        
        return self.results
    
    def save_results(self, output_dir: str) -> None:
        """Save results and config to JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(output_dir, 'config.json'))
        
        # Save results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def load(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        self.is_fitted = True
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            'total_trainable': total,
            'lstm': sum(p.numel() for p in self.model.lstm.parameters()),
            'fc': sum(p.numel() for p in self.model.fc.parameters()),
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_classical_lstm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Optional[LSTMConfig] = None,
    output_dir: str = "results/classical_lstm/"
) -> Dict[str, Any]:
    """
    Convenience function to run the full Classical LSTM baseline pipeline.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        config: Optional LSTMConfig
        output_dir: Where to save results
        
    Returns:
        Dict containing results, history, and model info
    """
    # Initialize
    baseline = ClassicalLSTMBaseline(config=config)
    
    # Train
    history = baseline.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    results = baseline.evaluate(X_test, y_test)
    
    # Save
    baseline.save_results(output_dir)
    
    return {
        'results': results,
        'history': history,
        'param_count': baseline.count_parameters(),
        'config': baseline.config.to_dict(),
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Classical LSTM baseline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/",
        help="Directory containing preprocessed numpy arrays"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/classical_lstm/",
        help="Directory to save results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="LSTM hidden size"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Configure
    config = LSTMConfig(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    
    # Run
    output = run_classical_lstm_baseline(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        config=config,
        output_dir=args.output_dir,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLASSICAL LSTM BASELINE — FINAL RESULTS")
    print("=" * 60)
    print(f"RMSE:                 {output['results']['rmse']:.6f}")
    print(f"MAE:                  {output['results']['mae']:.6f}")
    print(f"MAPE:                 {output['results']['mape']:.4f}%")
    print(f"R²:                   {output['results']['r2']:.6f}")
    print(f"Directional Accuracy: {output['results']['directional_accuracy']:.2f}%")
    print(f"Parameters:           {output['param_count']['total_trainable']:,}")
    print("=" * 60)
