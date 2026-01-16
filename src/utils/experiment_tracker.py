"""Experiment tracking for reproducibility."""
import mlflow
import wandb
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pickle
import hashlib
import numpy as np

class ExperimentTracker:
    """Tracks experiments, parameters, and results."""
    
    def __init__(self, experiment_name: str, config: Dict, use_mlflow: bool = True,
                 use_wandb: bool = True, log_dir: Optional[Path] = None):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
            use_mlflow: Whether to log to MLFlow
            use_wandb: Whether to log to Weights & Biases
            log_dir: Local directory for logging
        """
        self.experiment_name = experiment_name
        self.config = config
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.log_dir = log_dir or Path("experiment_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Generate unique experiment ID
        self.experiment_id = self._generate_experiment_id()
        
        # Initialize trackers
        self._init_trackers()
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID from config."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _init_trackers(self):
        """Initialize tracking services."""
        # MLFlow
        if self.use_mlflow:
            mlflow.set_experiment(f"{self.experiment_name}_{self.experiment_id}")
            self.mlflow_run = mlflow.start_run(run_name=self.experiment_id)
            
            # Log config
            mlflow.log_params(self._flatten_dict(self.config))
        
        # Weights & Biases
        if self.use_wandb:
            wandb.init(
                project="qgan-llm-cybersecurity",
                name=f"{self.experiment_name}_{self.experiment_id}",
                config=self.config
            )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all tracking services."""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        # Also save locally
        metrics_file = self.log_dir / f"metrics_step_{step or 'final'}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def log_artifact(self, file_path: Path, artifact_type: str = "model"):
        """Log artifact (model, plot, data) to trackers."""
        if self.use_mlflow:
            mlflow.log_artifact(str(file_path))
        
        if self.use_wandb:
            wandb.save(str(file_path))
    
    def log_figure(self, figure, name: str):
        """Log matplotlib figure."""
        if self.use_mlflow:
            mlflow.log_figure(figure, f"{name}.png")
        
        if self.use_wandb:
            wandb.log({name: wandb.Image(figure)})
    
    def log_quantum_state(self, state: np.ndarray, name: str):
        """Log quantum state for tomography analysis."""
        state_file = self.log_dir / f"{name}_quantum_state.npy"
        np.save(state_file, state)
        self.log_artifact(state_file, "quantum_state")
    
    def save_checkpoint(self, model_dict: Dict, filename: str):
        """Save model checkpoint."""
        checkpoint_file = self.log_dir / filename
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(model_dict, f)
        
        self.log_artifact(checkpoint_file, "checkpoint")
        return checkpoint_file
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def close(self):
        """Close all tracking connections."""
        if self.use_mlflow:
            mlflow.end_run()
        
        if self.use_wandb:
            wandb.finish()