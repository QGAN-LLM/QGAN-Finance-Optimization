"""Central configuration management for the research project."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path

@dataclass
class DataConfig:
    """Data acquisition and preprocessing configuration."""
    currency_pair: str = "EURUSD"
    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    data_source: str = "dukascopy"  # or "yfinance", "trader_made"
    granularity: str = "1min"  # 1min, 5min, 1H, 1D
    
    # Feature engineering
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma_20", "sma_50", "rsi", "macd", "bollinger_upper", 
        "bollinger_lower", "atr", "stochastic_k"
    ])
    volatility_regimes: List[str] = field(default_factory=lambda: [
        "low", "medium", "high"
    ])
    vix_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 15.0,
        "medium": 30.0,
        "high": 45.0
    })
    
    # Preprocessing
    train_split: float = 0.7  # 2010-2021
    val_split: float = 0.15   # 2022-2023
    test_split: float = 0.15  # 2024-2025

@dataclass
class QuantumConfig:
    """Quantum circuit and QGAN configuration."""
    n_qubits: int = 4
    circuit_depth: int = 3
    entanglement_type: str = "linear"  # "linear", "full", "circular"
    encoding_type: str = "angle"  # "angle", "amplitude", "basis"
    
    # Noise injection
    noise_model: Optional[str] = "depolarizing"  # None, "depolarizing", "amplitude_damping"
    noise_probability: float = 0.01
    
    # QGAN training
    generator_lr: float = 0.001
    discriminator_lr: float = 0.0002
    batch_size: int = 32
    n_epochs: int = 100
    
    # Hardware/Simulator
    backend: str = "default.qubit"  # PennyLane device
    shots: Optional[int] = None  # None for analytic

@dataclass
class ClassicalConfig:
    """Classical model configuration."""
    discriminator_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    discriminator_dropout: float = 0.2
    
    # LSTM Baseline
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    
    # LLM Integration
    llm_model: str = "gpt-3.5-turbo"  # Will upgrade to GPT-5 when available
    fine_tuning: bool = True
    max_tokens: int = 500

@dataclass
class AttackConfig:
    """Adversarial attack configuration."""
    attack_types: List[str] = field(default_factory=lambda: ["fgsm", "pgd"])
    epsilon: float = 0.1  # Perturbation magnitude for FGSM
    pgd_steps: int = 10
    pgd_step_size: float = 0.01
    
    # Threat scenarios from FIN-ATT&CK
    threat_scenarios: List[Dict] = field(default_factory=lambda: [
        {
            "id": "TA02.001",
            "name": "Order Book Spoofing",
            "description": "Inject fake orders to manipulate price",
            "simulation_params": {...}
        },
        {
            "id": "TA07.003", 
            "name": "Sentiment Poisoning",
            "description": "Manipulate NLP sentiment analysis",
            "simulation_params": {...}
        }
    ])

@dataclass
class EvaluationConfig:
    """Evaluation metrics and thresholds."""
    # FINSEC-QBENCH metrics
    metrics: List[str] = field(default_factory=lambda: [
        "rmse", "mae", "asr", "fpr", "latency", "fid", "cqr", "talis"
    ])
    
    # Statistical significance
    alpha: float = 0.05
    n_bootstrap: int = 1000
    
    # Quantum metrics
    tomography_method: str = "mle"  # Maximum Likelihood Estimation
    n_tomography_shots: int = 10000

@dataclass
class ExperimentConfig:
    """Main configuration container."""
    seed: int = 42
    project_name: str = "qgan_llm_cybersecurity"
    
    data: DataConfig = field(default_factory=DataConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent / "configs" / "default.yaml"
        self.config = self.load_config()
    
    def load_config(self) -> ExperimentConfig:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return self.dict_to_config(config_dict)
        else:
            return ExperimentConfig()
    
    def save_config(self, config: ExperimentConfig, path: Optional[Path] = None):
        """Save configuration to YAML file."""
        save_path = path or self.config_path
        config_dict = self.config_to_dict(config)
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def config_to_dict(self, config: ExperimentConfig) -> Dict:
        """Convert config object to dictionary."""
        # Implement conversion logic
        pass
    
    def dict_to_config(self, config_dict: Dict) -> ExperimentConfig:
        """Convert dictionary to config object."""
        # Implement conversion logic  
        pass