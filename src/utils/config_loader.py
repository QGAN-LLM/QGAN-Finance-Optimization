"""
Configuration Loader
Load and validate configuration files
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ConfigLoader:
    """Load and manage configuration files"""
    
    @staticmethod
    def load_yaml_config(file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def load_json_config(file_path: str) -> Dict[str, Any]:
        """Load JSON configuration file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any], config_type: str) -> bool:
        """Validate configuration for required fields"""
        validators = {
            'data': ConfigLoader._validate_data_config,
            'training': ConfigLoader._validate_training_config,
            'model': ConfigLoader._validate_model_config,
            'quantum': ConfigLoader._validate_quantum_config
        }
        
        if config_type not in validators:
            print(f"Warning: No validator for config type: {config_type}")
            return True
        
        return validators[config_type](config)
    
    @staticmethod
    def _validate_data_config(config: Dict[str, Any]) -> bool:
        """Validate data configuration"""
        required_fields = ['data_sources', 'feature_engineering']
        
        for field in required_fields:
            if field not in config:
                print(f"Error: Missing required field in data config: {field}")
                return False
        
        # Check ethical constraints
        if 'ethical_constraints' in config:
            constraints = config['ethical_constraints']
            if 'excluded_data' not in constraints:
                print("Warning: No excluded_data in ethical_constraints")
        
        return True
    
    @staticmethod
    def _validate_training_config(config: Dict[str, Any]) -> bool:
        """Validate training configuration"""
        required_fields = ['epochs', 'batch_size', 'learning_rate']
        
        for field in required_fields:
            if field not in config.get('training', {}):
                print(f"Error: Missing required field in training config: {field}")
                return False
        
        return True
    
    @staticmethod
    def _validate_model_config(config: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        required_fields = ['baseline_models', 'qgan']
        
        for field in required_fields:
            if field not in config:
                print(f"Error: Missing required field in model config: {field}")
                return False
        
        return True
    
    @staticmethod
    def _validate_quantum_config(config: Dict[str, Any]) -> bool:
        """Validate quantum configuration"""
        required_fields = ['n_qubits', 'n_layers']
        
        for field in required_fields:
            if field not in config.get('quantum', {}):
                print(f"Error: Missing required field in quantum config: {field}")
                return False
        
        return True
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations, with override taking precedence"""
        import copy
        
        merged = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                # Overwrite with new value
                merged[key] = value
        
        return merged
    
    @staticmethod
    def save_config(config: Dict[str, Any], file_path: str, format: str = 'yaml'):
        """Save configuration to file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'yaml':
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif format == 'json':
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Configuration saved to: {file_path}")

