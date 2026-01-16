"""
Adversarial attack implementations for cybersecurity testing.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AdversarialAttacker:
    """Implements various adversarial attacks for testing model robustness."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adversarial attacker.
        
        Args:
            config: Attack configuration
        """
        self.config = config
        self.attack_methods = {
            'fgsm': self._fgsm_attack,
            'pgd': self._pgd_attack,
            'cw': self._cw_attack,
            'data_poisoning': self._data_poisoning_attack,
            'model_inversion': self._model_inversion_attack
        }
    
    def attack_model(self, 
                    model: nn.Module,
                    data: np.ndarray,
                    targets: Optional[np.ndarray] = None,
                    attack_type: str = 'fgsm',
                    **kwargs) -> Dict[str, Any]:
        """
        Perform adversarial attack on model.
        
        Args:
            model: Target model to attack
            data: Input data
            targets: True labels/targets (optional)
            attack_type: Type of attack to perform
            **kwargs: Additional attack parameters
            
        Returns:
            Dictionary with attack results
        """
        if attack_type not in self.attack_methods:
            raise ValueError(f"Unknown attack type: {attack_type}. "
                           f"Available: {list(self.attack_methods.keys())}")
        
        logger.info(f"Performing {attack_type.upper()} attack")
        
        # Get attack parameters
        attack_params = self._get_attack_params(attack_type, kwargs)
        
        # Perform attack
        attack_results = self.attack_methods[attack_type](
            model=model,
            data=data,
            targets=targets,
            **attack_params
        )
        
        # Calculate attack success rate
        if 'adversarial_data' in attack_results and targets is not None:
            success_rate = self._calculate_success_rate(
                model, 
                attack_results['adversarial_data'], 
                targets,
                attack_results.get('original_predictions')
            )
            attack_results['success_rate'] = success_rate
        
        logger.info(f"{attack_type.upper()} attack complete. "
                   f"Success rate: {attack_results.get('success_rate', 'N/A')}")
        
        return attack_results
    
    def _get_attack_params(self, 
                          attack_type: str,
                          user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get attack parameters with defaults."""
        defaults = {
            'fgsm': {
                'epsilon': self.config.get('epsilon', 0.1),
                'targeted': False
            },
            'pgd': {
                'epsilon': self.config.get('epsilon', 0.1),
                'alpha': self.config.get('pgd_step_size', 0.01),
                'iterations': self.config.get('pgd_steps', 10),
                'targeted': False
            },
            'cw': {
                'c': 1.0,
                'kappa': 0,
                'iterations': 100,
                'lr': 0.01
            },
            'data_poisoning': {
                'poison_ratio': 0.1,
                'poison_strength': 0.5
            },
            'model_inversion': {
                'iterations': 100,
                'lr': 0.1
            }
        }
        
        params = defaults.get(attack_type, {}).copy()
        params.update(user_params)
        
        return params
    
    def _fgsm_attack(self,
                    model: nn.Module,
                    data: np.ndarray,
                    targets: Optional[np.ndarray] = None,
                    epsilon: float = 0.1,
                    targeted: bool = False,
                    **kwargs) -> Dict[str, Any]:
        """Fast Gradient Sign Method attack."""
        device = next(model.parameters()).device
        
        # Convert to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        data_tensor.requires_grad = True
        
        if targets is not None:
            target_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
        else:
            # If no targets, use model predictions
            with torch.no_grad():
                target_tensor = model(data_tensor)
        
        # Forward pass
        outputs = model(data_tensor)
        
        # Calculate loss
        criterion = nn.MSELoss() if targets is not None else nn.L1Loss()
        loss = criterion(outputs, target_tensor)
        
        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()
        
        # Get gradient sign
        data_grad = data_tensor.grad.data
        
        # Create adversarial example
        if targeted:
            # For targeted attack, move away from target
            perturbed_data = data_tensor - epsilon * data_grad.sign()
        else:
            # For untargeted attack, move away from original prediction
            perturbed_data = data_tensor + epsilon * data_grad.sign()
        
        # Clip to valid range
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        # Get predictions on adversarial data
        with torch.no_grad():
            adv_outputs = model(perturbed_data)
        
        return {
            'adversarial_data': perturbed_data.cpu().numpy(),
            'original_data': data_tensor.detach().cpu().numpy(),
            'perturbation': (perturbed_data - data_tensor).cpu().numpy(),
            'original_predictions': outputs.detach().cpu().numpy(),
            'adversarial_predictions': adv_outputs.cpu().numpy(),
            'epsilon': epsilon,
            'targeted': targeted
        }
    
    def _pgd_attack(self,
                   model: nn.Module,
                   data: np.ndarray,
                   targets: Optional[np.ndarray] = None,
                   epsilon: float = 0.1,
                   alpha: float = 0.01,
                   iterations: int = 10,
                   targeted: bool = False,
                   **kwargs) -> Dict[str, Any]:
        """Projected Gradient Descent attack."""
        device = next(model.parameters()).device
        
        # Convert to tensor
        original_data = torch.tensor(data, dtype=torch.float32).to(device)
        perturbed_data = original_data.clone().detach()
        perturbed_data.requires_grad = True
        
        if targets is not None:
            target_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
        else:
            with torch.no_grad():
                target_tensor = model(original_data)
        
        criterion = nn.MSELoss() if targets is not None else nn.L1Loss()
        
        for i in range(iterations):
            # Forward pass
            outputs = model(perturbed_data)
            loss = criterion(outputs, target_tensor)
            
            # Backward pass
            model.zero_grad()
            if perturbed_data.grad is not None:
                perturbed_data.grad.zero_()
            loss.backward()
            
            # Update perturbed data
            with torch.no_grad():
                if targeted:
                    perturbed_data.data = perturbed_data.data - alpha * perturbed_data.grad.sign()
                else:
                    perturbed_data.data = perturbed_data.data + alpha * perturbed_data.grad.sign()
                
                # Project back to epsilon-ball
                perturbation = perturbed_data - original_data
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                perturbed_data.data = original_data + perturbation
                
                # Clip to valid range
                perturbed_data.data = torch.clamp(perturbed_data.data, 0, 1)
        
        # Get final predictions
        with torch.no_grad():
            adv_outputs = model(perturbed_data)
            original_outputs = model(original_data)
        
        return {
            'adversarial_data': perturbed_data.detach().cpu().numpy(),
            'original_data': original_data.cpu().numpy(),
            'perturbation': (perturbed_data - original_data).detach().cpu().numpy(),
            'original_predictions': original_outputs.cpu().numpy(),
            'adversarial_predictions': adv_outputs.cpu().numpy(),
            'epsilon': epsilon,
            'alpha': alpha,
            'iterations': iterations,
            'targeted': targeted
        }
    
    def _cw_attack(self,
                  model: nn.Module,
                  data: np.ndarray,
                  targets: Optional[np.ndarray] = None,
                  c: float = 1.0,
                  kappa: float = 0,
                  iterations: int = 100,
                  lr: float = 0.01,
                  **kwargs) -> Dict[str, Any]:
        """Carlini & Wagner attack (simplified version)."""
        # This is a simplified implementation
        # Full CW attack is more complex
        
        device = next(model.parameters()).device
        original_data = torch.tensor(data, dtype=torch.float32).to(device)
        
        # Use PGD as approximation for CW attack
        return self._pgd_attack(
            model=model,
            data=data,
            targets=targets,
            epsilon=0.1,
            alpha=lr,
            iterations=iterations,
            targeted=True  # CW is typically targeted
        )
    
    def _data_poisoning_attack(self,
                              model: nn.Module,
                              data: np.ndarray,
                              targets: Optional[np.ndarray] = None,
                              poison_ratio: float = 0.1,
                              poison_strength: float = 0.5,
                              **kwargs) -> Dict[str, Any]:
        """Data poisoning attack during training."""
        logger.info(f"Simulating data poisoning attack with {poison_ratio:.1%} poisoned data")
        
        n_samples = len(data)
        n_poison = int(n_samples * poison_ratio)
        
        # Select random samples to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        # Create poisoned data
        poisoned_data = data.copy()
        poisoned_targets = targets.copy() if targets is not None else None
        
        for idx in poison_indices:
            # Add noise/perturbation
            if poisoned_data.ndim > 1:
                noise = np.random.normal(0, poison_strength, poisoned_data[idx].shape)
            else:
                noise = np.random.normal(0, poison_strength)
            
            poisoned_data[idx] += noise
            
            # If targets exist, flip or perturb them
            if poisoned_targets is not None:
                if poisoned_targets.ndim > 1:
                    target_noise = np.random.normal(0, poison_strength, poisoned_targets[idx].shape)
                else:
                    target_noise = np.random.normal(0, poison_strength)
                
                poisoned_targets[idx] += target_noise
        
        # Train model on poisoned data (simplified)
        # In practice, you would retrain the model
        
        return {
            'poisoned_data': poisoned_data,
            'poisoned_targets': poisoned_targets,
            'poison_indices': poison_indices,
            'poison_ratio': poison_ratio,
            'poison_strength': poison_strength,
            'n_poisoned': n_poison
        }
    
    def _model_inversion_attack(self,
                               model: nn.Module,
                               data: np.ndarray,
                               targets: Optional[np.ndarray] = None,
                               iterations: int = 100,
                               lr: float = 0.1,
                               **kwargs) -> Dict[str, Any]:
        """Model inversion attack to reconstruct training data."""
        logger.info("Simulating model inversion attack")
        
        device = next(model.parameters()).device
        
        # Initialize random data
        n_samples = len(data)
        input_shape = data.shape[1:] if data.ndim > 1 else (1,)
        
        reconstructed = torch.randn(n_samples, *input_shape, device=device)
        reconstructed.requires_grad = True
        
        if targets is not None:
            target_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
        else:
            # Use model predictions as targets
            with torch.no_grad():
                target_tensor = model(torch.tensor(data, dtype=torch.float32).to(device))
        
        optimizer = torch.optim.Adam([reconstructed], lr=lr)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(reconstructed)
            
            # Loss: match model outputs
            loss = nn.MSELoss()(outputs, target_tensor)
            
            # Regularization to encourage realistic values
            reg_loss = torch.mean(reconstructed ** 2)
            total_loss = loss + 0.01 * reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                logger.debug(f"Inversion iteration {i}, loss: {loss.item():.4f}")
        
        reconstructed_np = reconstructed.detach().cpu().numpy()
        
        # Calculate reconstruction error
        if data.ndim == 1:
            recon_error = np.mean((data - reconstructed_np.flatten()) ** 2)
        else:
            recon_error = np.mean((data - reconstructed_np) ** 2)
        
        return {
            'reconstructed_data': reconstructed_np,
            'original_data': data,
            'reconstruction_error': recon_error,
            'iterations': iterations,
            'learning_rate': lr
        }
    
    def _calculate_success_rate(self,
                               model: nn.Module,
                               adversarial_data: np.ndarray,
                               targets: np.ndarray,
                               original_predictions: Optional[np.ndarray] = None) -> float:
        """Calculate attack success rate."""
        device = next(model.parameters()).device
        
        with torch.no_grad():
            adv_tensor = torch.tensor(adversarial_data, dtype=torch.float32).to(device)
            adv_predictions = model(adv_tensor).cpu().numpy()
        
        # For regression tasks, success = significant prediction change
        if original_predictions is not None:
            # Calculate prediction change
            prediction_change = np.abs(adv_predictions - original_predictions)
            
            # Threshold for "significant" change (e.g., > 10% of target range)
            target_range = np.ptp(targets) if len(targets) > 1 else 1.0
            threshold = 0.1 * target_range
            
            # Count successful attacks
            successful = prediction_change > threshold
            success_rate = np.mean(successful)
        else:
            # Simplified: use MSE increase
            original_mse = np.mean((targets - original_predictions) ** 2) if original_predictions is not None else 0
            adv_mse = np.mean((targets - adv_predictions) ** 2)
            
            # Success if MSE increased by > 50%
            success_rate = 1.0 if adv_mse > 1.5 * original_mse else 0.0
        
        return float(success_rate)
    
    def create_threat_scenarios(self) -> List[Dict[str, Any]]:
        """Create realistic threat scenarios based on FIN-ATT&CK."""
        scenarios = []
        
        # TA02: Order Book Spoofing
        scenarios.append({
            'id': 'TA02.001',
            'name': 'Order Book Spoofing',
            'description': 'Inject fake orders to manipulate price perception',
            'attack_type': 'data_poisoning',
            'parameters': {
                'poison_ratio': 0.05,
                'poison_strength': 0.3,
                'target_feature': 'order_flow'  # Placeholder
            },
            'expected_impact': 'Price manipulation, false liquidity signals',
            'mitigation': 'Order book anomaly detection, trade surveillance'
        })
        
        # TA07: Data Poisoning
        scenarios.append({
            'id': 'TA07.003',
            'name': 'Sentiment Analysis Poisoning',
            'description': 'Manipulate NLP models through poisoned training data',
            'attack_type': 'data_poisoning',
            'parameters': {
                'poison_ratio': 0.1,
                'poison_strength': 0.5
            },
            'expected_impact': 'Incorrect sentiment signals, trading errors',
            'mitigation': 'Robust training, data validation, anomaly detection'
        })
        
        # TA10: Model Evasion
        scenarios.append({
            'id': 'TA10.001',
            'name': 'Adversarial Example Attack',
            'description': 'Craft inputs that evade fraud detection models',
            'attack_type': 'fgsm',
            'parameters': {
                'epsilon': 0.1,
                'targeted': False
            },
            'expected_impact': 'Undetected fraudulent transactions',
            'mitigation': 'Adversarial training, robust models, ensemble methods'
        })
        
        # Custom: Quantum-Specific Attacks
        scenarios.append({
            'id': 'CUSTOM_Q01',
            'name': 'Quantum Noise Exploitation',
            'description': 'Exploit noise sensitivity of quantum models',
            'attack_type': 'data_poisoning',
            'parameters': {
                'poison_ratio': 0.15,
                'poison_strength': 0.2,
                'noise_type': 'structured'  # Placeholder
            },
            'expected_impact': 'Degraded quantum model performance',
            'mitigation': 'Noise-resilient circuits, error correction, hybrid designs'
        })
        
        return scenarios