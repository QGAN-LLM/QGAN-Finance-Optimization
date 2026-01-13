v"""
Quantum Utility Functions
Helper functions for quantum computing operations
"""

import numpy as np
import torch
import pennylane as qml
from typing import List, Tuple, Optional

class QuantumUtils:
    """Utility functions for quantum computing"""
    
    @staticmethod
    def initialize_parameters(n_qubits, n_layers, method='random'):
        """
        Initialize quantum circuit parameters
        Different initialization strategies for different purposes
        """
        if method == 'random':
            # Random initialization
            return torch.randn(n_layers, n_qubits, 3) * 0.01
        
        elif method == 'zeros':
            # Zero initialization
            return torch.zeros(n_layers, n_qubits, 3)
        
        elif method == 'xavier':
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (n_qubits * 3))
            return torch.randn(n_layers, n_qubits, 3) * scale
        
        elif method == 'he':
            # He initialization (for ReLU-like activations)
            scale = np.sqrt(2.0 / n_qubits)
            return torch.randn(n_layers, n_qubits, 3) * scale
        
        elif method == 'finance_aware':
            # Initialization aware of financial patterns
            params = torch.zeros(n_layers, n_qubits, 3)
            
            # Bias towards certain rotations based on financial intuition
            # Small initial rotations for stability
            params[:, :, 0] = torch.randn(n_layers, n_qubits) * 0.001 # RZ
            params[:, :, 1] = torch.randn(n_layers, n_qubits) * 0.001 # RY
            params[:, :, 2] = torch.randn(n_layers, n_qubits) * 0.001 # RZ
            
            return params
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    @staticmethod
    def calculate_gradient_variance(circuit, params, n_samples=100):
        """
        Calculate gradient variance to detect barren plateaus
        High variance might indicate trainability issues
        """
        gradients = []
        
        for _ in range(n_samples):
            # Create random input
            random_input = torch.randn(params.shape[1])
            
            # Calculate gradient
            params_clone = params.clone().detach().requires_grad_(True)
            output = circuit(random_input, params_clone)
            loss = torch.mean(output)
            loss.backward()
            
            grad = params_clone.grad.flatten()
            if grad is not None:
                gradients.append(grad.detach().numpy())
        
        if gradients:
            gradients_array = np.array(gradients)
            variance = np.var(gradients_array, axis=0).mean()
            return variance
        else:
            return 0.0
    
    @staticmethod
    def apply_parameter_shift(circuit, params, shift=np.pi/2):
        """
        Apply parameter shift rule for gradient calculation
        Useful for quantum circuits where autograd might not work
        """
        # This is a simplified implementation
        # In practice, would implement full parameter shift rule
        gradients = torch.zeros_like(params)
        
        for i in range(params.shape[0]):
            for j in range(params.shape[1]):
                for k in range(params.shape[2]):
                    # Shift plus
                    params_plus = params.clone()
                    params_plus[i, j, k] += shift
                    
                    # Shift minus
                    params_minus = params.clone()
                    params_minus[i, j, k] -= shift
                    
                    # Calculate circuit outputs
                    random_input = torch.randn(params.shape[1])
                    output_plus = torch.mean(circuit(random_input, params_plus))
                    output_minus = torch.mean(circuit(random_input, params_minus))
                    
                    # Calculate gradient
                    gradients[i, j, k] = (output_plus - output_minus) / (2 * np.sin(shift))
        
        return gradients
    
    @staticmethod
    def mitigate_noise(circuit_output, noise_model='depolarizing', error_rate=0.01):
        """
        Apply simple noise mitigation techniques
        """
        if noise_model == 'depolarizing':
            # Simple depolarizing noise mitigation
            mitigated = circuit_output * (1 - error_rate)
            return mitigated
        
        elif noise_model == 'amplitude_damping':
            # Amplitude damping noise mitigation
            mitigated = circuit_output / (1 - error_rate)
            return mitigated
        
        else:
            # No mitigation
            return circuit_output
    
    @staticmethod
    def calculate_quantum_fisher_information(circuit, params, n_samples=100):
        """
        Calculate quantum Fisher information
        Useful for understanding the information geometry of the circuit
        """
        # Simplified calculation
        # In practice, would use more sophisticated methods
        gradients = []
        
        for _ in range(n_samples):
            random_input = torch.randn(params.shape[1])
            params_clone = params.clone().detach().requires_grad_(True)
            
            output = circuit(random_input, params_clone)
            loss = torch.mean(output)
            loss.backward()
            
            if params_clone.grad is not None:
                grad = params_clone.grad.flatten().detach().numpy()
                gradients.append(grad)
        
        if gradients:
            gradients_array = np.array(gradients)
            # Fisher information matrix (approximation)
            fisher_info = np.cov(gradients_array.T)
            return fisher_info
        else:
            return np.eye(params.numel())
    
    @staticmethod
    def optimize_circuit_depth(circuit_config, max_depth=10):
        """
        Optimize circuit depth while maintaining expressibility
        """
        n_layers = circuit_config.get('n_layers', 3)
        n_qubits = circuit_config.get('n_qubits', 8)
        
        if n_layers > max_depth:
            print(f"Reducing circuit depth from {n_layers} to {max_depth}")
            circuit_config['n_layers'] = max_depth
        
        # Estimate circuit depth
        estimated_depth = n_layers * 3 # Each layer has ~3 operations per qubit
        
        return {
            'original_depth': n_layers,
            'optimized_depth': circuit_config['n_layers'],
            'estimated_operations': estimated_depth,
            'qubit_count': n_qubits
        }


class QuantumDataEncoder:
    """Encode classical data into quantum states"""
    
    @staticmethod
    def amplitude_encoding(data, n_qubits):
        """
        Encode data using amplitude encoding
        Requires data to be normalized to unit vector
        """
        data = data.flatten()
        
        # Pad or truncate to length 2^n_qubits
        target_length = 2 ** n_qubits
        if len(data) < target_length:
            # Pad with zeros
            padded = np.zeros(target_length)
            padded[:len(data)] = data
            data = padded
        elif len(data) > target_length:
            # Truncate
            data = data[:target_length]
        
        # Normalize
        norm = np.linalg.norm(data)
        if norm > 0:
            data = data / norm
        
        return torch.tensor(data, dtype=torch.float32)
    
    @staticmethod
    def angle_encoding(data, n_qubits):
        """
        Encode each data point as rotation angle on separate qubits
        Simple but limited capacity
        """
        data = data.flatten()
        
        # Use first n_qubits data points
        if len(data) > n_qubits:
            data = data[:n_qubits]
        elif len(data) < n_qubits:
            # Pad with zeros
            padded = np.zeros(n_qubits)
            padded[:len(data)] = data
            data = padded
        
        return torch.tensor(data, dtype=torch.float32)
    
    @staticmethod
    def dense_angle_encoding(data, n_qubits, n_layers=1):
        """
        Dense angle encoding with multiple layers
        Can encode more data points
        """
        data = data.flatten()
        
        # Calculate how many data points we can encode
        points_per_layer = n_qubits * 3 # 3 rotations per qubit
        total_capacity = points_per_layer * n_layers
        
        if len(data) > total_capacity:
            data = data[:total_capacity]
        elif len(data) < total_capacity:
            # Pad with zeros
            padded = np.zeros(total_capacity)
            padded[:len(data)] = data
            data = padded
        
        # Reshape for circuit parameters
        encoded = torch.tensor(data, dtype=torch.float32)
        encoded = encoded.reshape(n_layers, n_qubits, 3)
        
        return encoded
    
    @staticmethod
    def get_encoding_recommendation(data_size, n_qubits, encoding_type='auto'):
        """
        Recommend encoding method based on data size and qubit count
        """
        if encoding_type == 'auto':
            # Choose encoding based on data size
            if data_size <= n_qubits:
                return 'angle_encoding'
            elif data_size <= n_qubits * 3:
                return 'dense_angle_encoding'
            else:
                return 'amplitude_encoding'
        else:
            return encoding_type

