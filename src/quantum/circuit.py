"""
Quantum Circuit Definitions
Various quantum circuit architectures for QGAN
"""

import pennylane as qml
import numpy as np
import torch
from typing import List, Tuple, Optional

class QuantumCircuitLibrary:
    """Library of quantum circuit architectures for different purposes"""
    
    @staticmethod
    def hardware_efficient_ansatz(n_qubits, n_layers, params):
        """
        Hardware-efficient ansatz suitable for NISQ devices
        Emphasizes local gates and limited connectivity
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Encode inputs
            for i in range(n_qubits):
                qml.RY(inputs[i % len(inputs)], wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                # Single-qubit rotations
                for i in range(n_qubits):
                    qml.RZ(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Nearest-neighbor entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Optional: Add some longer-range connections
                if n_qubits > 2:
                    qml.CNOT(wires=[0, n_qubits - 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return circuit
    
    @staticmethod
    def strongly_entangling_ansatz(n_qubits, n_layers, params):
        """
        Strongly entangling ansatz for more expressive power
        Better for learning complex distributions
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Encode inputs
            for i in range(n_qubits):
                qml.RY(inputs[i % len(inputs)], wires=i)
                qml.RZ(inputs[(i + 1) % len(inputs)], wires=i)
            
            # Strongly entangling layers
            for layer in range(n_layers):
                # Single-qubit rotations on all qubits
                for i in range(n_qubits):
                    qml.Rot(*weights[layer, i], wires=i)
                
                # Entangle all qubits in a circular pattern
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return circuit
    
    @staticmethod
    def data_reuploading_circuit(n_qubits, n_layers, params):
        """
        Data reuploading circuit for better data encoding
        Re-encodes data at each layer
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Initial encoding
            for i in range(n_qubits):
                qml.RY(inputs[i % len(inputs)], wires=i)
            
            # Layers with data reuploading
            for layer in range(n_layers):
                # Variational block
                for i in range(n_qubits):
                    qml.RZ(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Data reuploading
                for i in range(n_qubits):
                    qml.RY(inputs[(i + layer) % len(inputs)], wires=i)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return circuit
    
    @staticmethod
    def finance_specific_circuit(n_qubits, n_layers, params):
        """
        Circuit designed specifically for financial time series patterns
        Emphasizes features important for market microstructure
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Encode financial features
            # Assuming inputs contain: [returns, volatility, volume, etc.]
            if len(inputs) >= 3:
                # Returns encoding
                for i in range(min(3, n_qubits)):
                    qml.RY(inputs[0] * np.pi, wires=i) # Recent returns
                
                # Volatility encoding
                if n_qubits >= 6:
                    for i in range(3, min(6, n_qubits)):
                        qml.RZ(inputs[1] * np.pi, wires=i) # Volatility
                
                # Volume/other features
                if n_qubits >= 8 and len(inputs) >= 3:
                    for i in range(6, n_qubits):
                        qml.RY(inputs[2] * np.pi, wires=i) # Volume indicator
            
            # Financial pattern learning layers
            for layer in range(n_layers):
                # Autocorrelation learning (important for financial series)
                for i in range(n_qubits):
                    qml.RZ(weights[layer, i, 0], wires=i)
                
                # Momentum/mean-reversion patterns
                for i in range(n_qubits - 1):
                    qml.CRY(weights[layer, i, 1], wires=[i, i + 1])
                
                # Volatility clustering patterns
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 2], wires=i)
                
                # Fat-tail distribution encoding
                if layer % 2 == 0: # Every other layer
                    for i in range(0, n_qubits, 2):
                        if i + 1 < n_qubits:
                            qml.CNOT(wires=[i, i + 1])
                else:
                    for i in range(1, n_qubits, 2):
                        if i + 1 < n_qubits:
                            qml.CNOT(wires=[i, i + 1])
            
            # Measure in basis relevant for financial predictions
            measurements = []
            for i in range(n_qubits):
                # Mix of Z and X measurements for different aspects
                if i % 3 == 0:
                    measurements.append(qml.expval(qml.PauliZ(i))) # Trend
                elif i % 3 == 1:
                    measurements.append(qml.expval(qml.PauliX(i))) # Volatility
                else:
                    measurements.append(qml.expval(qml.PauliY(i))) # Momentum
            
            return measurements
        
        return circuit
    
    @staticmethod
    def get_circuit_by_name(name, n_qubits, n_layers, params):
        """Get circuit by name from library"""
        circuits = {
            'hardware_efficient': QuantumCircuitLibrary.hardware_efficient_ansatz,
            'strongly_entangling': QuantumCircuitLibrary.strongly_entangling_ansatz,
            'data_reuploading': QuantumCircuitLibrary.data_reuploading_circuit,
            'finance_specific': QuantumCircuitLibrary.finance_specific_circuit
        }
        
        if name not in circuits:
            print(f"Warning: Circuit {name} not found. Using hardware_efficient.")
            name = 'hardware_efficient'
        
        return circuits[name](n_qubits, n_layers, params)


class CircuitAnalyzer:
    """Analyze quantum circuits for various properties"""
    
    @staticmethod
    def calculate_expressibility(circuit, n_samples=1000):
        """
        Calculate expressibility of a quantum circuit
        How well it can explore the Hilbert space
        """
        # This is a simplified version
        # In practice, would use more sophisticated measures
        return 0.5 # Placeholder
    
    @staticmethod
    def calculate_entanglement_capability(circuit, n_qubits):
        """
        Calculate entanglement capability of the circuit
        """
        # Simplified measure based on number of CNOT gates
        # In practice, would use entanglement entropy measures
        return min(1.0, n_qubits / 8) # Placeholder
    
    @staticmethod
    def estimate_noise_resilience(circuit_config):
        """
        Estimate noise resilience based on circuit structure
        """
        # Factors: circuit depth, number of 2-qubit gates, etc.
        depth = circuit_config.get('n_layers', 3)
        n_qubits = circuit_config.get('n_qubits', 8)
        
        # Simplified resilience score
        resilience = 1.0 / (1.0 + depth * n_qubits / 100)
        return max(0.1, min(1.0, resilience))
    
    @staticmethod
    def get_circuit_recommendation(problem_type, n_qubits, noise_level='medium'):
        """
        Recommend circuit based on problem type and constraints
        """
        recommendations = {
            'financial_time_series': {
                'low_noise': 'finance_specific',
                'medium_noise': 'hardware_efficient',
                'high_noise': 'hardware_efficient'
            },
            'generative_modeling': {
                'low_noise': 'strongly_entangling',
                'medium_noise': 'data_reuploading',
                'high_noise': 'hardware_efficient'
            },
            'optimization': {
                'low_noise': 'strongly_entangling',
                'medium_noise': 'hardware_efficient',
                'high_noise': 'hardware_efficient'
            }
        }
        
        if problem_type not in recommendations:
            problem_type = 'generative_modeling'
        
        if noise_level not in ['low', 'medium', 'high']:
            noise_level = 'medium'
        
        return recommendations[problem_type][noise_level]

