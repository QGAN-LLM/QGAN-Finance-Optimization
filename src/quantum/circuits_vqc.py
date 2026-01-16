"""Variational Quantum Circuit definitions for QGAN."""
import pennylane as qml
from pennylane import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VQCGenerator:
    """Parameterized Quantum Circuit for QGAN generator."""
    
    def __init__(self, n_qubits: int = 4, depth: int = 3, 
                 entanglement: str = "linear", encoding: str = "angle",
                 noise_model: Optional[str] = None, noise_prob: float = 0.01):
        """
        Initialize the quantum generator circuit.
        
        Args:
            n_qubits: Number of qubits in the circuit
            depth: Number of repeating layers
            entanglement: Type of entanglement ("linear", "full", "circular")
            encoding: Data encoding method ("angle", "amplitude", "basis")
            noise_model: Type of noise to inject
            noise_prob: Probability of noise occurrence
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.encoding = encoding
        self.noise_model = noise_model
        self.noise_prob = noise_prob
        
        # Create device with or without noise
        if noise_model:
            self.device = self._create_noisy_device()
        else:
            self.device = qml.device("default.qubit", wires=n_qubits)
        
        # Define the quantum node
        self.qnode = qml.QNode(self.circuit, self.device)
        
        # Initialize parameters
        self.params = self.init_parameters()
        
        logger.info(f"Initialized VQCGenerator with {n_qubits} qubits, "
                   f"depth {depth}, {entanglement} entanglement")
    
    def _create_noisy_device(self):
        """Create a device with noise model."""
        # Simple depolarizing noise implementation
        if self.noise_model == "depolarizing":
            noise_gates = [qml.DepolarizingChannel(self.noise_prob, wires=i) 
                          for i in range(self.n_qubits)]
            return qml.device("default.mixed", wires=self.n_qubits)
        else:
            return qml.device("default.qubit", wires=self.n_qubits)
    
    def init_parameters(self) -> np.ndarray:
        """Initialize variational parameters."""
        # Parameters per layer: rotation angles + entanglement parameters
        n_params_per_qubit = 3  # RX, RY, RZ rotations
        n_params = self.depth * (self.n_qubits * n_params_per_qubit)
        
        # Initialize with small random values
        return np.random.uniform(0, 2*np.pi, size=n_params, requires_grad=True)
    
    def circuit(self, params: np.ndarray, latent_input: Optional[np.ndarray] = None):
        """
        Main quantum circuit.
        
        Args:
            params: Variational parameters
            latent_input: Classical latent vector (if using hybrid approach)
            
        Returns:
            Expectation values of Pauli-Z on each qubit
        """
        # Encode latent input if provided
        if latent_input is not None:
            self.encode_data(latent_input)
        
        # Apply variational layers
        param_idx = 0
        for layer in range(self.depth):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                qml.RX(params[param_idx], wires=qubit)
                qml.RY(params[param_idx + 1], wires=qubit)
                qml.RZ(params[param_idx + 2], wires=qubit)
                param_idx += 3
            
            # Entanglement layer
            self.apply_entanglement()
            
            # Apply noise if specified
            if self.noise_model:
                self.apply_noise()
        
        # Measure expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def encode_data(self, data: np.ndarray):
        """Encode classical data into quantum state."""
        if self.encoding == "angle":
            # Angle encoding: each data point maps to a rotation angle
            for i in range(min(len(data), self.n_qubits)):
                qml.RY(data[i], wires=i)
        elif self.encoding == "amplitude":
            # Amplitude encoding (requires data normalized to unit vector)
            qml.AmplitudeEmbedding(data, wires=range(self.n_qubits), normalize=True)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding}")
    
    def apply_entanglement(self):
        """Apply entanglement gates based on configuration."""
        if self.entanglement == "linear":
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        elif self.entanglement == "circular":
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i+1) % self.n_qubits])
        elif self.entanglement == "full":
            # Fully connected with CNOTs (expensive for many qubits)
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
    
    def apply_noise(self):
        """Apply noise gates if noise model is specified."""
        if self.noise_model == "depolarizing":
            for i in range(self.n_qubits):
                qml.DepolarizingChannel(self.noise_prob, wires=i)
    
    def generate(self, n_samples: int = 100) -> np.ndarray:
        """
        Generate synthetic samples from the quantum generator.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of generated samples
        """
        samples = []
        for _ in range(n_samples):
            # Sample from latent space if using, else use circuit parameters
            if hasattr(self, 'latent_dim'):
                z = np.random.randn(self.latent_dim)
                sample = self.qnode(self.params, z)
            else:
                sample = self.qnode(self.params)
            samples.append(sample)
        
        return np.array(samples)
    
    def get_state(self) -> np.ndarray:
        """Get the current quantum state for tomography."""
        @qml.qnode(self.device)
        def state_circuit():
            self.circuit(self.params)
            return qml.state()
        
        return state_circuit()