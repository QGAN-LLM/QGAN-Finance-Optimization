"""
Quantum state tomography and analysis tools.
"""
import numpy as np
from typing import Tuple, Dict, Any
import logging
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class QuantumStateAnalyzer:
    """Analyzes quantum states through tomography."""
    
    def __init__(self, n_qubits: int = 4):
        """
        Initialize quantum state analyzer.
        
        Args:
            n_qubits: Number of qubits in the system
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
    
    def perform_tomography(self, 
                          state: np.ndarray,
                          method: str = 'mle',
                          shots: int = 10000) -> Dict[str, Any]:
        """
        Perform quantum state tomography.
        
        Args:
            state: Quantum state density matrix or statevector
            method: Tomography method ('mle', 'linear', 'bayesian')
            shots: Number of measurement shots (for simulation)
            
        Returns:
            Dictionary with tomography results
        """
        logger.info(f"Performing quantum state tomography ({method})")
        
        # Ensure state is valid
        state = self._validate_state(state)
        
        if method == 'mle':
            results = self._mle_tomography(state, shots)
        elif method == 'linear':
            results = self._linear_tomography(state, shots)
        elif method == 'bayesian':
            results = self._bayesian_tomography(state, shots)
        else:
            raise ValueError(f"Unknown tomography method: {method}")
        
        # Calculate additional metrics
        results.update(self._calculate_state_metrics(state))
        
        logger.info(f"Tomography complete. Fidelity: {results.get('fidelity', 0):.4f}")
        return results
    
    def _validate_state(self, state: np.ndarray) -> np.ndarray:
        """Validate and normalize quantum state."""
        # Reshape if needed
        if state.ndim == 1:
            # Statevector
            state = state.reshape(-1, 1)
            density_matrix = state @ state.conj().T
        elif state.ndim == 2:
            # Already density matrix
            if state.shape[0] == state.shape[1]:
                density_matrix = state
            else:
                raise ValueError(f"Invalid state shape: {state.shape}")
        else:
            raise ValueError(f"Invalid state dimensions: {state.ndim}")
        
        # Ensure trace is 1
        trace = np.trace(density_matrix)
        if not np.isclose(trace, 1.0, atol=1e-10):
            density_matrix = density_matrix / trace
        
        return density_matrix
    
    def _mle_tomography(self, 
                       true_state: np.ndarray,
                       shots: int = 10000) -> Dict[str, Any]:
        """Maximum Likelihood Estimation tomography."""
        # Simulate measurements
        measurements = self._simulate_measurements(true_state, shots)
        
        # MLE reconstruction (simplified)
        # In practice, you would use a proper optimization algorithm
        reconstructed = self._reconstruct_from_measurements(measurements)
        
        # Calculate fidelity with true state
        fidelity = self.calculate_fidelity(reconstructed, true_state)
        
        return {
            'method': 'mle',
            'reconstructed_state': reconstructed,
            'fidelity': fidelity,
            'measurements': measurements,
            'shots': shots
        }
    
    def _linear_tomography(self,
                          true_state: np.ndarray,
                          shots: int = 10000) -> Dict[str, Any]:
        """Linear inversion tomography."""
        measurements = self._simulate_measurements(true_state, shots)
        
        # Linear inversion (simplified)
        pauli_basis = self._get_pauli_basis()
        reconstructed = np.zeros_like(true_state, dtype=complex)
        
        for pauli in pauli_basis:
            # Estimate expectation value from measurements
            expectation = self._estimate_expectation(measurements, pauli)
            reconstructed += expectation * pauli
        
        # Ensure valid density matrix
        reconstructed = self._make_valid_density_matrix(reconstructed)
        
        fidelity = self.calculate_fidelity(reconstructed, true_state)
        
        return {
            'method': 'linear',
            'reconstructed_state': reconstructed,
            'fidelity': fidelity,
            'measurements': measurements,
            'shots': shots
        }
    
    def _bayesian_tomography(self,
                            true_state: np.ndarray,
                            shots: int = 10000) -> Dict[str, Any]:
        """Bayesian tomography (placeholder)."""
        # This is a simplified implementation
        # Full Bayesian tomography would require MCMC sampling
        
        # Use linear inversion as starting point
        linear_result = self._linear_tomography(true_state, shots)
        reconstructed = linear_result['reconstructed_state']
        
        # Add Bayesian refinement (simplified)
        # In practice, you would sample from posterior distribution
        noise = np.random.normal(0, 0.01, reconstructed.shape) * (1 + 1j)
        reconstructed_bayes = reconstructed + noise
        reconstructed_bayes = self._make_valid_density_matrix(reconstructed_bayes)
        
        fidelity = self.calculate_fidelity(reconstructed_bayes, true_state)
        
        return {
            'method': 'bayesian',
            'reconstructed_state': reconstructed_bayes,
            'fidelity': fidelity,
            'shots': shots
        }
    
    def _simulate_measurements(self, 
                              state: np.ndarray,
                              shots: int = 10000) -> Dict[str, np.ndarray]:
        """Simulate measurements in different bases."""
        measurements = {}
        
        # Pauli bases to measure
        bases = ['X', 'Y', 'Z']
        
        for basis in bases:
            # Simulate measurements in this basis
            # This is a simplified simulation
            probabilities = np.diag(state).real
            probabilities = np.maximum(probabilities, 0)  # Ensure non-negative
            probabilities = probabilities / probabilities.sum()
            
            # Sample measurement outcomes
            outcomes = np.random.choice(self.dim, size=shots, p=probabilities)
            counts = np.bincount(outcomes, minlength=self.dim)
            
            measurements[basis] = counts / shots
        
        return measurements
    
    def _reconstruct_from_measurements(self, 
                                     measurements: Dict[str, np.ndarray]) -> np.ndarray:
        """Reconstruct state from measurement counts."""
        # Simplified reconstruction
        # In practice, you would solve the MLE optimization problem
        
        dim = self.dim
        reconstructed = np.eye(dim) / dim  # Start with maximally mixed state
        
        # Update based on measurements (simplified)
        for basis, counts in measurements.items():
            diagonal = counts
            basis_matrix = np.diag(diagonal)
            reconstructed = 0.7 * reconstructed + 0.3 * basis_matrix
        
        # Ensure valid density matrix
        reconstructed = self._make_valid_density_matrix(reconstructed)
        
        return reconstructed
    
    def _get_pauli_basis(self) -> list:
        """Generate Pauli basis matrices."""
        # For single qubit
        pauli_matrices = [
            np.array([[1, 0], [0, 1]]),      # I
            np.array([[0, 1], [1, 0]]),      # X
            np.array([[0, -1j], [1j, 0]]),   # Y
            np.array([[1, 0], [0, -1]])      # Z
        ]
        
        # Generate tensor products for multi-qubit system
        basis = []
        # This is simplified - in practice you need all tensor products
        for i in range(4):
            basis.append(pauli_matrices[i])
        
        return basis
    
    def _estimate_expectation(self, 
                            measurements: Dict[str, np.ndarray],
                            operator: np.ndarray) -> float:
        """Estimate expectation value from measurements."""
        # Simplified estimation
        # In practice, you would decompose operator into measurable components
        
        if operator.shape == (2, 2):  # Single qubit operator
            # Estimate from Z-basis measurements
            if 'Z' in measurements:
                p0, p1 = measurements['Z'][0], measurements['Z'][1]
                # For Pauli Z, expectation is p0 - p1
                if np.array_equal(operator, np.array([[1, 0], [0, -1]])):
                    return p0 - p1
        
        return 0.0  # Default
    
    def _make_valid_density_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is a valid density matrix."""
        # Hermitian
        matrix = (matrix + matrix.conj().T) / 2
        
        # Positive semi-definite (simplified)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Reconstruct
        valid_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        
        # Normalize trace to 1
        valid_matrix = valid_matrix / np.trace(valid_matrix)
        
        return valid_matrix
    
    def calculate_fidelity(self, rho: np.ndarray, sigma: np.ndarray) -> float:
        """Calculate fidelity between two density matrices."""
        sqrt_rho = sqrtm(rho)
        fidelity_matrix = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
        fidelity = np.real(np.trace(fidelity_matrix))
        
        # Ensure fidelity is in [0, 1]
        fidelity = max(0, min(1, fidelity))
        
        return fidelity
    
    def _calculate_state_metrics(self, state: np.ndarray) -> Dict[str, float]:
        """Calculate various metrics for quantum state."""
        metrics = {}
        
        # Purity
        purity = np.real(np.trace(state @ state))
        metrics['purity'] = purity
        
        # Von Neumann entropy
        eigenvalues = np.linalg.eigvalsh(state)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Remove zeros
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        metrics['entropy'] = entropy
        
        # For multi-qubit states, calculate entanglement entropy
        if self.n_qubits > 1:
            entanglement = self._calculate_entanglement_entropy(state)
            metrics['entanglement_entropy'] = entanglement
        
        return metrics
    
    def _calculate_entanglement_entropy(self, state: np.ndarray) -> float:
        """Calculate entanglement entropy for bipartite systems."""
        if self.n_qubits < 2:
            return 0.0
        
        # Simplified: trace out half the qubits
        n_qubits_a = self.n_qubits // 2
        n_qubits_b = self.n_qubits - n_qubits_a
        
        # This is a placeholder - proper partial trace calculation
        # would require reshaping and tracing
        
        # For now, return a simple measure
        if self.n_qubits == 2:
            # For 2-qubit states, calculate concurrence
            try:
                # Calculate concurrence (simplified)
                R = state @ np.kron([[0, 0, 0, -1], [0, 0, 1, 0], 
                                    [0, 1, 0, 0], [-1, 0, 0, 0]], 
                                   np.eye(2))
                eigenvalues = np.linalg.eigvals(R)
                eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
                concurrence = max(0, eigenvalues[0] - np.sum(eigenvalues[1:]))
                
                # Convert concurrence to entanglement entropy
                # Using formula for two-qubit states
                if concurrence > 0:
                    entropy = self._entropy_from_concurrence(concurrence)
                    return entropy
            except:
                pass
        
        return 0.0
    
    def _entropy_from_concurrence(self, concurrence: float) -> float:
        """Convert concurrence to entanglement entropy."""
        # For two-qubit states
        c = concurrence
        if c == 0:
            return 0
        
        # Binary entropy function
        def h(x):
            if x <= 0 or x >= 1:
                return 0
            return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
        
        lambda_plus = (1 + np.sqrt(1 - c**2)) / 2
        return h(lambda_plus)
    
    def visualize_state(self, 
                       state: np.ndarray,
                       save_path: Optional[str] = None) -> plt.Figure:
        """Visualize quantum state."""
        fig = plt.figure(figsize=(15, 5))
        
        # 1. Density matrix heatmap
        ax1 = plt.subplot(131)
        im = ax1.imshow(np.abs(state), cmap='viridis')
        plt.colorbar(im, ax=ax1)
        ax1.set_title('Density Matrix (Magnitude)')
        ax1.set_xlabel('Basis State')
        ax1.set_ylabel('Basis State')
        
        # 2. Real and imaginary parts
        ax2 = plt.subplot(132)
        ax2.plot(state.real.flatten(), label='Real', alpha=0.7)
        ax2.plot(state.imag.flatten(), label='Imag', alpha=0.7)
        ax2.set_title('State Vector Components')
        ax2.set_xlabel('Element Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogram of diagonal elements (probabilities)
        ax3 = plt.subplot(133)
        probabilities = np.diag(state).real
        ax3.bar(range(len(probabilities)), probabilities)
        ax3.set_title('Measurement Probabilities')
        ax3.set_xlabel('Basis State')
        ax3.set_ylabel('Probability')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig