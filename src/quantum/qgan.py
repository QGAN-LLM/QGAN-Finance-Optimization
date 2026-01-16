"""
Quantum Generative Adversarial Network implementation.
"""
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
import logging
from datetime import datetime

from .circuits import VQCGenerator

logger = logging.getLogger(__name__)

class QGANModel:
    """Quantum Generative Adversarial Network for financial data generation."""
    
    def __init__(self, 
                 quantum_config: Dict[str, Any],
                 classical_config: Dict[str, Any],
                 latent_dim: int = 10):
        """
        Initialize QGAN model.
        
        Args:
            quantum_config: Configuration for quantum components
            classical_config: Configuration for classical components
            latent_dim: Dimension of latent space
        """
        self.quantum_config = quantum_config
        self.classical_config = classical_config
        self.latent_dim = latent_dim
        
        # Initialize components
        self.generator = self._init_generator()
        self.discriminator = self._init_discriminator()
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=quantum_config.get('generator_lr', 0.001)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=quantum_config.get('discriminator_lr', 0.0002)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }
        
        logger.info(f"Initialized QGAN with {quantum_config['n_qubits']} qubits")
    
    def _init_generator(self):
        """Initialize quantum generator."""
        return HybridQuantumGenerator(
            n_qubits=self.quantum_config['n_qubits'],
            circuit_depth=self.quantum_config['circuit_depth'],
            entanglement_type=self.quantum_config['entanglement_type'],
            encoding_type=self.quantum_config['encoding_type'],
            latent_dim=self.latent_dim,
            noise_model=self.quantum_config.get('noise_model'),
            noise_probability=self.quantum_config.get('noise_probability', 0.01)
        )
    
    def _init_discriminator(self):
        """Initialize classical discriminator."""
        # Input size depends on generator output
        input_size = self.quantum_config['n_qubits']
        hidden_dims = self.classical_config.get('discriminator_hidden_dims', [128, 64, 32])
        
        return ClassicalDiscriminator(
            input_size=input_size,
            hidden_dims=hidden_dims,
            dropout=self.classical_config.get('discriminator_dropout', 0.2)
        )
    
    def train(self, 
              train_data: np.ndarray,
              val_data: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              tracker: Optional[Any] = None) -> Dict[str, list]:
        """
        Train the QGAN model.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            tracker: Experiment tracker for logging
            
        Returns:
            Training history
        """
        logger.info(f"Starting QGAN training for {epochs} epochs")
        
        n_samples = len(train_data)
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_d_real_acc = 0
            epoch_d_fake_acc = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for batch_idx in range(n_batches):
                # Get batch
                batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                real_batch = train_data[batch_indices]
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                self.d_optimizer.zero_grad()
                
                # Real data
                real_labels = torch.ones(batch_size, 1)
                real_output = self.discriminator(torch.tensor(real_batch, dtype=torch.float32))
                d_real_loss = self.criterion(real_output, real_labels)
                d_real_acc = (real_output > 0.5).float().mean().item()
                
                # Fake data
                fake_data = self.generator.generate(batch_size)
                fake_labels = torch.zeros(batch_size, 1)
                fake_output = self.discriminator(fake_data)
                d_fake_loss = self.criterion(fake_output, fake_labels)
                d_fake_acc = (fake_output < 0.5).float().mean().item()
                
                # Combined discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # -----------------
                # Train Generator
                # -----------------
                self.g_optimizer.zero_grad()
                
                # Generate fake data
                fake_data = self.generator.generate(batch_size)
                
                # Try to fool discriminator
                misleading_labels = torch.ones(batch_size, 1)
                g_output = self.discriminator(fake_data)
                g_loss = self.criterion(g_output, misleading_labels)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                # Accumulate metrics
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_d_real_acc += d_real_acc
                epoch_d_fake_acc += d_fake_acc
            
            # Average metrics over batches
            epoch_g_loss /= n_batches
            epoch_d_loss /= n_batches
            epoch_d_real_acc /= n_batches
            epoch_d_fake_acc /= n_batches
            
            # Update history
            self.history['g_loss'].append(epoch_g_loss)
            self.history['d_loss'].append(epoch_d_loss)
            self.history['d_real_acc'].append(epoch_d_real_acc)
            self.history['d_fake_acc'].append(epoch_d_fake_acc)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"G Loss: {epoch_g_loss:.4f} | "
                    f"D Loss: {epoch_d_loss:.4f} | "
                    f"D Real Acc: {epoch_d_real_acc:.3f} | "
                    f"D Fake Acc: {epoch_d_fake_acc:.3f}"
                )
            
            # Log to tracker if provided
            if tracker:
                tracker.log_metrics({
                    'g_loss': epoch_g_loss,
                    'd_loss': epoch_d_loss,
                    'd_real_acc': epoch_d_real_acc,
                    'd_fake_acc': epoch_d_fake_acc
                }, step=epoch)
            
            # Early stopping check (simplified)
            if epoch > 20 and self._check_early_stopping():
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("QGAN training complete")
        return self.history
    
    def _check_early_stopping(self, patience: int = 10) -> bool:
        """Check if training should stop early."""
        if len(self.history['g_loss']) < patience * 2:
            return False
        
        # Check if generator loss hasn't improved
        recent_losses = self.history['g_loss'][-patience:]
        min_recent = min(recent_losses)
        min_overall = min(self.history['g_loss'][:-patience])
        
        return min_recent >= min_overall
    
    def generate_synthetic(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate synthetic data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Synthetic data array
        """
        logger.info(f"Generating {n_samples} synthetic samples")
        return self.generator.generate(n_samples)
    
    def get_quantum_state(self) -> np.ndarray:
        """Get current quantum state for tomography."""
        return self.generator.get_state()
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'history': self.history,
            'config': {
                'quantum': self.quantum_config,
                'classical': self.classical_config
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Model loaded from {filepath}")


class HybridQuantumGenerator(nn.Module):
    """Hybrid quantum-classical generator with latent space."""
    
    def __init__(self, n_qubits: int = 4, circuit_depth: int = 3,
                 entanglement_type: str = "linear", encoding_type: str = "angle",
                 latent_dim: int = 10, noise_model: Optional[str] = None,
                 noise_probability: float = 0.01):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.circuit_depth = circuit_depth
        
        # Classical neural network to process latent vector
        self.latent_processor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_qubits * 2),  # Output: angles for quantum circuit
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Quantum circuit
        self.vqc = VQCGenerator(
            n_qubits=n_qubits,
            depth=circuit_depth,
            entanglement=entanglement_type,
            encoding=encoding_type,
            noise_model=noise_model,
            noise_prob=noise_probability
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator."""
        # Process latent vector
        processed_z = self.latent_processor(z)
        
        # Convert to numpy for quantum circuit
        processed_z_np = processed_z.detach().numpy().reshape(-1, self.n_qubits, 2)
        
        # Generate samples using quantum circuit
        samples = []
        for batch in processed_z_np:
            # Use angles from classical NN as rotation parameters
            batch_sample = self.vqc.qnode(self.vqc.params, batch.flatten())
            samples.append(batch_sample)
        
        return torch.tensor(np.array(samples), dtype=torch.float32)
    
    def generate(self, n_samples: int = 100) -> torch.Tensor:
        """Generate samples from random latent vectors."""
        z = torch.randn(n_samples, self.latent_dim)
        with torch.no_grad():
            samples = self.forward(z)
        return samples
    
    def get_state(self) -> np.ndarray:
        """Get current quantum state."""
        return self.vqc.get_state()


class ClassicalDiscriminator(nn.Module):
    """Classical discriminator network."""
    
    def __init__(self, input_size: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.network(x)