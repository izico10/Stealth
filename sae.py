"""
Module for Semantic Autoencoding and Sparse Distributed Representations (SDR).
This module handles the projection of dense embeddings into a high-dimensional 
sparse space, achieving monosemanticity through sparsity.
"""

import numpy as np
import logging
from collections.abc import Iterable

# Configure logger for this module
logger = logging.getLogger(__name__)

class SDR:
    """
    Represents a Sparse Distributed Representation (SDR).
    
    An SDR is a high-dimensional binary vector where only a small percentage 
    of bits are active (1). This class stores the indices of active bits 
    for efficiency.
    """
    
    def __init__(self, sdr_dims: int, active_indices: Iterable[int]) -> None:
        """
        Initializes an SDR.
        
        Args:
            sdr_dims: The total dimensionality of the SDR.
            active_indices: An iterable of indices that are set to 1.
        """
        self.sdr_dims: int = sdr_dims
        self.active_indices: set[int] = set(active_indices)

    def to_dense(self) -> np.ndarray:
        """
        Converts the SDR to a dense numpy array of floats (0.0 or 1.0).
        
        Returns:
            A numpy array of shape (sdr_dims,).
        """
        dense: np.ndarray = np.zeros(self.sdr_dims, dtype=np.float32)
        for idx in self.active_indices:
            dense[idx] = 1.0
        return dense

    def circular_shift(self, shift: int) -> 'SDR':
        """
        Applies a circular bit-shift to the SDR.
        
        Shifting creates an orthogonal address in high-dimensional space, 
        allowing the system to distinguish the same token at different 
        sequence positions.
        
        Args:
            shift: The number of positions to shift.
            
        Returns:
            A new SDR instance with shifted indices.
        """
        shifted_indices: set[int] = { (idx + shift) % self.sdr_dims for idx in self.active_indices }
        return SDR(self.sdr_dims, shifted_indices)

    def __len__(self) -> int:
        """
        Returns the number of active bits.
        """
        return len(self.active_indices)


class SparseAutoencoder:
    """
    Encodes dense vectors into Sparse Distributed Representations (SDRs).
    
    Uses a linear projection followed by a Top-K activation function to 
    ensure a fixed level of sparsity, which helps in achieving monosemanticity.
    
    Architecture inspired by Bricken et al. (2023) Top-K SAEs.
    """
    
    def __init__(
        self, 
        input_dims: int, 
        sdr_dims: int = 10000, 
        density_k: int = 50, 
        use_ternary_weights: bool = True,
        projection_weight_density: float = 0.1,
        use_centering: bool = True,
        seed: int = 42
    ) -> None:
        """
        Initializes the SAE.
        """
        self.input_dims: int = input_dims
        self.sdr_dims: int = sdr_dims
        self.density_k: int = density_k
        self.use_ternary_weights: bool = use_ternary_weights
        self.projection_weight_density: float = projection_weight_density
        self.use_centering: bool = use_centering
        
        rng: np.random.Generator = np.random.default_rng(seed)
        
        # Encoder Weights (W_enc)
        if use_ternary_weights:
            self.encoder_weights: np.ndarray = rng.choice(
                a=[-1.0, 0.0, 1.0],
                size=(input_dims, sdr_dims),
                p=[projection_weight_density / 2, 1 - projection_weight_density, projection_weight_density / 2]
            ).astype(np.float32)
        else:
            self.encoder_weights = rng.standard_normal((input_dims, sdr_dims), dtype=np.float32) * 0.02
            
        # Decoder Weights (W_dec) - Normalized to Unit Length
        self.decoder_weights: np.ndarray = rng.standard_normal((sdr_dims, input_dims), dtype=np.float32)
        self._normalize_decoder()
        
        # Biases
        # b_pre: The mean of the data (centering)
        self.b_pre: np.ndarray = np.zeros(input_dims, dtype=np.float32)
        # b_enc: Initialized to be negative to act as a gate
        self.encoder_bias: np.ndarray = np.full(sdr_dims, -0.1, dtype=np.float32)

    def _normalize_decoder(self) -> None:
        """Normalizes decoder weights to unit length (L2 = 1)."""
        norms = np.linalg.norm(self.decoder_weights, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.decoder_weights /= norms

    def encode(self, dense_vector: np.ndarray) -> SDR:
        """
        Encodes a single dense vector into an SDR using Centering and Top-K.
        """
        # Step 1: Centering
        x_centered = dense_vector - self.b_pre if self.use_centering else dense_vector
        
        # Step 2: Encoder Projection + Bias
        activations = np.dot(x_centered, self.encoder_weights) + self.encoder_bias
        
        # Step 3: Top-K Sparsity
        top_k_indices = np.argpartition(activations, -self.density_k)[-self.density_k:]
        
        return SDR(self.sdr_dims, top_k_indices)

    def decode(self, sdr: SDR) -> np.ndarray:
        """
        Reconstructs the dense vector from an SDR.
        """
        # Step 4: Reconstruction (Sum active decoder weights + b_pre)
        # Vectorized indexing for speed and hardware-alignment
        active_list = list(sdr.active_indices)
        reconstruction = np.sum(self.decoder_weights[active_list, :], axis=0)
            
        if self.use_centering:
            reconstruction += self.b_pre
            
        return reconstruction

    def train(self, training_data: np.ndarray, epochs: int = 10, lr: float = 0.01) -> list[float]:
        """
        Trains the SAE weights to minimize reconstruction error using Top-K.
        
        Args:
            training_data: 2D array of embeddings (N, input_dims).
            epochs: Number of training passes.
            lr: Learning rate.
            
        Returns:
            List of average loss per epoch.
        """
        # Initialize b_pre as the mean of the training data
        if self.use_centering:
            self.b_pre = np.mean(training_data, axis=0)
            
        logger.info(f"Starting SAE Training: epochs={epochs}, lr={lr}, samples={len(training_data)}")
        logger.info(f"Config: sdr_dims={self.sdr_dims}, density_k={self.density_k}, centering={self.use_centering}, ternary={self.use_ternary_weights}")
        
        losses = []
        n_samples = len(training_data)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                x = training_data[i]
                
                # Forward Pass
                sdr = self.encode(x)
                x_hat = self.decode(sdr)
                
                # Loss (MSE)
                error = x_hat - x
                epoch_loss += np.mean(error**2)
                
                # Backward Pass (STE)
                grad_x_hat = error / self.input_dims
                
                # Update Decoder & Normalize
                active_list = list(sdr.active_indices)
                self.decoder_weights[active_list, :] -= lr * grad_x_hat
                self._normalize_decoder()
                
                # Update Encoder (only for active bits)
                for idx in active_list:
                    grad_act = np.dot(grad_x_hat, self.decoder_weights[idx, :])
                    # Update weights and bias
                    x_centered = x - self.b_pre if self.use_centering else x
                    self.encoder_weights[:, idx] -= lr * grad_act * x_centered
                    self.encoder_bias[idx] -= lr * grad_act
            
            avg_loss = epoch_loss / n_samples
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}")
            
        logger.info("SAE Training Complete.")
        return losses

    def encode_batch(self, dense_vectors: np.ndarray) -> list[SDR]:
        """
        Encodes a batch of dense vectors into SDRs.
        """
        x_centered = dense_vectors - self.b_pre if self.use_centering else dense_vectors
        batch_activations = np.dot(x_centered, self.encoder_weights) + self.encoder_bias
        
        sdrs = []
        for i in range(batch_activations.shape[0]):
            activations = batch_activations[i]
            top_k_indices = np.argpartition(activations, -self.density_k)[-self.density_k:]
            sdrs.append(SDR(self.sdr_dims, top_k_indices))
            
        return sdrs
