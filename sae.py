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
    """
    
    def __init__(self, sdr_dims: int, active_indices: Iterable[int]) -> None:
        """
        Initializes an SDR.
        """
        self.sdr_dims: int = sdr_dims
        self.active_indices: set[int] = set(active_indices)

    def to_dense(self) -> np.ndarray:
        """
        Converts the SDR to a dense numpy array of floats (0.0 or 1.0).
        """
        dense: np.ndarray = np.zeros(self.sdr_dims, dtype=np.float32)
        for idx in self.active_indices:
            dense[idx] = 1.0
        return dense

    def circular_shift(self, shift: int) -> 'SDR':
        """
        Applies a circular bit-shift to the SDR.
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
    
    Uses a linear projection followed by a Top-K activation function.
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
        
        # Learned Scaling Factor (s)
        self.decoder_scale: float = 1.0 / np.sqrt(density_k)
        
        # Biases
        self.b_pre: np.ndarray = np.zeros(input_dims, dtype=np.float32)
        self.encoder_bias: np.ndarray = np.full(sdr_dims, -0.1, dtype=np.float32)

    def _normalize_decoder(self, indices: np.ndarray = None) -> None:
        """Normalizes decoder weights to unit length (L2 = 1)."""
        if indices is None:
            norms = np.linalg.norm(self.decoder_weights, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self.decoder_weights /= norms
        else:
            norms = np.linalg.norm(self.decoder_weights[indices], axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self.decoder_weights[indices] /= norms

    def encode(self, dense_vector: np.ndarray) -> SDR:
        """
        Encodes a single dense vector into an SDR using Centering and Top-K.
        """
        x_centered = dense_vector - self.b_pre if self.use_centering else dense_vector
        activations = np.dot(x_centered, self.encoder_weights) + self.encoder_bias
        top_k_indices = np.argpartition(activations, -self.density_k)[-self.density_k:]
        return SDR(self.sdr_dims, top_k_indices)

    def decode(self, sdr: SDR) -> np.ndarray:
        """
        Reconstructs the dense vector from an SDR using the learned scale.
        """
        active_list = list(sdr.active_indices)
        reconstruction = self.decoder_scale * np.sum(self.decoder_weights[active_list, :], axis=0)
        if self.use_centering:
            reconstruction += self.b_pre
        return reconstruction

    def train(self, training_data: np.ndarray, val_data: np.ndarray = None, epochs: int = 10, lr: float = 0.01, batch_size: int = 32) -> dict[str, list[float]]:
        """
        Trains the SAE weights using Total Squared Error (SSE) for intuitive reporting.
        """
        if self.use_centering:
            self.b_pre = np.mean(training_data, axis=0)
            
        logger.info(f"Starting SAE Training: epochs={epochs}, lr={lr}, batch_size={batch_size}, samples={len(training_data)}")
        
        history = {"train_loss": [], "val_loss": []}
        n_samples = len(training_data)
        scale_lr = lr * 0.1
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            shuffled_indices = np.random.permutation(n_samples)
            updated_neurons = set()
            batch_grad_scale = 0.0
            
            for i, idx in enumerate(shuffled_indices):
                x = training_data[idx]
                x_centered = x - self.b_pre if self.use_centering else x
                
                # 1. Forward Pass
                activations = np.dot(x_centered, self.encoder_weights) + self.encoder_bias
                top_k_indices = np.argpartition(activations, -self.density_k)[-self.density_k:]
                
                # 2. Reconstruction
                bit_sum = np.sum(self.decoder_weights[top_k_indices], axis=0)
                x_hat = self.decoder_scale * bit_sum
                if self.use_centering:
                    x_hat += self.b_pre
                
                # 3. Loss (Total Squared Error)
                error = x_hat - x
                total_squared_error = np.sum(error**2)
                epoch_loss += total_squared_error
                
                # 4. Backward Pass (Simplified SSE Gradients)
                grad_x_hat = 2.0 * error
                
                batch_grad_scale += np.dot(grad_x_hat, bit_sum)
                self.decoder_weights[top_k_indices] -= lr * self.decoder_scale * grad_x_hat
                
                grad_act = self.decoder_scale * np.dot(self.decoder_weights[top_k_indices], grad_x_hat)
                self.encoder_weights[:, top_k_indices] -= lr * np.outer(x_centered, grad_act)
                self.encoder_bias[top_k_indices] -= lr * grad_act
                
                updated_neurons.update(top_k_indices)
                
                if (i + 1) % batch_size == 0 or (i + 1) == n_samples:
                    self.decoder_scale -= scale_lr * (batch_grad_scale / batch_size)
                    self.decoder_scale = max(0.001, self.decoder_scale)
                    batch_grad_scale = 0.0
                    self._normalize_decoder(np.array(list(updated_neurons)))
                    updated_neurons.clear()
            
            avg_train_loss = epoch_loss / n_samples
            history["train_loss"].append(avg_train_loss)
            
            avg_val_loss = 0.0
            if val_data is not None:
                val_errors = [np.sum((self.decode(self.encode(x_v)) - x_val)**2) for x_v, x_val in zip(val_data, val_data)]
                avg_val_loss = np.mean(val_errors)
                history["val_loss"].append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f", Val Loss: {avg_val_loss:.4f}" if val_data is not None else ""
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}{val_str}, Scale: {self.decoder_scale:.4f}")
            
        logger.info("SAE Training Complete.")
        return history

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
