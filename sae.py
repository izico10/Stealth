"""
Module for Semantic Autoencoding and Sparse Distributed Representations (SDR).
This module handles the projection of dense embeddings into a high-dimensional 
sparse space, achieving monosemanticity through sparsity.
"""

import numpy as np
from collections.abc import Iterable

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
    """
    
    def __init__(
        self, 
        input_dims: int, 
        sdr_dims: int = 10000, 
        sparsity_k: int = 50, 
        use_ternary_projection: bool = True,
        projection_weight_density: float = 0.1,
        seed: int = 42
    ) -> None:
        """
        Initializes the SAE with a random projection.
        
        Args:
            input_dims: Dimension of the input dense embeddings.
            sdr_dims: Dimension of the output SDR.
            sparsity_k: Number of active bits to keep (sparsity constraint).
            use_ternary_projection: If True, uses a sparse ternary matrix (-1, 0, 1).
            projection_weight_density: Probability of a non-zero weight in ternary mode.
            seed: Random seed for reproducibility.
        """
        self.input_dims: int = input_dims
        self.sdr_dims: int = sdr_dims
        self.sparsity_k: int = sparsity_k
        self.use_ternary_projection: bool = use_ternary_projection
        self.projection_weight_density: float = projection_weight_density
        
        rng: np.random.Generator = np.random.default_rng(seed)
        
        if use_ternary_projection:
            # Create a sparse ternary matrix: values in {-1, 0, 1}
            # Probability of +1 is density/2, -1 is density/2, 0 is 1-density
            self.weights: np.ndarray = rng.choice(
                a=[-1.0, 0.0, 1.0],
                size=(input_dims, sdr_dims),
                p=[projection_weight_density / 2, 1 - projection_weight_density, projection_weight_density / 2]
            ).astype(np.float32)
        else:
            # Standard dense Gaussian projection
            self.weights = rng.standard_normal((input_dims, sdr_dims), dtype=np.float32)
            # Normalize weights to unit length for stability
            self.weights /= np.linalg.norm(self.weights, axis=0)

    def encode(self, dense_vector: np.ndarray) -> SDR:
        """
        Encodes a single dense vector into an SDR.
        
        Args:
            dense_vector: The input embedding.
            
        Returns:
            An SDR instance with exactly sparsity_k active bits.
        """
        # Project to high-dimensional space
        activations: np.ndarray = np.dot(dense_vector, self.weights)
        
        # Top-K selection: find indices of the sparsity_k largest activations
        top_k_indices: np.ndarray = np.argpartition(activations, -self.sparsity_k)[-self.sparsity_k:]
        
        return SDR(self.sdr_dims, top_k_indices)

    def encode_batch(self, dense_vectors: np.ndarray) -> list[SDR]:
        """
        Encodes a batch of dense vectors into SDRs.
        
        Args:
            dense_vectors: A 2D numpy array of embeddings.
            
        Returns:
            A list of SDR instances.
        """
        # Project all vectors
        batch_activations: np.ndarray = np.dot(dense_vectors, self.weights)
        
        sdrs: list[SDR] = []
        for i in range(batch_activations.shape[0]):
            activations: np.ndarray = batch_activations[i]
            top_k_indices: np.ndarray = np.argpartition(activations, -self.sparsity_k)[-self.sparsity_k:]
            sdrs.append(SDR(self.sdr_dims, top_k_indices))
            
        return sdrs
