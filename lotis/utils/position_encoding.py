import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple


class PositionGetter:
    """
    Generates and caches 2D spatial positions for patches in a grid.
    """
    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache = {}
        
    def __call__(self, batch_size, height, width, device):
        """
        Generates spatial positions for a batch of patches.
        
        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.
            
        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[(height, width)] = positions
            
        cached_positions = self.position_cache[(height, width)]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1)


class RotaryPositionEmbedding(nn.Module):
    """
    2D Rotary Position Embedding implementation for spatial positions.
    
    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.
    """
    def __init__(self, dim, grid_size=(7, 7), frequency=100.0, scaling_factor=1.0):
        super().__init__()
        assert dim % 4 == 0, "Feature dimension must be divisible by 4 for 2D RoPE"
        
        self.dim = dim
        self.grid_size = grid_size
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        
        # Cache for frequency components
        self.frequency_cache = {}
        
        # Position generator
        self.position_getter = PositionGetter()

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)
            
    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        """
        Computes frequency components for rotary embeddings.
        
        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.
            
        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            
            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            # angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = positions[:, None] * inv_freq[None, :]
            
            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)
            
        return self.frequency_cache[cache_key]
    
    @staticmethod
    def _rotate_features(x):
        """
        Performs feature rotation by splitting and recombining feature dimensions.
        
        Args:
            x: Input tensor to rotate.
            
        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., :feature_dim//2], x[..., feature_dim//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x, positions):
        """
        Applies 2D rotary position embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_patches, dim]
                where num_patches = grid_height * grid_width
                
        Returns:
            Tensor with rotary position embeddings applied
        """
        feature_dim = x.size(-1) // 2

        # Compute frequency components (for max position in the grid)
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, x.device, x.dtype
        )
        
        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = x.chunk(2, dim=-1)
        
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp)
        # Combine vertical and horizontal features
        x_rotated = torch.cat((vertical_features, horizontal_features), dim=-1)
        
        return x_rotated


class SequencePositionalEncoding(nn.Module):
    """
    Rotary positional encoding for sequence positions.
    
    This implementation uses rotation-based encoding (RoPE) for sequential positions,
    adapting the approach from the 2D RoPE implementation.
    """
    def __init__(self, d_model, max_len=100, base_frequency=10000.0):
        super().__init__()
        # Ensure dimension is divisible by 2 for rotary embeddings
        assert d_model % 2 == 0, "Feature dimension must be divisible by 2 for 1D RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base_frequency = base_frequency
        
        # Cache for frequency components
        self.frequency_cache = {}
        
    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        """
        Computes frequency components for rotary embeddings.
        
        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.
            
        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            
            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            # angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = positions[:, None] * inv_freq[None, :]
            
            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)
            
        return self.frequency_cache[cache_key]
    
    @staticmethod
    def _rotate_features(x):
        """Rotates half the feature dimensions of x."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        """
        Applies 1D rotary position embeddings.

        Args:
            tokens: Input token features [batch_size, seq_len, p, dim]
            positions: Position indices [seq_len]
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with rotary position embeddings applied.
        """
        # Get cosine and sine embeddings for each position
        cos = cos_comp[positions]  # [seq_len, dim]
        sin = sin_comp[positions]  # [seq_len, dim]
        
        # Apply rotation: (x_i * cos_θ) + (rotate(x_i) * sin_θ)
        return (tokens * cos.unsqueeze(0).unsqueeze(2)) + (self._rotate_features(tokens) * sin.unsqueeze(0).unsqueeze(2))
    
    def forward(self, x):
        """
        Applies rotary positional encoding to input tensor along sequence dimension.
        
        Args:
            x: Input tensor of shape 
               [B, seq_len, patches, dim]
            
        Returns:
            Tensor with rotary positional encoding applied
        """
        # Get sequence length (assuming it's the second-to-last dimension for [..., S, P, D] 
        # or second dimension for [B, S, D])
        if len(x.shape) >= 3 :
            # Case: [..., seq_len, patches, dim]
            orig_shape = x.shape
            batch_size, seq_len, patches, feat_dim = orig_shape
            seq_dim = len(orig_shape) - 3
            seq_len = orig_shape[seq_dim]

            # Process each sequence position
            positions = torch.arange(seq_len, device=x.device)
            cos_comp, sin_comp = self._compute_frequency_components(
                self.d_model, self.max_len, x.device, x.dtype
            )
            
            # Apply RoPE
            x = x.view(batch_size, seq_len, patches, feat_dim)
            x_rotated = self._apply_1d_rope(x, positions, cos_comp, sin_comp)
            return x_rotated.view(orig_shape)
        else:
            # Simple case: [batch, seq, dim]
            seq_len = x.shape[1]
            positions = torch.arange(seq_len, device=x.device)
            
            # Get frequency components
            cos_comp, sin_comp = self._compute_frequency_components(
                self.d_model, self.max_len, x.device, x.dtype
            )
            
            # Apply rotary embeddings directly
            return self._apply_1d_rope(x, positions, cos_comp, sin_comp)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence positions.
    DEPRECATED: Use SequencePositionalEncoding instead.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Calculate sinusoidal pattern
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter but should be saved with model)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
        
    def forward(self, x):
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x