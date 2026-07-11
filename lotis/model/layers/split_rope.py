import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
import time

USE_EINSUM = False
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


class SplitDimensionRoPE(nn.Module):
    """
    Applies rotary position embeddings separately to spatial and temporal dimensions.
    Splits the embedding dimension in half, dedicating one half to spatial positions and 
    the other half to temporal positions.
    """
    def __init__(self, spatial_grid_size=(7, 7), max_seq_len=40, 
                 spatial_base_freq=100.0, temporal_base_freq=100.0,
                 rope_jitter=False,
                 rope_jitter_range=2.0,
                 inference_scale=1.0
                 ):
        super().__init__()
        self.rope_jitter = rope_jitter
        self.rope_jitter_range = rope_jitter_range
        self.inference_scale = inference_scale
        # Create dedicated RoPE modules for each dimension half
        self.spatial_rope = RotaryEmbeddingHalf(
            grid_size=spatial_grid_size, 
            frequency=spatial_base_freq
        )
        
        self.temporal_rope = TemporalEmbeddingHalf(
            max_len=max_seq_len, 
            base_frequency=temporal_base_freq,
            rope_jitter=rope_jitter,
            rope_jitter_range=rope_jitter_range,
            inference_scale=inference_scale
        )
        
        # Position getter for convenience
        self.position_getter = PositionGetter()
        
    def forward(self, q, k, spatial_pos=None, temporal_indices=None, reshape_functions=None, reshape_functions_k=None, min_seq_len=None, max_seq_len=None):
        """
        Applies split dimension RoPE to query and key tensors.
        
        Args:
            q: Query tensor [B, S, P, D] or [B, L, D]
            k: Key tensor [B, S, P, D] or [B, L, D]
            spatial_pos: Spatial positions [B*S, P, 2] or [B, L, 2]
            temporal_indices: Tensor or range of temporal indices
            reshape_functions: Optional functions to reshape the input tensors before applying RoPE.
        Returns:
            q_pos: Query tensor with position encodings applied
            k_pos: Key tensor with position encodings applied
        """
        # We want to apply rope always in float32, so we cast the inputs to float32.
        original_dtype = q.dtype
        is_using_nested = q.is_nested
        # q = q
        # k = k
        if reshape_functions is not None:
            pre_reshape_q, intermediate_reshape_q, post_reshape_q = reshape_functions
            if reshape_functions_k is not None:
                pre_reshape_k, intermediate_reshape_k, post_reshape_k = reshape_functions_k
            else:
                pre_reshape_k, intermediate_reshape_k, post_reshape_k = pre_reshape_q, intermediate_reshape_q, post_reshape_q
            q = pre_reshape_q(q)
            k = pre_reshape_k(k)
        else:
            pre_reshape_q, intermediate_reshape_q, post_reshape_q = None, None, None
            pre_reshape_k, intermediate_reshape_k, post_reshape_k = None, None, None
        feature_dim = q.shape[-1]
        assert feature_dim == k.shape[-1], "Query and key must have the same feature dimension"
        
        

        # Split query and key along feature dimension
        if spatial_pos is None and temporal_indices is None:
            raise ValueError("At least one of spatial_pos or temporal_indices must be provided.")
        if spatial_pos is not None and temporal_indices is None:
            q_spatial, q_temporal = q, None
            k_spatial, k_temporal = k, None
        elif spatial_pos is None and temporal_indices is not None:
            q_spatial, q_temporal = None, q
            k_spatial, k_temporal = None, k
        else:
            spatial_size = feature_dim * 3 // 4
            temporal_size = feature_dim - spatial_size
            q_spatial, q_temporal = torch.split(q, (spatial_size, temporal_size), dim=-1)
            k_spatial, k_temporal = torch.split(k, (spatial_size, temporal_size), dim=-1)
        # Apply spatial RoPE if spatial positions are provided
        if spatial_pos is not None:
            q_spatial = self.spatial_rope(q_spatial, positions=spatial_pos)
            k_spatial = self.spatial_rope(k_spatial, positions=spatial_pos)

        # Prepare for global RoPE, we need to apply the function to both spatial and temporal parts.
        if intermediate_reshape_q is not None:
            if q_temporal is not None:
                q_temporal = intermediate_reshape_q(q_temporal)
                k_temporal = intermediate_reshape_q(k_temporal)
            if q_spatial is not None:
                q_spatial = intermediate_reshape_q(q_spatial)
                k_spatial = intermediate_reshape_k(k_spatial)

        # Apply temporal RoPE if temporal indices are provided
        if temporal_indices is not None:
            q_temporal = self.temporal_rope(q_temporal, is_using_nested, 
                                            min_seq_len=min_seq_len, 
                                            max_seq_len=max_seq_len)
            k_temporal = self.temporal_rope(k_temporal, is_using_nested,
                                            min_seq_len=min_seq_len, 
                                            max_seq_len=max_seq_len)
        
        # Recombine the spatial and temporal parts
        if q_temporal is None:
            q_pos = q_spatial
            k_pos = k_spatial
        elif q_spatial is None:
            q_pos = q_temporal
            k_pos = k_temporal
        else:
            # Concatenate along the last dimension
            q_pos = torch.cat([q_spatial, q_temporal], dim=-1)
            k_pos = torch.cat([k_spatial, k_temporal], dim=-1)
        if post_reshape_q is not None:
            q_pos = post_reshape_q(q_pos)
            k_pos = post_reshape_k(k_pos)
        # Restore original dtype
        # q_pos = q_pos.to(original_dtype)
        # k_pos = k_pos.to(original_dtype)
        return q_pos, k_pos


class RotaryEmbeddingHalf(nn.Module):
    """
    2D Rotary Position Embedding implementation for spatial positions.
    Simplified version that works directly with half the input dimension.
    """
    def __init__(self, grid_size=(7, 7), frequency=100.0):
        super().__init__()
        
        self.grid_size = grid_size
        self.base_frequency = frequency
        
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
            angles = torch.einsum("i,j->ij", positions, inv_freq) if USE_EINSUM else positions[:, None] * inv_freq[None, :]
            # breakpoint()
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
    
    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        """Applies 1D rotary position embeddings along one dimension."""
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)
        sin = F.embedding(positions, sin_comp)
        if len(tokens.shape) == 4:
            # For 4D tensors, we need to unsqueeze cos and sin to match the head dim.
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
        # Apply rotation
        # print(tokens.shape, cos.shape, sin.shape)
        return (tokens * cos) + (self._rotate_features(tokens) * sin)
    
    def forward(self, x, positions):
        """
        Applies 2D rotary position embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_patches, dim/2]
            positions: Spatial positions [B*S, P, 2] or [B, L, 2]
                
        Returns:
            Tensor with rotary position embeddings applied
        """
        feature_dim = x.size(-1)
        assert feature_dim % 2 == 0, "Feature dimension must be divisible by 2 for RoPE"

        # Compute frequency components (for max position in the grid)
        max_position = 40#int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim // 2, max_position, x.device, x.dtype
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


class TemporalEmbeddingHalf(nn.Module):
    """
    Rotary positional encoding for sequence positions.
    Simplified version that works directly with half the input dimension.
    """
    def __init__(self, max_len=100, base_frequency=100.0,
                 rope_jitter=False,
                 rope_jitter_range=2.0,
                 inference_scale=1.0
                 ):
        super().__init__()
        self.rope_jitter = rope_jitter
        self.rope_jitter_range = rope_jitter_range
        self.inference_scale = inference_scale
        
        self.max_len = max_len
        self.base_frequency = base_frequency
        
        # Cache for frequency components
        self.frequency_cache = {}
        
    def _compute_frequency_components_explicit(self, dim, positions, device, dtype):
        """
        Computes frequency components for rotary embeddings given explicit positions.
        
        Args:
            dim: Feature dimension (must be even).
            positions: Explicit position indices tensor.
            device: Target device for computations.
            dtype: Data type for the computed tensors.
        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        # Compute frequency bands
        exponents = torch.arange(0, dim, 2, device=device).float() / dim
        inv_freq = 1.0 / (self.base_frequency ** exponents)
        
        # Generate position-dependent frequencies
        angles = torch.einsum("i,j->ij", positions, inv_freq) if USE_EINSUM else positions[:, None] * inv_freq[None, :]

        # Compute frequency components
        angles = angles.to(dtype)
        angles = torch.cat((angles, angles), dim=-1)
        cos_components = angles.cos().to(dtype)
        sin_components = angles.sin().to(dtype)
        
        return cos_components, sin_components

    def _get_effective_base(self, dim):
        """Compute NTK-aware base frequency."""
        if self.inference_scale == 1.0:
            return self.base_frequency
        # NTK formula
        return self.base_frequency * (self.inference_scale ** (dim / (dim - 2)))

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
            effective_base = self._get_effective_base(dim)
            inv_freq = 1.0 / (effective_base ** exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq) if USE_EINSUM else positions[:, None] * inv_freq[None, :]

            
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
    
    def _apply_1d_rope_no_embedding(self, tokens, cos, sin, is_using_nested=False, min_seq_len=None, max_seq_len=None):
        if (min_seq_len is None or max_seq_len is None) and is_using_nested:
            raise ValueError("min_seq_len and max_seq_len must be provided for nested tensors.")

        # Apply rotation: (x_i * cos_θ) + (rotate(x_i) * sin_θ)
        if is_using_nested:
            # min_seq_len = tokens._get_min_seqlen()
            # max_seq_len = tokens._get_max_seqlen()
            # We will reshape the cos and sin tensors to match the nested tensor.
            cos = torch.nested.nested_tensor_from_jagged(cos,
                                                        tokens._offsets,
                                                        min_seqlen=min_seq_len,
                                                        max_seqlen=max_seq_len
                                                        ).unsqueeze(2)
            sin = torch.nested.nested_tensor_from_jagged(sin,
                                                        tokens._offsets,
                                                        min_seqlen=min_seq_len,
                                                        max_seqlen=max_seq_len
                                                        ).unsqueeze(2)
            if len(tokens.shape) == 5: # We have head dims
                cos = cos.unsqueeze(3)
                sin = sin.unsqueeze(3)
            # The batch dimension is explicit in thise case.
            return (tokens * cos) + (self._rotate_features(tokens) * sin)
        else:
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, 16]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, 16]
            if len(tokens.shape) == 5:  # We have head dims
                cos = cos.unsqueeze(3)  # [1, seq_len, 1, 1, 16]
                sin = sin.unsqueeze(3)  # [1, seq_len, 1, 1, 16]

            return (tokens * cos) + (self._rotate_features(tokens) * sin)
    
    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp, is_using_nested=False, min_seq_len=None, max_seq_len=None):
        """
        Applies 1D rotary position embeddings.

        Args:
            tokens: Input token features [batch_size, seq_len, p, dim/2]
            positions: Position indices [seq_len]
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with rotary position embeddings applied.
        """
        if (min_seq_len is None or max_seq_len is None) and is_using_nested:
            raise ValueError("min_seq_len and max_seq_len must be provided for nested tensors.")
        # Get cosine and sine embeddings for each position
        cos = cos_comp[positions]  # [seq_len], or [ sum(lengths)] if nested
        sin = sin_comp[positions]  # [seq_len], or [ sum(lengths)] if nested

        # Apply rotation: (x_i * cos_θ) + (rotate(x_i) * sin_θ)
        if is_using_nested:
            # min_seq_len = tokens._get_min_seqlen()
            # max_seq_len = tokens._get_max_seqlen()
            # We will reshape the cos and sin tensors to match the nested tensor.
            cos = torch.nested.nested_tensor_from_jagged(cos,
                                                        tokens._offsets,
                                                        min_seqlen=min_seq_len,
                                                        max_seqlen=max_seq_len
                                                        ).unsqueeze(2)
            sin = torch.nested.nested_tensor_from_jagged(sin,
                                                        tokens._offsets,
                                                        min_seqlen=min_seq_len,
                                                        max_seqlen=max_seq_len
                                                        ).unsqueeze(2)
            if len(tokens.shape) == 5: # We have head dims
                cos = cos.unsqueeze(3)
                sin = sin.unsqueeze(3)
            # The batch dimension is explicit in thise case.
            return (tokens * cos) + (self._rotate_features(tokens) * sin)
        else:
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, 16]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, 16]
            if len(tokens.shape) == 5:  # We have head dims
                cos = cos.unsqueeze(3)  # [1, seq_len, 1, 1, 16]
                sin = sin.unsqueeze(3)  # [1, seq_len, 1, 1, 16]

            return (tokens * cos) + (self._rotate_features(tokens) * sin)
    
    def forward(self, x, is_using_nested=False, min_seq_len=None, max_seq_len=None):
        """
        Applies rotary positional encoding to input tensor along sequence dimension.
        
        Args:
            x: Input tensor of shape [B, S, P, dim/2] or [B, S, dim/2]
            is_using_nested: Boolean indicating if nested tensors are used.
            
        Returns:
            Tensor with rotary positional encoding applied
        """
        # Get sequence length and determine tensor format
        if is_using_nested and (min_seq_len is None or max_seq_len is None):
            raise ValueError("min_seq_len and max_seq_len must be provided for nested tensors.")
        feature_dim = x.size(-1)
        assert feature_dim % 2 == 0, "Feature dimension must be divisible by 2 for RoPE"
        if len(x.shape) >= 3:
            # Case: [..., seq_len, patches, dim]
            orig_shape = x.shape
            batch_size, seq_len = orig_shape[0], orig_shape[1]
            
            # Process each sequence position
            if not self.rope_jitter:
                # This the standard path - fetch the frequency components once, and they might be cached.
                if not is_using_nested:
                    positions = torch.arange(seq_len, device=x.device)
                    cos_comp, sin_comp = self._compute_frequency_components(
                        feature_dim, self.max_len, x.device, x.dtype
                    )
                    # Apply RoPE
                    return self._apply_1d_rope(x, positions, cos_comp, sin_comp, is_using_nested, min_seq_len, max_seq_len)
                else:
                    # In this case we need batched positions. This is slightly redundant but should be fine overall.
                    # offsets = x._offsets
                    vals = x.values()
                    positions = torch.arange(vals.shape[0], device=x.device) - torch.repeat_interleave(x._offsets[:-1], torch.diff(x._offsets), output_size=vals.shape[0])

                    cos_comp, sin_comp = self._compute_frequency_components(
                        feature_dim, self.max_len, x.device, x.dtype
                    )
                    # Apply RoPE
                    return self._apply_1d_rope(x, positions, cos_comp, sin_comp, is_using_nested, min_seq_len, max_seq_len)
            else:
                # We follow DINOv3's rope jitter - https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py
                # In essence this does the following:
                # 1. Rescale coordinates to range [-1, 1]
                # 2. Multiply the [-1, 1] range by a log-uniform value in [1/rope_jitter_range, rope_jitter_range]
                if not is_using_nested:
                    raise NotImplementedError("Rope jitter is only implemented for nested tensors currently.")
                else:
                    # We want the positions to be in range [-1, 1], this means we need to rescale the integer positions.
                    # Here, each sample should be rescaled independenly (not by max_seq_len).
                    vals = x.values() # (should be cheap)
                    lengths = torch.diff(x._offsets)
                    positions = torch.arange(vals.shape[0], device=x.device) - torch.repeat_interleave(x._offsets[:-1], lengths, output_size=vals.shape[0])
                    # Rescale to [-1, 1]
                    positions = positions.float()
                    positions = positions / (lengths.repeat_interleave(lengths, output_size=vals.shape[0]).float() - 1)  # Scale to [0, 1]
                    positions = 2.0 * positions - 1.0  # Scale to [-1, 1]
                    # Now, if we are training, we apply the jitter, otherwise keep the range
                    if self.training:
                        jitter_max = math.log(self.rope_jitter_range)
                        jitter_min = -jitter_max
                        jitter_vals = torch.empty(lengths.shape[0], device=x.device).uniform_(jitter_min, jitter_max).exp()
                        # Now we need to multiply each position by the corresponding jitter value
                        positions = positions * torch.repeat_interleave(jitter_vals, lengths, output_size=vals.shape[0])
                        # Compute the frequency components. We need to explictily compute them here.
                    cos, sin = self._compute_frequency_components_explicit(
                        feature_dim, positions, x.device, x.dtype
                    )
                    return self._apply_1d_rope_no_embedding(x, cos, sin, is_using_nested, min_seq_len, max_seq_len)

        else:
            raise NotImplementedError(
                "TemporalEmbeddingHalf currently only supports input tensors with at least 3 dimensions."
            )
            # Simple case: [batch, seq, dim]
            seq_len = x.shape[1]
            positions = torch.arange(seq_len, device=x.device)
            
            # Get frequency components
            cos_comp, sin_comp = self._compute_frequency_components(
                feature_dim, self.max_len, x.device, x.dtype
            )
            
            # Apply rotary embeddings directly
            return self._apply_1d_rope(x, positions, cos_comp, sin_comp)
        

# --- Main execution block ---

def main():
    """
    Main function to demonstrate and benchmark SplitDimensionRoPE with nested tensors.
    """
    # Configuration
    B = 40
    H, W = 16, 16
    P = H * W  # Number of patches
    D = 512  # Embedding dimension
    MAX_SEQ_LEN = 40
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print(f"Running on device: {DEVICE} with dtype: {DTYPE}")
    print(f"Batch={B}, Patches={P}, Dim={D}, MaxSeqLen={MAX_SEQ_LEN}")

    # 1. Instantiate the model
    model = SplitDimensionRoPE(
        spatial_grid_size=(H, W),
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE, dtype=DTYPE)

    # 2. Create nested tensor inputs q and k with variable sequence lengths
    # These lengths are deliberately chosen to be different.
    lengths = torch.tensor([10, 8, 12, 9])
    offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths), dim=0)), device=DEVICE)
    print(f"Variable sequence lengths: {lengths}")

    q = torch.randn(torch.sum(lengths), P, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(torch.sum(lengths), P, D, device=DEVICE, dtype=DTYPE)

    q_nested = torch.nested.nested_tensor_from_jagged(q, offsets=offsets)
    k_nested = torch.nested.nested_tensor_from_jagged(k, offsets=offsets)
    print(q_nested.shape, k_nested.shape)
    # 3. Prepare spatial and temporal position information
    
    # The spatial positions are the same for every frame. When we flatten the nested
    # tensor from [B, S(var), P, D] to [sum(S), P, D], we need to provide
    # spatial positions for each of the `sum(S)` frames.
    position_getter = PositionGetter()
    base_pos = position_getter(sum(lengths), H, W, DEVICE) # Shape: [1, P, 2]
    total_frames = sum(lengths)
    spatial_pos = base_pos.expand(total_frames, -1, -1) # Shape: [sum(S), P, 2]
    
    # This is just a flag to activate the temporal RoPE path in the model's forward pass.
    # The actual temporal positions are calculated inside TemporalEmbeddingHalf.
    temporal_indices_flag = 1


    print("\n--- Running in Spatial Mode ---")
    
    # Warmup
    for _ in range(5):
        _ = model(q, k, spatial_pos, None)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    q_pos_eager, k_pos_eager = model(q, k, spatial_pos, None)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    eager_time = (end_time - start_time) * 1000
    print(f"Spatial execution eager time: {eager_time:.3f} ms")
    
    # --- Compiled Mode Execution ---
    print("\n--- Running in Compiled Mode ---")
    # try:
    compiled_model = torch.compile(model)
    
    # Warmup (includes compilation time)
    for _ in range(5):
            _ = compiled_model(q, k, spatial_pos, None)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    q_pos_compiled, k_pos_compiled = compiled_model(q, k, spatial_pos, None)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    compiled_time = (end_time - start_time) * 1000
    print(f"Compiled execution time: {compiled_time:.3f} ms")

    # --- Verification ---
    print("\n--- Verifying outputs ---")
    # print(q_pos_compiled.shape)
    # assert q_pos_eager.is_nested and q_pos_compiled.is_nested
    # assert k_pos_eager.is_nested and k_pos_compiled.is_nested
    
    # Compare the underlying data buffers of the nested tensors
    q_verified = torch.allclose(q_pos_eager, q_pos_compiled, atol=1e-2, rtol=1e-2)
    k_verified = torch.allclose(k_pos_eager, k_pos_compiled, atol=1e-2, rtol=1e-2)
    
    print(f"Query outputs match: {q_verified}")
    print(f"Key outputs match: {k_verified}")
    
    if q_verified and k_verified:
        print("✅ Verification successful!")
        print(f"Speedup with torch.compile: {eager_time / compiled_time:.2f}x")
    else:
        print("❌ Verification failed!")








    print("\n--- Running temporal ---")
    # Warmup
    for _ in range(5):
        _ = model(q_nested, k_nested, None, temporal_indices_flag)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    q_pos_eager, k_pos_eager = model(q_nested, k_nested, None, temporal_indices_flag)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    eager_time = (end_time - start_time) * 1000
    print(f"Spatial execution eager time: {eager_time:.3f} ms")
    
    # --- Compiled Mode Execution ---
    print("\n--- Running in Compiled Mode ---")
    # try:
    compiled_model = torch.compile(model)
    
    # Warmup (includes compilation time)
    for _ in range(5):
            _ = compiled_model(q_nested, k_nested, None, temporal_indices_flag)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    q_pos_compiled, k_pos_compiled = compiled_model(q_nested, k_nested, None, temporal_indices_flag)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    compiled_time = (end_time - start_time) * 1000
    print(f"Compiled execution time: {compiled_time:.3f} ms")

    # --- Verification ---
    print("\n--- Verifying outputs ---")
    # print(q_pos_compiled.shape)
    # assert q_pos_eager.is_nested and q_pos_compiled.is_nested
    # assert k_pos_eager.is_nested and k_pos_compiled.is_nested
    
    # Compare the underlying data buffers of the nested tensors
    q_verified = torch.allclose(q_pos_eager.values(), q_pos_compiled.values(), atol=1e-2, rtol=1e-2)
    k_verified = torch.allclose(k_pos_eager.values(), k_pos_compiled.values(), atol=1e-2, rtol=1e-2)
    
    print(f"Query outputs match: {q_verified}")
    print(f"Key outputs match: {k_verified}")
    
    if q_verified and k_verified:
        print("✅ Verification successful!")
        print(f"Speedup with torch.compile: {eager_time / compiled_time:.2f}x")
    else:
        print("❌ Verification failed!")    


if __name__ == "__main__":
    main()
