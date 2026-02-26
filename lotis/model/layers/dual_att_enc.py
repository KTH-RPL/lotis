import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# from xformers.components.attention import AttentionConfig
# from xformers.ops.fmha.attn_bias import LocalAttentionFromBottomRightMask
from .layer_scale import LayerScale
import torch
import gc
from .drop_path import DropPath
from .njt_utils.nested_metadata import NestedTensorMetadata
import functools
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import CheckpointPolicy
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

def print_largest_tensors():
    # First collect all tensors
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # Get the size in megabytes
                size_mb = obj.element_size() * obj.nelement() / 1024 / 1024
                tensors.append((size_mb, obj))
        except:
            pass
    
    # Sort by size (largest first)
    tensors.sort(key=lambda x: x[0], reverse=True)
    
    # Print the top 10 tensors by size
    print("Top 10 tensors by memory usage:")
    print("-------------------------------")
    for i, (size_mb, tensor) in enumerate(tensors[:10]):
        print(f"{i+1}. Size: {size_mb:.2f} MB, Shape: {tensor.shape}, Dtype: {tensor.dtype}")
        
    # Calculate total memory
    total_memory = sum(size for size, _ in tensors)
    print(f"\nTotal memory used by tensors: {total_memory:.2f} MB")

# Call this function when you want to check memory usage
# print_largest_tensors()

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
        use_qk_norm=True,
        layernorm = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.use_qk_norm = use_qk_norm

        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        # bfloat16
        self.q_norm = layernorm(self.E_head) if use_qk_norm else None
        self.k_norm = layernorm(self.E_head) if use_qk_norm else None
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
        rope=None,
        min_seq_len_q: Optional[int] = None,
        max_seq_len_q: Optional[int] = None,
        min_seq_len_k: Optional[int] = None,
        max_seq_len_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        if min_seq_len_q is not None and min_seq_len_k is None:
            min_seq_len_k = min_seq_len_q
        if max_seq_len_q is not None and max_seq_len_k is None:
            max_seq_len_k = max_seq_len_q            
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head])
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head])
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head])

        # NOTE We moved the transpose to after the normalization step to make handling the nested tensors easier
        # Apply QK norm optionally
        if self.use_qk_norm:
            # Normalize along the head dimension
            if query.is_nested:
                # Workaround to handle nested tensors for which norm doesn't seem implemented yet
                # Unpack the nested tensor
                q_values = query.values()
                k_values = key.values()

                # Normalize the dense values tensor
                normalized_q_values = self.q_norm(q_values)
                normalized_k_values = self.k_norm(k_values)

                # Repack into a NestedTensor, preserving the original structure (pre-transpose)
                query = torch.nested.nested_tensor_from_jagged(normalized_q_values,
                                                            query._offsets,
                                                            min_seqlen=min_seq_len_q,
                                                            max_seqlen=max_seq_len_q,
                                                            )
                key = torch.nested.nested_tensor_from_jagged(normalized_k_values,
                                                            key._offsets,
                                                            min_seqlen=min_seq_len_k,
                                                            max_seqlen=max_seq_len_k,
                                                            )
            else:
                # For regular (non-nested) tensors, the direct operation is fine
                query = self.q_norm(query)
                key = self.k_norm(key)
        
        query, key = rope(query, key) if rope is not None else (query, key)

        # Step 3. Run SDPA
        
        # (N, nheads, L_t, E_head)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # print("Pre: ", attn_mask.shape if attn_mask is not None else "No mask")
        if attn_mask is not None:
            if len(attn_mask.shape) < len(query.shape):
                # If the mask doesn't fit, we likely need the head dim
                attn_mask = attn_mask.unsqueeze(1).expand(
                    attn_mask.shape[0], self.nheads, attn_mask.shape[1]
                ).unsqueeze(-2)  # Expand to match the head dimension
        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            # print("Post: ", attn_mask.shape if attn_mask is not None else "No mask")
            # print(query.shape, key.shape, value.shape)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal,
            attn_mask=~attn_mask if attn_mask is not None else None # TODO Dirty hack, we have them mixed up here. They are all negative but should be positive xd
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)
        # torch._dynamo.graph_break()
        return attn_output

class DualAttentionEncoderBlock(nn.Module):
    """
    Custom encoder block that provides fine control over positional encoding application.
    Sequentially applies:
    1. Spatial (frame-wise) attention
    2. Global attention over all tokens (seq + patches)
    
    This matches the original model's approach but with better position encoding control.
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        mlp_ratio: int = 3,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        droppath: float = 0.1,
        norm_layer = None,
        apply_spatial_rope: bool = True,
        apply_temporal_rope: bool = True,
        full_global_attention: bool = True,
        attention_stride: int = 1,
        spatial_attention_window: int = 5,
        use_nested_tensor: bool = False,
        skip_global: bool = False,
        skip_spatial: bool = False,
        compile: bool = False,
        use_checkpointing: bool = False,
    ): 
        super().__init__()
        self.skip_global = skip_global
        self.skip_spatial = skip_spatial
        if skip_global and skip_spatial:
            raise ValueError("Both global and spatial attention cannot be skipped.")
        # Only needed if not using full global attention
        self.use_checkpointing = use_checkpointing
        self.full_global_attention = full_global_attention
        self.attention_stride = attention_stride
        self.spatial_attention_window = spatial_attention_window
        self.use_nested_tensor = use_nested_tensor
        
        # Model parameters
        self.dim = dim
        self.num_heads = num_heads
        self.apply_spatial_rope = apply_spatial_rope
        self.apply_temporal_rope = apply_temporal_rope
        
        # Normalization layers
        self.norm1_spatial = norm_layer(dim, dtype=torch.float32) # float32
        self.norm2_spatial = norm_layer(dim, dtype=torch.float32) # float32
        self.norm1_global = norm_layer(dim, dtype=torch.float32) if not skip_global else nn.Identity() # float32
        self.norm2_global = norm_layer(dim, dtype=torch.float32) if not skip_global else nn.Identity() # float32

        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)
        self.ls3 = LayerScale(dim) if not skip_global else nn.Identity()
        self.ls4 = LayerScale(dim) if not skip_global else nn.Identity()

        self.drop_path1 = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.drop_path2 = DropPath(droppath) if droppath > 0. else nn.Identity()
        
        # Multi-head attention modules
        # if not self.use_nested_tensor:
        #     if not self.skip_spatial:
        #         self.spatial_attn = nn.MultiheadAttention(
        #             embed_dim=dim,
        #             num_heads=num_heads,
        #             dropout=attention_dropout,
        #             add_bias_kv=True,
        #             batch_first=True
        #         )
        #     if not self.skip_global:
        #         self.global_attn = nn.MultiheadAttention(
        #             embed_dim=dim,
        #             num_heads=num_heads,
        #             dropout=attention_dropout,
        #             add_bias_kv=True,
        #             batch_first=True
        #         )
        # else:
        if not self.skip_spatial:
            self.spatial_attn = torch.compile(MultiHeadAttention(
                E_q=dim,
                E_k=dim,
                E_v=dim,
                E_total=dim,
                nheads=num_heads,
                dropout=attention_dropout,
                bias=True,
                layernorm=norm_layer,
            ), dynamic=True, 
            disable=True) 
        if not self.skip_global:
            self.global_attn = torch.compile(MultiHeadAttention(
                E_q=dim,
                E_k=dim,
                E_v=dim,
                E_total=dim,
                nheads=num_heads,
                dropout=attention_dropout,
                layernorm=norm_layer,
                bias=True
            ), dynamic=True,
            disable=True)
        if not self.full_global_attention:
            raise NotImplementedError("Sliding window attention not implemented yet")
        # MLP blocks
        if not self.skip_spatial:
            self.spatial_mlp = nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * mlp_ratio, dim),
                nn.Dropout(dropout)
            )
        if not self.skip_global:
            self.global_mlp = nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * mlp_ratio, dim),
                nn.Dropout(dropout)
            )
        
        
        
    def _spatial_attention(
        self, 
        x: torch.Tensor, 
        split_rope: Optional[nn.Module] = None,
        spatial_pos: Optional[torch.Tensor] = None, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply spatial (frame-wise) attention to the input tensor.
        
        Args:
            x: Input tensor with shape [B*S, P, C]
            split_rope: Optional SplitDimensionRoPE module
            spatial_pos: Spatial positions with shape [B*S, P, 2]
            attn_mask: Optional attention mask
            
        Returns:
            Processed tensor with same shape as input
        """
        # Apply layer normalization
        x_norm = self.norm1_spatial(x)
        
        # Base queries, keys, values
        q, k, v = x_norm, x_norm, x_norm

        rope_spatial = functools.partial(
            split_rope, 
            spatial_pos=spatial_pos, 
            temporal_indices=None
        ) if split_rope is not None and self.apply_spatial_rope else None
        # Apply attention
        if self.use_nested_tensor:
            attn_output = self.spatial_attn(
                query=q,
                key=k,
                value=v,
                rope=rope_spatial,
            )
        else:
            attn_output = self.spatial_attn(
                query=q,
                key=k,
                value=v,
                rope=rope_spatial,
                # attn_mask=attn_mask,
            )
        attn_output = self.ls1(attn_output)
        
        if self.training:
            # Add residual connection
            x = x + self.drop_path1(attn_output)
            # Apply MLP with residual connection
            x = x + self.drop_path1(self.ls2(self.spatial_mlp(self.norm2_spatial(x))))
        else:
            x = x + attn_output
            x = x + self.ls2(self.spatial_mlp(self.norm2_spatial(x)))

        return x
    
    def _global_attention(
        self, 
        x: torch.Tensor, 
        split_rope: Optional[nn.Module] = None,
        temporal_indices: Optional[torch.Tensor] = None,
        spatial_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        nested_metadata: Optional[NestedTensorMetadata] = None,
    ) -> torch.Tensor:
        """
        Apply global attention to all tokens in the sequence.
        
        Args:
            x: Input tensor with shape [B, S, P, C] - all tokens flattened
            split_rope: Optional SplitDimensionRoPE module
            temporal_indices: Temporal indices for position encoding
            spatial_pos: Spatial positions for position encoding
            attn_mask: Optional attention mask
            
        Returns:
            Processed tensor with same shape as input
        """
        if not self.full_global_attention:
            raise NotImplementedError("Sliding window attention not implemented yet")
        # Apply layer normalization
        x = self.norm1_global(x)
        B, S, P, C = x.shape
        # Base queries, keys, values
        # q, k, v = x, x, x

        if (nested_metadata.min_seq_len is None or nested_metadata.max_seq_len is None) and self.use_nested_tensor:
            raise ValueError("min_seq_len and max_seq_len must be provided for nested tensors.")
        
        min_seq_len = nested_metadata.min_seq_len if self.use_nested_tensor else S
        max_seq_len = nested_metadata.max_seq_len if self.use_nested_tensor else S
        min_seq_len_patched = nested_metadata.min_seq_len_patched if self.use_nested_tensor else S * P
        max_seq_len_patched = nested_metadata.max_seq_len_patched if self.use_nested_tensor else S * P

        
        # if q.shape != x.shape:
        #     # There is no way this happens, right?
        #     # q = q.view(B, S, P, C)
        #     # k = k.view(B, S, P, C)
        #     # v = v.view(B, S, P, C)  

        if self.use_nested_tensor:
            # torch._dynamo.graph_break()

            x_repacked = torch.nested.nested_tensor_from_jagged(
                x.values().view(-1, C),  # This introduces some problem and the graph break is needed
                nested_metadata.patched_offsets,
                min_seqlen=min_seq_len_patched,
                max_seqlen=max_seq_len_patched
            )
            # We neeed to make x a dense tensor and then merge S and P, like this
            # min_seq_len = x._get_min_seqlen()
            # max_seq_len = x._get_max_seqlen()
            # length = original_offsets[-1]
            
            q, k, v = x_repacked, x_repacked, x_repacked


            # This rope is supposed to be used inside the attention module, where the query and key tensors are reshaped
            # to [B, S*P, C] and then split into heads yielding [B, S*P, ...].
            # We will need to define and pass the necessary reshape functions for the nested tensors here.

            # [B, S*P, ...] -> [B * S, P, ...], this packs it for spatial RoPE
            unpack_fun = lambda x: x.values().view(-1, P, self.num_heads, C // self.num_heads)
            # [B * S, P, ...] -> [B, S, P, ...], this packs it for global RoPE
            intermediate_reshape = lambda x: torch.nested.nested_tensor_from_jagged(
                x,
                nested_metadata.offsets,
                min_seqlen=min_seq_len,
                max_seqlen=max_seq_len
                )

            # [B, S, P, ...] -> [B, S*P, ...], packing the S and P dimensions, since attention wants this.
            pack_fun  = lambda x: torch.nested.nested_tensor_from_jagged(x.values().view(-1, self.num_heads, C // self.num_heads),
                                                                        nested_metadata.patched_offsets,
                                                                        min_seqlen=min_seq_len_patched,
                                                                        max_seqlen=max_seq_len_patched
                )
            
            # Define the rope function to apply the RoPE to the global attention
            rope_global = functools.partial(
                split_rope,
                spatial_pos=spatial_pos,
                temporal_indices=temporal_indices,
                reshape_functions=(unpack_fun, intermediate_reshape, pack_fun),
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len
                ) if split_rope is not None and self.apply_temporal_rope and temporal_indices is not None else None
            attn_output = self.global_attn(
                query=q,
                key=k,
                value=v,
                rope=rope_global,
                min_seq_len_q=min_seq_len_patched,
                max_seq_len_q=max_seq_len_patched
            )

            # and back to patched version
            attn_output = torch.nested.nested_tensor_from_jagged(attn_output.values().view(-1, P, C),
                                                                nested_metadata.offsets,
                                                                min_seqlen=min_seq_len,
                                                                max_seqlen=max_seq_len
                                                                )
        else:
            x_shaped = x.view(B, S*P, C)
            q, k, v = x_shaped, x_shaped, x_shaped
            # In this case we "just" need to merge the sequence and patch dimensions.
            # Need to be careful about the global mask here, this needs to be correctly reshaped.
            # q = q
            # k = k.view(B, S*P, C)
            # v = v.view(B, S*P, C)

            # We want to apply rope again in the attention, and it wants S, P
            # [B, S*P, ...] -> [B * S, P, ...], unpacking the S and P dimensions
            unpack_fun = lambda x: x.view(B * S, P, self.num_heads, C // self.num_heads)
            # [B * S, P, ...] -> [B, S, P, ...], this packs it for global RoPE
            intermediate_reshape = lambda x: x.view(B, S, P, self.num_heads, -1)
            # [B, S, P, ...] -> [B, S*P, ...], packing the S and P dimensions, since attention wants this.
            pack_fun  = lambda x: x.view(B, S*P, self.num_heads, C // self.num_heads)
            rope = functools.partial(
                split_rope,
                spatial_pos=spatial_pos,
                temporal_indices=temporal_indices,
                reshape_functions=(unpack_fun, intermediate_reshape, pack_fun),
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len
            ) if split_rope is not None and self.apply_temporal_rope and temporal_indices is not None else None
            attn_output = self.global_attn(
                query=q,
                key=k,
                value=v,
                rope=rope,
                attn_mask=attn_mask,
                min_seq_len_q=min_seq_len * P,
                max_seq_len_q=max_seq_len * P
            )
            attn_output = attn_output.view(B, S, P, C)
        attn_output = self.ls3(attn_output)
        if self.training:
            # Add residual connection
            x = x + self.drop_path1(attn_output)
            # Apply MLP with residual connection
            x = x + self.drop_path1(self.ls4(self.global_mlp(self.norm2_global(x))))
        else:
            x = x + attn_output
            x = x + self.ls4(self.global_mlp(self.norm2_global(x)))
        return x
    
    def forward(self, 
        x: torch.Tensor, 
        split_rope: Optional[nn.Module] = None,
        spatial_pos: Optional[torch.Tensor] = None,
        temporal_indices: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
        global_mask: Optional[torch.Tensor] = None,
        nested_metadata: Optional[NestedTensorMetadata] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional gradient checkpointing.
        """
        compute_intensive_ops = [   torch.ops.aten._scaled_dot_product_flash_attention,]
        def policy_fn(ctx, op, *args, **kwargs):
            if op in compute_intensive_ops:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE
        if self.training and self.use_checkpointing:
            return checkpoint(
                self._forward,
                x,
                split_rope,
                spatial_pos,
                temporal_indices,
                spatial_mask,
                global_mask,
                nested_metadata,
                use_reentrant=False,
                context_fn=functools.partial(create_selective_checkpoint_contexts, policy_fn),
            )
        else:
            return self._forward(
                x,
                split_rope,
                spatial_pos,
                temporal_indices,
                spatial_mask,
                global_mask,
                nested_metadata
            )
    
    def _forward(
        self, 
        x: torch.Tensor, 
        split_rope: Optional[nn.Module] = None,
        spatial_pos: Optional[torch.Tensor] = None,
        temporal_indices: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
        global_mask: Optional[torch.Tensor] = None,
        nested_metadata: Optional[NestedTensorMetadata] = None,
    ) -> torch.Tensor:
        """
        Forward pass with sequential spatial and global attention.
        
        Args:
            x: Input tensor with shape [B, S, P, C]
            split_rope: Optional SplitDimensionRoPE module
            spatial_pos: Spatial positions tensor
            temporal_indices: Temporal position indices
            spatial_mask: Optional mask for spatial attention 
            global_mask: Optional mask for global attention
        Returns:
            Processed tensor with shape [B, S, P, C]
        """
        # pass_type = "RECOMPUTED (BACKWARD)" if torch.is_grad_enabled() else "ORIGINAL (FORWARD)"
        
        # if self.use_nested_tensor and x.is_nested:
            # print(f"Input x.shape: {x.shape}")
        B, S, P, C = x.shape
        if temporal_indices is not None and self.use_nested_tensor and (nested_metadata is None or nested_metadata.min_seq_len is None or nested_metadata.max_seq_len is None):
            # We want global attention, we want nested, but we lack information.
            raise ValueError("min_seq_len and max_seq_len must be provided for nested tensors.")
        
        # seq_len_sum = nested_metadata.offsets[-1] if self.use_nested_tensor else B * S
        offsets = nested_metadata.offsets if self.use_nested_tensor else None

        # 1. Spatial attention (within each frame)
        if not self.skip_spatial:
            # [B*S, P, C] Reshape to process each frame independently
            x = x.values() if self.use_nested_tensor else x.view(-1, P, C)  

            # Apply spatial attention
            x = self._spatial_attention(
                x=x,
                split_rope=split_rope,
                spatial_pos=spatial_pos,
                # attn_mask=spatial_mask
            )

            # [B, S, P, C] Reshape back. 
            if self.use_nested_tensor:
                x = torch.nested.nested_tensor_from_jagged(x, offsets, min_seqlen=nested_metadata.min_seq_len, max_seqlen=nested_metadata.max_seq_len) if self.use_nested_tensor else x.view(B, S, P, C)
            else:
                x = x.view(B, S, P, C)
            # if self.use_nested_tensor and x.is_nested:
                # print(f"Post spatial reshape x.shape: {x.shape}")
        # 2. Global attention (across sequences and patches)
        if not self.skip_global:
            # x = x.view(B, S, P, C)

            # Apply global attention
            x = self._global_attention(
                x=x,
                split_rope=split_rope,
                temporal_indices=temporal_indices,
                spatial_pos=spatial_pos,
                # attn_mask=global_mask,
                nested_metadata=nested_metadata,
            )

            # Reshape back to [B, S, P, C]
            x = x.view(B, S, P, C)
        return x
