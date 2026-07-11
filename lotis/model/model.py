import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torch.cuda.nvtx as nvtx

from ..utils.position_encoding import PositionGetter
from .layers.split_rope import SplitDimensionRoPE
from .layers.dual_att_enc import DualAttentionEncoderBlock, MultiHeadAttention
from .layers.layer_scale import LayerScale
from .layers.njt_utils.nested_metadata import NestedTensorMetadata
from .layers.njt_utils.slice_njt import slice_njt
from .layers.njt_utils.repeat_interleave_njt import repeat_nested_tensor_efficient

from .prediction_head import LoTISPredictionHead
from .layers.drop_path import DropPath
from torch.utils.checkpoint import checkpoint
import os
ONNX_EXPORT = os.getenv("ONNX_EXPORT", "0") == "1"


def slice_expand_and_flatten(token_tensor, B, BS):
    """
    See: https://github.com/facebookresearch/vggt/blob/main/vggt/models/aggregator.py#L308
    Args:
        token_tensor (torch.Tensor): Input tensor with shape (1, 2, X, C)
        B (int): Batch size
        BS (int): Batch size * Sequence lengths. We need this since we might have nested tensors

    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    # 4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
    #    followed by (S-1) second-position tokens
    # 5) Flattens to (B*S, X, C) for processing

    # Returns:
    #     torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[0, 1:, ...].expand(BS, *token_tensor.shape[2:])
    return query, others

class LoTIS(nn.Module):
    """
    Improved trajectory localization model with proper handling of position encodings.
    Uses the custom DualAttentionEncoderBlock and SplitDimensionRoPE.
    """
    def __init__(self, feature_dim=768,
                 input_patches=(8, 8),
                 hidden_dim=512,
                 num_heads=8,
                 dropout=0.1,
                 attention_dropout=0.1,
                 droppath=0.1,
                 max_seq_len=40,
                 num_blocks=3,
                 head_depth=4,
                 use_nested_tensor=False,
                 rope_freq_seq=100,
                 rope_freq_spat=500,
                 heads=["center"],
                 full_global_attention=True,
                 mini_batch_size = 8,
                 rope_jitter=False,
                 rope_jitter_range=2.0,
                 layernorm_type=None,
                 compile=False,
                 rope_inference_scale=1.0
                 ):
        super().__init__()
        # First check if layernorm is valid, must be either LayerNorm or RMSNorm
        if layernorm_type not in ["LayerNorm", "RMSNorm"]:
            raise ValueError(f"Invalid layernorm_type: {layernorm_type}. Must be 'LayerNorm' or 'RMSNorm'.")
        
        ln_t = nn.LayerNorm if layernorm_type == "LayerNorm" else nn.RMSNorm
        self.P = 197
        self.num_decoder_blocks = num_blocks // 2
        self.full_global_attention = full_global_attention
        # If not full_global attention, we do 1.: patchwise over sequence 2.: framewise over patches
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.input_patches = input_patches
        self.num_heads = num_heads
        self.use_nested_tensor = use_nested_tensor
        self.head_types = list(heads)
        if "center" not in self.head_types:
            raise ValueError("LoTIS release model requires the 'center' prediction head.")
        unsupported_heads = set(self.head_types) - {"center", "visibility", "distances"}
        if unsupported_heads:
            raise ValueError(f"Unsupported prediction heads: {sorted(unsupported_heads)}")
        self.rope_freq_seq = rope_freq_seq
        self.rope_freq_spat = rope_freq_spat
        self.droppath = droppath
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.mini_batch_size = mini_batch_size
        # grad chkpt
        self.use_gradient_checkpointing = True
        self.enable_compile = compile
        self.use_reentrant = compile  # Somehow only works with reentrant=True if using torch.compile

        self.max_seq_len = max_seq_len
        
        # Position getter for spatial positions
        self.position_getter = PositionGetter()

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        
        # Camera tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, hidden_dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        self.patch_start_idx = 1  # due to camera token
        
        # Feature dimension adaptation
        self.feature_downsample = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Feature dimension adaptation
        self.feature_downsample_query = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )

        # Split dimension RoPE for position encoding
        self.split_rope = SplitDimensionRoPE(
            spatial_grid_size=input_patches,
            max_seq_len=max_seq_len,
            spatial_base_freq = self.rope_freq_spat,
            temporal_base_freq = self.rope_freq_seq,
            rope_jitter=rope_jitter,
            rope_jitter_range=rope_jitter_range,
            inference_scale=rope_inference_scale
        )
        
        # Dual attention encoder blocks (compile as a loop, not individually)
        self.encoder_blocks = nn.ModuleList([
            torch.compile(
            DualAttentionEncoderBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=3,
                dropout=dropout,
                attention_dropout=attention_dropout,
                droppath=droppath,
                apply_spatial_rope=True,
                apply_temporal_rope=True,
                full_global_attention=full_global_attention,
                use_nested_tensor=use_nested_tensor,
                compile=False,  # Individual blocks not compiled
                norm_layer=ln_t,
                use_checkpointing=False,
            ), disable=not compile, dynamic=True, fullgraph=True)
            for i in range(num_blocks)
        ])
        
        # Query encoder blocks (spatial only) - compile as a loop
        self.query_encoder_blocks = nn.ModuleList([
            DualAttentionEncoderBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=3,
                dropout=dropout,
                attention_dropout=attention_dropout,
                droppath=droppath,
                apply_spatial_rope=True,
                apply_temporal_rope=False,
                full_global_attention=full_global_attention,
                use_nested_tensor=False,
                skip_global=True,
                compile=False,  # Individual blocks not compiled
                norm_layer=ln_t,
            ) for i in range(num_blocks // 2)
        ])
        
        self.feature_adapter = nn.Sequential(
            ln_t(hidden_dim), # float32
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            ln_t(hidden_dim)  # Final normalization for stable cross-attention
        ) # Postprocess query features before cross-attention

        # Cross-attention blocks
        self.cross_attention_blocks = nn.ModuleList([torch.compile(
            MultiHeadAttention(
                E_q=self.hidden_dim,
                E_k=self.hidden_dim,
                E_v=self.hidden_dim,
                E_total=self.hidden_dim,
                nheads=num_heads,
                dropout=attention_dropout,
                bias=True,
                layernorm=ln_t
            ),
            dynamic=True,
            disable=True) for _ in range(self.num_decoder_blocks)])

        self.att_ls = nn.ModuleList([
            LayerScale(hidden_dim) for _ in range(self.num_decoder_blocks)
        ])
        self.ffn_ls = nn.ModuleList([
            LayerScale(hidden_dim) for _ in range(self.num_decoder_blocks)
        ])

        self.query_layer_norm = ln_t(hidden_dim)
        self.cross_att_ffn = nn.ModuleList([
            nn.ModuleList([
                ln_t(hidden_dim), # for q, float32
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout)
                ),
                ln_t(hidden_dim) # float32
            ]) for _ in range(self.num_decoder_blocks)
        ])

        self.decoder_local_blocks = nn.ModuleList([
            DualAttentionEncoderBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=3,
                dropout=dropout,
                attention_dropout=attention_dropout,
                droppath=droppath,
                apply_spatial_rope=True,
                apply_temporal_rope=False,
                full_global_attention=full_global_attention,
                use_nested_tensor=self.use_nested_tensor,
                skip_global=True,
                compile=compile,
                norm_layer=ln_t,
            )
            for i in range(self.num_decoder_blocks)
        ])

        self.center_decoder = LoTISPredictionHead(
            dim_in = hidden_dim,
            trunk_depth=head_depth,
            num_heads= num_heads,
            mlp_ratio=3,
            dropout=dropout,
            attention_dropout=attention_dropout,
            droppath=droppath,
            use_nested_tensor=self.use_nested_tensor,  
            compile=compile,  
            predict_visibility=True,
            layernorm=ln_t,
        )

        self._block_fwd = torch.compile(
            self._block_fwd_impl,
            dynamic=True,
            disable=not compile,
            fullgraph=True,
            # options={"fallback_random": True, "trace.enabled" :True}
        )

        # Compile the entire encoder loop instead of individual blocks
        self._encoder_loop = torch.compile(
            self._encoder_loop_impl,
            dynamic=True,
            disable=True, # IT seems that this is *NOT* worth it.
            fullgraph=True,
            # options={"fallback_random": True, "trace.enabled" :True, "shape_padding": False}
        )

        # Compile the query encoder loop
        self._query_encoder_loop = torch.compile(
            self._query_encoder_loop_impl,
            dynamic=True,
            disable=not compile,
            fullgraph=True,
            # options={"fallback_random": True, "trace.enabled" :True}
        )

    def _encoder_loop_impl(self, trajectory_features, spatial_mask, global_mask, temporal_indices, nested_metadata):
        """Run all encoder blocks in a loop - this entire loop can be compiled as one unit."""
        if not ONNX_EXPORT:
            spatial_pos = self.spatial_pos
        else:
            spatial_pos = self.position_getter(1, self.input_patches[0], self.input_patches[1], device=trajectory_features.device)
            spatial_pos = spatial_pos + 1
            pos_special = torch.zeros(1, self.patch_start_idx, 2, device=trajectory_features.device, dtype=spatial_pos.dtype)
            spatial_pos = torch.cat((pos_special, spatial_pos), dim=1)
        for block in self.encoder_blocks:
            trajectory_features = block(
                x=trajectory_features,
                split_rope=self.split_rope,
                spatial_pos=spatial_pos,
                temporal_indices=temporal_indices,
                spatial_mask=spatial_mask,
                global_mask=global_mask,
                nested_metadata=nested_metadata
            )
        return trajectory_features

    def _query_encoder_loop_impl(self, query_features):
        """Run all query encoder blocks in a loop - this entire loop can be compiled as one unit."""
        if ONNX_EXPORT:
            # Cannot access self.
            query_spatial_pos = self.position_getter(1, self.input_patches[0], self.input_patches[1], device=query_features.device)
            query_spatial_pos = query_spatial_pos + 1
            query_pos_special = torch.zeros(1, self.patch_start_idx, 2, device=query_features.device, dtype=query_spatial_pos.dtype)
            query_spatial_pos = torch.cat((query_pos_special, query_spatial_pos), dim=1)
        else:
            query_spatial_pos = self.query_spatial_pos
        for block in self.query_encoder_blocks:
            query_features = block(
                x=query_features,
                split_rope=self.split_rope,
                spatial_pos=query_spatial_pos,
                temporal_indices=None,
                spatial_mask=None,
                global_mask=None,
            )
        return query_features

    def _block_fwd_impl(self, query_camera_feats, cross_attn_features, min_patched_seq_len, max_patched_seq_len, B,  BS, BSP, P, C, global_mask, nested_metadata, offsets_p_single, spatial_pos):
        """Forward function for a single cross-attention block with interleaved local self-attention.
        Args:
            query_camera_feats: [B, P, C] - query features to attend to
            cross_attn_features: [B, S*P, C] - trajectory features attending from
            min_patched_seq_len: minimum sequence length in the batch (patched)
            max_patched_seq_len: maximum sequence length in the batch (patched)
            B: batch size
            P: number of patches + 1 (camera token)
            C: feature dimension
            global_mask: attention mask for cross-attention
            nested_metadata: NestedTensorMetadata for cross_attn_features
            offsets_p_single: offsets for patches only (no temporal dimension)
            spatial_pos: spatial position encodings
        Returns:
            cross_attn_features after cross-attention and local self-attention
        """

        # We also need to setup reshape functions. q, k, v are all of shape [B, S*P, ...], where S is sequence length for q, and 1 for k.

        # [B, S*P, num_heads, head_dim] -> [B * S, P, num_heads, head_dim], for spatial attention. Need to be careful, the offsets depends on whether we have q or k.
        batch_size = B
        # BS = nested_metadata.offsets[-1]  # B * S 
        # BSP = nested_metadata.patched_offsets[-1]  # B * S * P
        BP = batch_size * P
        if self.use_nested_tensor:
            pre_reshape_fun_q = lambda x: x.values().view(BS, P, self.num_heads, C // self.num_heads)
            pre_reshape_fun_k = lambda x: x.values().view(batch_size, P, self.num_heads, C // self.num_heads)
            # [B*S, P, num_heads, head_dim] -> [B, S*P, num_heads, head_dim], back for attention
            post_reshape_fun_q = lambda x: torch.nested.nested_tensor_from_jagged(
                x.view(BSP, self.num_heads, C // self.num_heads), # This needs to be [B * S * P, num_heads, head_dim]
                nested_metadata.patched_offsets, # offsets_p_single = torch.arange(batch_size+1, device=x.device) * P
                min_seqlen=nested_metadata.min_seq_len_patched,
                max_seqlen=nested_metadata.max_seq_len_patched,
            )
            post_reshape_fun_k = lambda x: torch.nested.nested_tensor_from_jagged(
                x.view(BP, self.num_heads, C // self.num_heads),
                offsets_p_single, # offsets_p_single = torch.arange(batch_size+1, device=x.device) * P
                min_seqlen=P,
                max_seqlen=P,
            )

            rope_spatial = functools.partial(
                self.split_rope,
                spatial_pos=spatial_pos,
                temporal_indices=None,
                reshape_functions=(pre_reshape_fun_q, None, post_reshape_fun_q),
                reshape_functions_k=(pre_reshape_fun_k, None, post_reshape_fun_k),
            )
        else:
            pre_reshape_fun_q = lambda x: x.view(BS, P, self.num_heads, C // self.num_heads)
            pre_reshape_fun_k = lambda x: x.view(batch_size, P, self.num_heads, C // self.num_heads)
            post_reshape_fun_q = lambda x: x.view(B, BS//B * P, self.num_heads, C // self.num_heads)
            post_reshape_fun_k = lambda x: x.view(B, P, self.num_heads, C // self.num_heads)
            rope_spatial = functools.partial(
                self.split_rope,
                spatial_pos=spatial_pos,
                temporal_indices=None,
                reshape_functions=(pre_reshape_fun_q, None, post_reshape_fun_q),
                reshape_functions_k=(pre_reshape_fun_k, None, post_reshape_fun_k),
            )

        # Hoist k/v nested tensor creation outside loop - query_camera_feats doesn't change
        # if self.use_nested_tensor:
        #     k_cached = torch.nested.nested_tensor_from_jagged(query_camera_feats.view(B*P, C),
        #                                             offsets_p_single,
        #                                             min_seqlen=P,
        #                                             max_seqlen=P,
        #                                             )
        #     v_cached = torch.nested.nested_tensor_from_jagged(query_camera_feats.view(B*P, C),
        #                                             offsets_p_single,
        #                                             min_seqlen=P,
        #                                             max_seqlen=P,
        #                                             )
        # else:
        #     k_cached = query_camera_feats
        #     v_cached = query_camera_feats

        for i, (block, ffn, decoder_local_block) in enumerate(zip(self.cross_attention_blocks, self.cross_att_ffn, self.decoder_local_blocks)):
            cross_attn_features_norm = ffn[0](cross_attn_features)  # First Layer norm

            q = cross_attn_features_norm
            k = query_camera_feats
            v = query_camera_feats
            if self.use_nested_tensor:
                k = torch.nested.nested_tensor_from_jagged(k.view(B*P, C),
                                                        offsets_p_single,
                                                        min_seqlen=P,
                                                        max_seqlen=P, 
                                                        )
                v = torch.nested.nested_tensor_from_jagged(v.view(B*P, C),
                                                        offsets_p_single,
                                                        min_seqlen=P,
                                                        max_seqlen=P,
                                                        )
                attn_output = block(query=q, key=k, value=v,
                                    min_seq_len_q=min_patched_seq_len, max_seq_len_q=max_patched_seq_len,
                                    min_seq_len_k=P, max_seq_len_k=P,
                                    rope=rope_spatial,
                                    )
            else:
                q = cross_attn_features_norm
                query_camera_feats_viewed = query_camera_feats.view(B, 1, P, C)
                k = query_camera_feats_viewed
                v = query_camera_feats
                # print(v.shape)

                attn_output = block(query=q, key=k, value=v, min_seq_len_q= min_patched_seq_len, max_seq_len_q=max_patched_seq_len,
                                        min_seq_len_k=P, max_seq_len_k=P,
                                     attn_mask=None, rope=rope_spatial)
            attn_output = self.att_ls[i](attn_output)


            if self.training:
                attn_w_residual = cross_attn_features + self.drop_path(attn_output)
                attn_w_residual_norm = ffn[2](attn_w_residual)  # Second Layer norm
                forward_out = ffn[1](attn_w_residual_norm)  # Forward
                forward_out = self.ffn_ls[i](forward_out)
                cross_attn_features = attn_w_residual + self.drop_path(forward_out)
            else:
                attn_w_residual = cross_attn_features + attn_output
                attn_w_residual_norm = ffn[2](attn_w_residual)
                forward_out = ffn[1](attn_w_residual_norm)  # Forward
                forward_out = self.ffn_ls[i](forward_out)
                cross_attn_features = attn_w_residual + forward_out

            # We now want to apply the local self attention on the cross-attention features
            # This module wants [B, S, P, C] input, so we need to reshape the cross-attn features
            if self.use_nested_tensor:
                # Reshape to [B, S, P, C]
                cross_attn_features = torch.nested.nested_tensor_from_jagged(
                    cross_attn_features.values().view(BS, P, C),  # needs to be [B*S, P, C]
                    nested_metadata.offsets,
                    min_seqlen=nested_metadata.min_seq_len,
                    max_seqlen=nested_metadata.max_seq_len,
                )
            else:
                cross_attn_features = cross_attn_features.view(B, BS // B, P, C)  # [B, S, P, C]

            cross_attn_features = decoder_local_block(
                x=cross_attn_features,
                split_rope=self.split_rope,
                spatial_pos=spatial_pos,
                temporal_indices=None,
                spatial_mask= None,
                global_mask=None,
                nested_metadata=nested_metadata)
            # Reshape back to [B, S*P, C]
            if self.use_nested_tensor:
                cross_attn_features = torch.nested.nested_tensor_from_jagged(
                    cross_attn_features.values().view(BSP, C), # [B*S*P, C]
                    nested_metadata.patched_offsets,
                    min_seqlen=nested_metadata.min_seq_len_patched,
                    max_seqlen=nested_metadata.max_seq_len_patched,
                )
            else:
                cross_attn_features = cross_attn_features.view(B, BSP // B, C)  # [B, S*P, C]

        # L = cross_attn_features_norm.sum()
        # L.backward()
        # # Print one
        # raise NotImplementedError("Debugging stop")             
        return cross_attn_features
    

    def decode_traj_mode(self, x, seq_lens, seq_mask_=None):
        """Decode features."""
        nvtx.range_push("decode_prep") if not ONNX_EXPORT else None
        trajectory_features, query_features = x
        full_batch_size = trajectory_features.shape[0]

        mini_batch_size = self.mini_batch_size if not ONNX_EXPORT else -1

        full_nested_metadata = NestedTensorMetadata.from_seqlens(trajectory_features, self.P, seq_lens)
        full_offsets = full_nested_metadata.offsets
        max_full_seq_len = full_nested_metadata.max_seq_len

        if self.use_nested_tensor:
            pre_head_camera_features_values = torch.zeros(
                (sum(seq_lens), query_features.shape[3]),
                dtype=query_features.dtype,
                device=query_features.device
                )
            pre_head_camera_features_nested = torch.nested.nested_tensor_from_jagged(
                pre_head_camera_features_values,
                full_offsets,
                min_seqlen=full_nested_metadata.min_seq_len,
                max_seqlen=full_nested_metadata.max_seq_len
            )
        else:
            pre_head_camera_features = torch.zeros(
                (query_features.shape[0], trajectory_features.shape[1], query_features.shape[3]),
                dtype=query_features.dtype,
                device=query_features.device
                )
        query_features = query_features.squeeze(1)  # [B, S, C] -> [B, C]
        query_features = self.query_layer_norm(query_features)
        
        nvtx.range_pop() if not ONNX_EXPORT else None

        # ============================================================================
        # ONNX EXPORT PATH: Process entire batch at once without mini-batching
        # ============================================================================
        if ONNX_EXPORT:
            nvtx.range_push("decode_batches") if not ONNX_EXPORT else None
            
            # Process entire batch
            B, S, P, C = trajectory_features.shape
            seq_lens_sum = seq_lens.sum()
            
            if self.use_nested_tensor:
                nested_metadata = full_nested_metadata
                patched_offsets = nested_metadata.patched_offsets
                min_patched_seq_len = nested_metadata.min_seq_len_patched
                max_patched_seq_len = nested_metadata.max_seq_len_patched
                
                trajectory_camera_feats = torch.nested.nested_tensor_from_jagged(
                    trajectory_features.values().view(seq_lens_sum * P, C),
                    patched_offsets,
                    min_seqlen=min_patched_seq_len,
                    max_seqlen=max_patched_seq_len,
                )
            else:
                trajectory_camera_feats = trajectory_features.reshape(B, S * P, C)
                min_patched_seq_len = S * P
                max_patched_seq_len = S * P
                nested_metadata = full_nested_metadata

            cross_attn_features = trajectory_camera_feats
            offsets_p_single = torch.arange(B+1, device=trajectory_features.device) * P
            BS = seq_lens_sum
            BSP = seq_lens_sum * P
            if ONNX_EXPORT:
                spatial_pos = self.position_getter(1, self.input_patches[0], self.input_patches[1], device=trajectory_features.device)
                spatial_pos = spatial_pos + 1
                pos_special = torch.zeros(1, self.patch_start_idx, 2, device=
                    trajectory_features.device, dtype=spatial_pos.dtype)
                spatial_pos = torch.cat((pos_special, spatial_pos), dim=1)
            else:
                spatial_pos = self.spatial_pos
            
            if self.use_nested_tensor:
                cross_attn_features = self._block_fwd(
                    query_features, cross_attn_features, min_patched_seq_len, max_patched_seq_len, 
                    B, BS, BSP, P, C, None, nested_metadata, offsets_p_single, spatial_pos)
                cross_attn_features = cross_attn_features.values().view(seq_lens_sum, P, C)[:, 0, :]
                pre_head_camera_features_values[:, :] = cross_attn_features
            else:
                cross_attn_features = self._block_fwd(
                    query_features, cross_attn_features, S * P, S * P, 
                    B, BS, BSP, P, C, None, nested_metadata, offsets_p_single, spatial_pos)
                pre_head_camera_features[:] = cross_attn_features.view(B, S, P, C)[:, :, 0, :]
            
            nvtx.range_pop() if not ONNX_EXPORT else None
        
        # ============================================================================
        # TRAINING PATH: Use mini-batching with loop
        # ============================================================================
        else:
            # Careful, the shapes might not be divisible by mini_batch_size, need to ensure we handle the last batch correctly
            if mini_batch_size == -1:
                mini_batch_size = full_batch_size
            num_batches = (trajectory_features.shape[0] + mini_batch_size - 1) // mini_batch_size

            # Precompute all BS and BSP values with ONE sync to avoid repeated CPU syncs in loop
            nvtx.range_push("precompute_batch_sizes") if not ONNX_EXPORT else None
            seq_lens_cumsum = torch.cat([torch.tensor([0], device=seq_lens.device), seq_lens.cumsum(0)])
            batch_start_indices = torch.arange(0, full_batch_size, mini_batch_size, device=seq_lens.device)
            batch_end_indices = torch.minimum(batch_start_indices + mini_batch_size, torch.tensor(full_batch_size, device=seq_lens.device))

            # Compute BS values for all mini-batches
            all_BS_values = seq_lens_cumsum[batch_end_indices] - seq_lens_cumsum[batch_start_indices]
            all_BS_values_list = all_BS_values.tolist()  # Single sync for all mini-batches
            all_BSP_values_list = [bs * self.P for bs in all_BS_values_list]  # Precompute BSP
            nvtx.range_pop() if not ONNX_EXPORT else None

            nvtx.range_push("decode_batches") if not ONNX_EXPORT else None
            for batch_idx in range(num_batches):
                start_idx = batch_idx * mini_batch_size
                end_idx = min((batch_idx + 1) * mini_batch_size, trajectory_features.shape[0])

                nvtx.range_push("seq_lens_sum")
                seq_lens_sum = all_BS_values_list[batch_idx]  # No sync needed - already computed
                nvtx.range_pop()

                seq_mask = seq_mask_[start_idx:end_idx] if seq_mask_ is not None else None

                nvtx.range_push("slice_njt")
                batch_trajectory_features = slice_njt(
                    trajectory_features, start_idx, end_idx, offsets=full_offsets)
                nvtx.range_pop()

                batch_query_features = query_features[start_idx:end_idx]

                nvtx.range_push("nested_metadata_creation")
                nested_metadata = NestedTensorMetadata.from_seqlens(batch_trajectory_features, self.P, seq_lens[start_idx:end_idx])
                nvtx.range_pop()
                nvtx.range_push("get_shape")
                B, S, P, C = batch_trajectory_features.shape
                nvtx.range_pop()

                query_camera_feats = batch_query_features
                if self.use_nested_tensor:
                    patched_offsets = nested_metadata.patched_offsets
                    min_patched_seq_len = nested_metadata.min_seq_len_patched
                    max_patched_seq_len = nested_metadata.max_seq_len_patched

                    nvtx.range_push("view_seq_lens_sum")
                    trajectory_camera_feats = torch.nested.nested_tensor_from_jagged(
                        batch_trajectory_features.values().view(seq_lens_sum * P, C),
                        patched_offsets,
                        min_seqlen=min_patched_seq_len,
                        max_seqlen=max_patched_seq_len,
                    )
                    nvtx.range_pop()
                else:
                    trajectory_camera_feats = batch_trajectory_features.reshape(B, S * P, C)
                    min_patched_seq_len = S*P
                    max_patched_seq_len = S*P

                cross_attn_features = trajectory_camera_feats

                nvtx.range_push("offsets_p_single") if not ONNX_EXPORT else None
                offsets_p_single = torch.arange(B+1, device=batch_trajectory_features.device) * P
                nvtx.range_pop() if not ONNX_EXPORT else None

                # Use precomputed values - no sync needed
                BS = all_BS_values_list[batch_idx]
                BSP = all_BSP_values_list[batch_idx]
                nvtx.range_push("block_fwd") if not ONNX_EXPORT else None
                if self.use_nested_tensor:
                    if self.training and self.use_gradient_checkpointing:
                        cross_attn_features = checkpoint(self._block_fwd,
                            query_camera_feats, cross_attn_features, min_patched_seq_len, max_patched_seq_len, B, BS, BSP, P, C, None, nested_metadata, offsets_p_single, self.spatial_pos, use_reentrant=self.use_reentrant)
                    else:
                        cross_attn_features = self._block_fwd(
                            query_camera_feats, cross_attn_features, min_patched_seq_len, max_patched_seq_len, B, BS, BSP, P, C, None, nested_metadata, offsets_p_single, self.spatial_pos)
                else:
                    cross_attn_features = self._block_fwd(
                        query_camera_feats, cross_attn_features, S * P, S * P, B, BS, BSP, P, C, None, nested_metadata, offsets_p_single, self.spatial_pos)
                
                nvtx.range_pop() if not ONNX_EXPORT else None
                
                # Extract camera token for heads
                if self.use_nested_tensor:
                    nvtx.range_push("viewing") if not ONNX_EXPORT else None
                    cross_attn_features = cross_attn_features.values().view(seq_lens_sum, P, C)[:, 0, :]
                    nvtx.range_pop() if not ONNX_EXPORT else None
                    nvtx.range_push("index_assignment") if not ONNX_EXPORT else None
                    pre_head_camera_features_values[full_offsets[start_idx]:full_offsets[end_idx], :] += cross_attn_features
                    nvtx.range_pop() if not ONNX_EXPORT else None
                else:
                    pre_head_camera_features[start_idx:end_idx] += cross_attn_features.view(B, S, P, C)[:, :, 0, :]
            
            nvtx.range_pop() if not ONNX_EXPORT else None
        
        # ============================================================================
        # HEAD DECODING (common to both paths)
        # ============================================================================
        nvtx.range_push("decode_heads") if not ONNX_EXPORT else None
        camera_feats_in = pre_head_camera_features_nested if self.use_nested_tensor else pre_head_camera_features
        center_coords, predicted_vis, predicted_dists = self.center_decoder(
            camera_feats_in,
            mask=seq_mask_,
            nested_metadata=full_nested_metadata,
        )
        return_dict = {
            "center": {"coords": center_coords if self.training else center_coords[-1]},
            "visibility": {"logits": predicted_vis if self.training else predicted_vis[-1]},
            "distances": {"values": predicted_dists if self.training else predicted_dists[-1]},
        }
        
        nvtx.range_pop() if not ONNX_EXPORT else None
        return return_dict

    def decode(self, x, seq_lens_sum, seq_mask=None, compute_variance=False, nested_metadata=None):
        raise NotImplementedError("Use decode_traj_mode for trajectory-mode inference.")
    
    def compute_seq_variance(self, x, name="", detailed=False):
        """Compute variance across sequence dimension."""
        if isinstance(x, tuple):
            return  # Skip if tuple
            
        # Determine the shape and compute variance appropriately
        if len(x.shape) == 4:  # [B, S, P, C]
            B, S, P, C = x.shape
            
            # Variance across sequence for each batch and position
            var_seq = torch.var(x, dim=1)  # [B, P, C]
            
            # Average variance across batch, positions, and channels
            var_mean = var_seq.median().item()
            
            if detailed:
                # Variance per position (averaged across batch and channels)
                var_per_pos = var_seq.mean(dim=(0, 2))  # [P]
                # Variance per channel (averaged across batch and positions)
                var_per_chan = var_seq.mean(dim=(0, 1))  # [C]
                
                print(f"Variance {name} - Overall: {var_mean:.6f}")
                print(f"  Per position min/max: {var_per_pos.min().item():.6f}/{var_per_pos.max().item():.6f}")
                print(f"  Per channel min/max: {var_per_chan.min().item():.6f}/{var_per_chan.max().item():.6f}")
            else:
                print(f"Variance {name}: {var_mean:.6f}")
                
        elif len(x.shape) == 3:  # [B, S, C]
            B, S, C = x.shape
            
            # Variance across sequence for each batch and channel
            var_seq = torch.var(x, dim=1)  # [B, C]
            
            # Average variance across batch and channels
            var_mean = var_seq.mean().item()
            
            if detailed:
                # Variance per channel (averaged across batch)
                var_per_chan = var_seq.mean(dim=0)  # [C]
                
                print(f"Variance {name} - Overall: {var_mean:.6f}")
                print(f"  Per channel min/max: {var_per_chan.min().item():.6f}/{var_per_chan.max().item():.6f}")
            else:
                print(f"Variance {name}: {var_mean:.6f}")
        else:
            print(f"Skipping variance for {name} - unexpected shape {x.shape}")


    def encode_trajectory(self, trajectory_features, seq_lens, seq_mask=None, num_future_masks=None, compute_variance=False, nested_metadata=None):
        B = trajectory_features.shape[0]
        S = trajectory_features.shape[1]
        P_X = trajectory_features.shape[2]
        P_Y = trajectory_features.shape[3]
        C = trajectory_features.shape[4]
        P = P_X * P_Y
        # We should be able to get this from the seq_lens thingy, but probably does not matter
        if nested_metadata is None:
            # We add 1 because of the camera token
            nested_metadata = NestedTensorMetadata.from_tensor(trajectory_features, self.P)
        min_seq_len = nested_metadata.min_seq_len if self.use_nested_tensor else S
        max_seq_len = nested_metadata.max_seq_len if self.use_nested_tensor else S
        seq_len_sum = nested_metadata.offsets[-1] if self.use_nested_tensor else B * S
        if self.use_nested_tensor:
            seq_lens_cum = F.pad(seq_lens.cumsum(dim=0), (1, 0))
        # Reshape features
        trajectory_features = trajectory_features.view(B, S, P, C)
        
        if compute_variance:
            self.compute_seq_variance(trajectory_features, "raw features")
            
        # Downsample features to hidden dimension
        trajectory_features = self.feature_downsample(trajectory_features)

        # Get spatial positions
        if not ONNX_EXPORT:
            self.spatial_pos = self.position_getter(1, P_X, P_Y, device=trajectory_features.device)
            
            # Adapt encoding for camera token
            if self.patch_start_idx > 0:
                self.spatial_pos = self.spatial_pos + 1
                spatial_pos_special = torch.zeros(1, self.patch_start_idx, 2).to(trajectory_features.device).to(self.spatial_pos.dtype)
                self.spatial_pos = torch.cat((spatial_pos_special, self.spatial_pos), dim=1)
            spatial_pos = self.spatial_pos
        else:
            spatial_pos = self.position_getter(1, P_X, P_Y, device=trajectory_features.device)
            
            # Adapt encoding for camera token
            if self.patch_start_idx > 0:
                spatial_pos = spatial_pos + 1
                spatial_pos_special = torch.zeros(1, self.patch_start_idx, 2).to(trajectory_features.device).to(spatial_pos.dtype)
                spatial_pos = torch.cat((spatial_pos_special, spatial_pos), dim=1)
        
        # Validate dimensions
        if C != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {C}")
        if P_X != self.input_patches[0] or P_Y != self.input_patches[1]:
            raise ValueError(f"Expected input patches {self.input_patches}, got ({P_X}, {P_Y})")
        
        # Concatenate camera tokens
        _, trajectory_camera_token = slice_expand_and_flatten(self.camera_token, B, seq_len_sum)
        if self.use_nested_tensor:
            nested_traj_camera_token = torch.nested.nested_tensor_from_jagged(trajectory_camera_token,
                                                                              offsets=nested_metadata.offsets,
                                                                              min_seqlen=min_seq_len,
                                                                              max_seqlen=max_seq_len
                                                                              )
            # print(trajectory_features._offsets)
            trajectory_features = torch.cat((nested_traj_camera_token, trajectory_features), dim=2)
            # print(trajectory_features._offsets)
        else:
            trajectory_camera_token = trajectory_camera_token.view(B, S, 1, self.hidden_dim)
            trajectory_features = torch.cat((trajectory_camera_token, trajectory_features), dim=2)
        B, S, P, C = trajectory_features.shape  # Update dimensions
        # Create spatial attention mask
        spatial_mask = None
        global_mask = None
        temporal_indices = None

        if not self.use_nested_tensor:
            temporal_indices = torch.arange(S, device=trajectory_features.device)

            # if seq_mask is not None:
            #     # Convert sequence mask to patch mask for spatial attention
            #     # Each frame either has all patches masked or none
            #     expanded_mask = seq_mask.view(B, S, 1).expand(B, S, P)
            #     spatial_mask = ~expanded_mask.reshape(B*S, P).bool()  # True means positions to mask
            
            # # Create global attention mask

            # if seq_mask is not None:
            #     # Expand sequence mask to account for all patches
            #     # From [B, S] to [B, S*P], where all patches in a masked frame are masked
            #     expanded_mask = seq_mask.unsqueeze(-1).expand(B, S, P).reshape(B, S*P)
            #     global_mask = ~expanded_mask.bool()  # True means positions to mask
        else:
            # Handled by the rope
            temporal_indices = seq_lens_cum
    

        # Process trajectory with dual attention encoder blocks
        # Use compiled loop for entire encoder (compiles all blocks as one unit)

        if self.training and self.use_gradient_checkpointing:
            # For gradient checkpointing, still iterate through blocks individually
            for i, block in enumerate(self.encoder_blocks):
                if compute_variance:
                    print(f"\n--- Processing block {i} ---")
                trajectory_features = checkpoint(
                    block,
                    trajectory_features,
                    self.split_rope,
                    spatial_pos,
                    temporal_indices,
                    spatial_mask,
                    global_mask,
                    nested_metadata,
                    use_reentrant=self.use_reentrant,

                )
                if compute_variance:
                    self.compute_seq_variance(trajectory_features, f"after encoder block {i}")
        else:
            # Standard forward pass - use compiled loop
            if compute_variance:
                print("\n--- Processing all encoder blocks (compiled loop) ---")
            trajectory_features = self._encoder_loop(
                trajectory_features,
                spatial_mask,
                global_mask,
                temporal_indices,
                nested_metadata
            )
            if compute_variance:
                self.compute_seq_variance(trajectory_features, "after all encoder blocks")

        return trajectory_features
    
    def encode_query(self, query_features, seq_lens, seq_mask=None, num_future_masks=None, compute_variance=False):
        # print(query_features.shape)
        if len(query_features.shape) != 5:
            query_features = query_features.unsqueeze(1)  # Add sequence dimension if missing
        B = query_features.shape[0]
        P_X = query_features.shape[2]
        P_Y = query_features.shape[3]
        C = query_features.shape[4]
        P = P_X * P_Y

        
        # Process query features
        query_features = query_features.view(B, 1, P, C)
        # breakpoint()
        query_features = self.feature_downsample_query(query_features)
        
        # Validate dimensions
        if C != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {C}")
        if P_X != self.input_patches[0] or P_Y != self.input_patches[1]:
            raise ValueError(f"Expected input patches {self.input_patches}, got ({P_X}, {P_Y})")
        
        # Process query positions
        if not ONNX_EXPORT:
            self.query_spatial_pos = self.position_getter(1, P_X, P_Y, device=query_features.device)
            self.query_spatial_pos = self.query_spatial_pos + 1
            query_pos_special = torch.zeros(1, self.patch_start_idx, 2, device=query_features.device, dtype=self.query_spatial_pos.dtype)
            self.query_spatial_pos = torch.cat((query_pos_special, self.query_spatial_pos), dim=1)
            query_spatial_pos = self.query_spatial_pos
        else:
            query_spatial_pos = self.position_getter(1, P_X, P_Y, device=query_features.device)
            query_spatial_pos = query_spatial_pos + 1
            query_pos_special = torch.zeros(1, self.patch_start_idx, 2, device=query_features.device, dtype=query_spatial_pos.dtype)
            query_spatial_pos = torch.cat((query_pos_special, query_spatial_pos), dim=1)
        
        # Concatenate camera tokens
        query_camera_token, _ = slice_expand_and_flatten(self.camera_token, B, 1)
        query_features = torch.cat((query_camera_token, query_features), dim=2)

        # Process query with spatial-only encoder blocks (compiled loop)
        query_features = self._query_encoder_loop(query_features)

        # Postprocess query features
        query_features = self.feature_adapter(query_features)
        return query_features
    
    def forward(self, imgs1, imgs2, trajectory_features, seq_lens, query_features, seq_mask=None, num_future_masks=None, compute_variance=False, skip_traj_encoding=False, traj_query_counts=None):
        """
        Forward pass through the model.
        
        Args:
            trajectory_features: [batch_size, seq_len, patches_x, patches_y, feature_dim]
            seq_lens: Sequence lengths for each batch element
            query_features: [batch_size, patches_x, patches_y, feature_dim]
            seq_mask: Optional binary mask for valid sequence positions
            compute_variance: If True, compute and print variance across sequence dimension
            
        Returns:
            predicted_masks: [batch_size, seq_len, H, W]
            visibility: [batch_size, seq_len, 1]
        """
        nvtx.range_push("model_forward") if not ONNX_EXPORT else None
        # Encode trajectory features
        B = trajectory_features.shape[0]
        if not ONNX_EXPORT:
            if not self.use_nested_tensor:
                # Non-nested tensor mode requires all sequences to have the same length
                if not torch.all(seq_lens == seq_lens[0]):
                    raise ValueError("Non-nested tensor mode requires all sequences to have the same length.")
                if self.training:
                    raise ValueError("Non-nested tensor mode only supports evaluation for now.")
        S = trajectory_features.shape[1]
        seq_len_sum = seq_lens.sum() if self.use_nested_tensor else B * S

        nested_metadata = NestedTensorMetadata.from_seqlens(trajectory_features, self.P, seq_lens)
        if not skip_traj_encoding:
            nvtx.range_push("encode_trajectory") if not ONNX_EXPORT else None
            trajectory_features = self.encode_trajectory(
                trajectory_features, 
                seq_lens, 
                seq_mask=seq_mask, 
                num_future_masks=num_future_masks, 
                compute_variance=compute_variance,
                nested_metadata=nested_metadata
            )
            nvtx.range_pop() if not ONNX_EXPORT else None
        # breakpoint()
        if traj_query_counts is not None:
            nvtx.range_push("repeat_trajectory_features") if not ONNX_EXPORT else None
            if self.use_nested_tensor:
                trajectory_features, new_seqlens, _ = repeat_nested_tensor_efficient(trajectory_features, seq_lens, traj_query_counts)
            else:
                trajectory_features = torch.repeat_interleave(trajectory_features, traj_query_counts, dim=0)
                new_seqlens = torch.repeat_interleave(seq_lens, traj_query_counts, dim=0)

            seq_mask_expanded = torch.repeat_interleave(seq_mask,
                                            traj_query_counts, dim=0) if seq_mask is not None else None
            nvtx.range_pop() if not ONNX_EXPORT else None

        # Encode query features
        nvtx.range_push("encode_query") if not ONNX_EXPORT else None
        query_features = self.encode_query(
            query_features,
            seq_lens,
            seq_mask=seq_mask,
            num_future_masks=num_future_masks,
            compute_variance=compute_variance
        )
        nvtx.range_pop() if not ONNX_EXPORT else None

        # Handle query replication for batched trajectories in non-nested tensor mode
        if not self.use_nested_tensor and B > 1:
            B_query = query_features.shape[0]
            if B_query == 1:
                # Replicate single query to match number of trajectories (memory efficient)
                query_features = query_features.expand(B, -1, -1, -1).contiguous()
            elif B_query < B:
                raise ValueError(f"Query batch size ({B_query}) must be 1 or >= trajectory batch size ({B})")
            # If B_query >= B, we're good (training case with multiple queries per trajectory)

        # Decode to predict masks
        # trajectory_features = trajectory_features.to_padded_tensor() if self.use_nested_tensor else trajectory_features
        if traj_query_counts is not None:
            nvtx.range_push("decode_traj_mode") if not ONNX_EXPORT else None
            # We are in trajectory mode, so we need to decode with the trajectory mode decoder
            ret =  self.decode_traj_mode(
                (trajectory_features, query_features),
                seq_lens=new_seqlens,
                seq_mask_=seq_mask_expanded,
            )
            nvtx.range_pop() if not ONNX_EXPORT else None
            nvtx.range_pop() if not ONNX_EXPORT else None
            return ret
        nvtx.range_push("decode") if not ONNX_EXPORT else None
        ret = self.decode(
            (trajectory_features, query_features),
            seq_mask=seq_mask,
            seq_lens_sum=seq_len_sum,
            compute_variance=compute_variance,
            nested_metadata=nested_metadata
        )
        nvtx.range_pop() if not ONNX_EXPORT else None
        nvtx.range_pop() if not ONNX_EXPORT else None
    
