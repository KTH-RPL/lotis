import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

from .layers.dual_att_enc import MultiHeadAttention
from .layers.layer_scale import LayerScale
from .layers.drop_path import DropPath
from .layers.njt_utils.nested_metadata import NestedTensorMetadata
# torch._dynamo.config.capture_scalar_outputs = True

class LoTISPredictionHead(nn.Module):
    """
    LoTISPredictionHead predicts trajectory-relative query coordinates from token representations.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 256,
        trunk_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: int = 3,
        init_values: float = 0.01,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        droppath: float = 0.0,
        predict_visibility: bool = False,
        compile: bool = False,
        layernorm = None,
        use_nested_tensor: bool = False,
    ):
        super().__init__()
        # self.target_dim = 2 if not predict_visibility else 3
        self.target_dim = 2 if not predict_visibility else 4
        self.trunk_depth = trunk_depth
        self.predict_visibility = predict_visibility
        self.use_nested_tensor = use_nested_tensor
        # Build the trunk using a sequence of transformer blocks.
        if self.use_nested_tensor:
            self.mha = nn.ModuleList([
                torch.compile(MultiHeadAttention(
                    E_q=dim_in,
                    E_k=dim_in,
                    E_v=dim_in,
                    E_total=dim_in,
                    nheads=num_heads,
                    dropout=attention_dropout,
                    bias=True,
                    layernorm=layernorm,
                ), disable=True, fullgraph=True, dynamic=True)
            for _ in range(trunk_depth)])
        else:
            self.mha = nn.ModuleList([
                torch.compile(MultiHeadAttention(
                    E_q=dim_in,
                    E_k=dim_in,
                    E_v=dim_in,
                    E_total=dim_in,
                    nheads=num_heads,
                    dropout=attention_dropout,
                    bias=True,
                    layernorm=layernorm,
                ), disable=True, fullgraph=True, dynamic=True)
            for _ in range(trunk_depth)])

        self.mlp = nn.ModuleList([
                nn.Sequential(
                nn.Linear(dim_in, dim_in * mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_in * mlp_ratio, dim_in),
                nn.Dropout(dropout)
            )for _ in range(trunk_depth)])
        
        self.n1s = nn.ModuleList([
            layernorm(dim_in) for _ in range(trunk_depth)
        ])  # float32
        self.n2s = nn.ModuleList([
            layernorm(dim_in) for _ in range(trunk_depth)
        ])

        self.ls1s = nn.ModuleList([
            LayerScale(dim_in, init_values) for _ in range(trunk_depth)
        ])
        self.ls2s = nn.ModuleList([
            LayerScale(dim_in, init_values) for _ in range(trunk_depth)
        ])

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, self.target_dim)) 
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.token_norm = layernorm(dim_in) # float32 # float32
        self.trunk_norm = layernorm(dim_in) # float32
        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))  # 3 because we want: shift, scale, and gate

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = layernorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.GELU(),
            nn.Linear(dim_in // 2, self.target_dim),
        )
        self.droppath = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.trunk_fn = torch.compile(
            self._trunk_fn,
            dynamic=True,
            disable=True,
            fullgraph=True,
            # mode="reduce-overhead",
            # options={"fallback_random": True}
            )
        self.trunk_inner = torch.compile(
            self._trunk_inner,
            dynamic=True,
            disable=not compile,
            fullgraph=True,
            # mode="reduce-overhead",
            # options={"fallback_random": True}
            )

    def forward(self, tokens: torch.Tensor, mask, num_iterations: int = 4, nested_metadata: NestedTensorMetadata = None) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            tokens (torch.Tensor): Input camera tokens with shape [B, S, C].
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Extract the camera tokens
        pose_tokens = tokens
        pose_tokens = self.token_norm(pose_tokens)
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        offsets = nested_metadata.offsets
        seq_len_sum = B * S if not self.use_nested_tensor else offsets[-1]
        pred_pose_enc_list = self.trunk_fn(pose_tokens, mask, num_iterations, nested_metadata, seq_len_sum)
        return pred_pose_enc_list

    def _trunk_inner(self, module_input, pose_tokens, key_padding_mask, min_seq_len, max_seq_len):
        shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)  # [B, S, C]

        # Adaptive layer normalization and modulation.
        pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)

        pose_tokens_modulated = pose_tokens_modulated + pose_tokens

        for i in range(self.trunk_depth):
            # Standard multi-head attention with norm, layer-scale, residuals and ffn.
            pose_tokens_modulated_inner = self.n1s[i](pose_tokens_modulated)
            q, k, v = pose_tokens_modulated_inner, pose_tokens_modulated_inner, pose_tokens_modulated_inner
            if self.use_nested_tensor:
                q = self.mha[i](q, k, v, min_seq_len_q=min_seq_len, max_seq_len_q=max_seq_len)
            else:
                q = self.mha[i](q, k, v, min_seq_len_q=min_seq_len, max_seq_len_q=max_seq_len)
            q = self.ls1s[i](q)

            if self.training:
                pose_tokens_modulated = pose_tokens_modulated + self.droppath(q)
                pose_tokens_modulated = pose_tokens_modulated + self.droppath(self.ls2s[i](self.mlp[i](self.n2s[i](pose_tokens_modulated))))
            else:
                pose_tokens_modulated = pose_tokens_modulated + q
                pose_tokens_modulated = pose_tokens_modulated + self.ls2s[i](self.mlp[i](self.n2s[i](pose_tokens_modulated)))

        # Compute the delta update for the pose encoding.
        result = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
        return result

    def _trunk_fn(self, pose_tokens: torch.Tensor, key_padding_mask, num_iterations: int, nested_metadata: NestedTensorMetadata, seq_len_sum) -> list:
            """
            Iteratively refine camera pose predictions.

            Args:
                pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
                num_iterations (int): Number of refinement iterations.

            Returns:
                list: List of activated camera encodings from each iteration.
            """
            B, S, C = pose_tokens.shape  # S is expected to be 1.
            offsets = nested_metadata.offsets
            # seq_len_sum = B * S if not self.use_nested_tensor else offsets[-1]
            min_seq_len = nested_metadata.min_seq_len if self.use_nested_tensor else S
            max_seq_len = nested_metadata.max_seq_len if self.use_nested_tensor else S
            pred_pose_enc = None
            pred_coords_list = []
            pred_visibility_logits_list = []
            pred_dists_list = []

            for iter_idx in range(num_iterations):
                # Use a learned empty pose for the first iteration.
                if pred_pose_enc is None:
                    # module_input = self.embed_pose(self.empty_pose_tokens) # In broadcasting we trust
                    module_input = self.embed_pose(self.empty_pose_tokens.expand(seq_len_sum, -1))

                else:
                    # Detach the previous prediction to avoid backprop through time.
                    pred_pose_enc = pred_pose_enc.detach()
                    module_input = self.embed_pose(pred_pose_enc)
                # Right now, module_input is of shape [B * S, C]

                # Generate modulation parameters and split them into shift, scale, and gate components.
                if self.use_nested_tensor:
                    # Not required any longer, broadcasting works fine
                    # pass
                    module_input = torch.nested.nested_tensor_from_jagged(module_input,
                                                                        offsets=offsets,
                                                                        min_seqlen=min_seq_len,
                                                                        max_seqlen=max_seq_len
                                                                        ) if not module_input.is_nested else module_input
                else:
                    module_input = module_input.view(B, S, -1)

                # module_input = module_input.expand(seq_len_sum, -1)

                # L = module_input.sum()

                # L.backward()
                # # print(shift_msa, scale_msa, gate_msa)
                # print(L.item(), module_input.shape, self.empty_pose_tokens.grad)
                nvtx.range_push(f"trunk_inner_iter_{iter_idx}")
                pred_pose_enc_delta = self.trunk_inner(module_input, pose_tokens, key_padding_mask, min_seq_len, max_seq_len)
                nvtx.range_pop()
                # print(pred_pose_enc_delta.min(), pred_pose_enc_delta.max(), pred_pose_enc_delta.mean())
                if pred_pose_enc is None:
                    pred_pose_enc = pred_pose_enc_delta
                else:
                    pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

                # print(pred_pose_enc.min(), pred_pose_enc.max(), pred_pose_enc.mean())
                if self.predict_visibility:
                    # If visibility is predicted, split the output into coordinates and visibility logits.
                    # We should pad here
                    pred_pose_enc_out = pred_pose_enc.to_padded_tensor(0, (B, max_seq_len, self.target_dim)) if self.use_nested_tensor else pred_pose_enc
                    pred_coords = F.tanh(pred_pose_enc_out[:, :, :2])
                    pred_visibility_logits = pred_pose_enc_out[:, :, 2:3]
                    pred_dists = (F.tanh(pred_pose_enc_out[:, :, 3]) + 1.0) / 2.0 # Scale to [0, 1]
                    pred_coords_list.append(pred_coords)
                    pred_visibility_logits_list.append(pred_visibility_logits)
                    pred_dists_list.append(pred_dists)
                else:
                    # Otherwise, just use the coordinates.
                    pred_coords = F.tanh(pred_pose_enc)
                    pred_coords_list.append(pred_coords)
            if self.predict_visibility:
                # If visibility is predicted, return both coordinates and visibility logits.
                return pred_coords_list, pred_visibility_logits_list, pred_dists_list
                # return pred_coords_list, pred_visibility_logits_list, None
            # If visibility is not predicted, return only coordinates.
            else:
                return pred_coords_list

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
