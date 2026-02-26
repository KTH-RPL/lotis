# TODO Could be autoregressive?
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.dual_att_enc import DualAttentionEncoderBlock, MultiHeadAttention
from .layers.layer_scale import LayerScale
from .layers.drop_path import DropPath

class ProgressHead(nn.Module):
    """
    ProgressHead predicts the camera pose in path-local coordinates via cross-attention
    and a series of transformer blocks
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
        compile: bool = False,
        use_nested_tensor: bool = False,
    ):
        super().__init__()
        self.target_dim = 2 # Progress, distance to path
        self.trunk_depth = trunk_depth
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
                    bias=True
                ), disable=not compile)
            for _ in range(trunk_depth)])
        else:
            self.mha = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=dim_in,
                    num_heads=num_heads,
                    batch_first=True,
                    dropout=attention_dropout,
                )
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
            nn.RMSNorm(dim_in, dtype=torch.float32) for _ in range(trunk_depth)
        ])
        self.n2s = nn.ModuleList([
            nn.RMSNorm(dim_in, dtype=torch.float32) for _ in range(trunk_depth)
        ])

        self.ls1s = nn.ModuleList([
            LayerScale(dim_in, init_values) for _ in range(trunk_depth)
        ])
        self.ls2s = nn.ModuleList([
            LayerScale(dim_in, init_values) for _ in range(trunk_depth)
        ])

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.token_norm = nn.RMSNorm(dim_in, dtype=torch.float32)
        self.query_token_norm = nn.RMSNorm(dim_in, dtype=torch.bfloat16)
        self.trunk_norm = nn.RMSNorm(dim_in, dtype=torch.float32)
        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))  # 3 because we want: shift, scale, and gate

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.RMSNorm(dim_in, elementwise_affine=False, eps=1e-6, dtype=torch.bfloat16)
        self.pose_branch = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.GELU(),
            nn.Linear(dim_in // 2, self.target_dim),
        )
        self.droppath = DropPath(droppath) if droppath > 0. else nn.Identity()

    def forward(self, tokens: torch.Tensor, query_tokens: torch.Tensor, mask, num_iterations: int = 4) -> list:
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

        query_tokens = query_tokens
        query_tokens = self.query_token_norm(query_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, query_tokens, mask, num_iterations)
        return pred_pose_enc_list
    def trunk_fn(self, pose_tokens: torch.Tensor, query_tokens, key_padding_mask, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S + 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be S
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, 1, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            # print(f"query_tokens.shape: {query_tokens.shape}")

            # Adaptive layer normalization and modulation.
            query_tokens_modulated = gate_msa * modulate(self.adaln_norm(query_tokens), shift_msa, scale_msa)
            # print(f"query_tokens_modulated.shape: {query_tokens_modulated.shape}")

            query_tokens_modulated = query_tokens_modulated + query_tokens
            # print(f"query_tokens_modulated.shape: {query_tokens_modulated.shape}")

            # print(f"pose_tokens.shape: {pose_tokens.shape}")
            for i in range(self.trunk_depth):
                # Standard multi-head attention with norm, layer-scale, residuals and ffn.
                
                query_tokens_modulated_inner = self.n1s[i](query_tokens_modulated)
                q = query_tokens_modulated_inner
                k, v = pose_tokens, pose_tokens
                if self.use_nested_tensor:
                    q = self.mha[i](q, k, v)
                else:
                    q, _ = self.mha[i](q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
                q = self.ls1s[i](q)
                if self.training:
                    query_tokens_modulated = query_tokens_modulated + self.droppath(q)
                    query_tokens_modulated = query_tokens_modulated + self.droppath(self.ls2s[i](self.mlp[i](self.n2s[i](query_tokens_modulated))))
                else:
                    query_tokens_modulated = query_tokens_modulated + q
                    query_tokens_modulated = query_tokens_modulated + self.ls2s[i](self.mlp[i](self.n2s[i](query_tokens_modulated)))
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(query_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
            pred_pose_enc_list.append(F.sigmoid(pred_pose_enc))

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
