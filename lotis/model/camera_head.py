import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

from .layers.dual_att_enc import DualAttentionEncoderBlock, MultiHeadAttention
from .layers.layer_scale import LayerScale
from .layers.drop_path import DropPath
from .layers.njt_utils.nested_metadata import NestedTensorMetadata
# torch._dynamo.config.capture_scalar_outputs = True

class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

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
        # TODO Change RMSNorm to LayerNorm if problems arise.Actually, might be wrong :)
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


def main():
    """
    Debug function to compare gradients with and without torch.compile.
    Tests gradient stability with AMP bfloat16.
    """
    import numpy as np

    def set_seed(seed=42):
        """Set seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def get_model_gradients(model):
        """Extract all gradients from a model."""
        grads = {}
        for name, param in model.named_parameters():
            if "empty_pose_tokens" in name:
                print(param.grad[0,0].item() if param.grad is not None else None)
            if param.grad is not None:
                grads[name] = param.grad.clone().cpu()
        return grads

    def compare_gradients(grads1, grads2, rtol=1e-3, atol=1e-5):
        """Compare two gradient dictionaries."""
        print("\n" + "="*80)
        print("GRADIENT COMPARISON")
        print("="*80)

        all_close = True
        for name in grads1.keys():
            if name not in grads2:
                print(f"❌ {name}: Missing in second run")
                all_close = False
                continue

            g1 = grads1[name]
            g2 = grads2[name]

            # Compute various metrics
            max_diff = torch.abs(g1 - g2).max().item()
            mean_diff = torch.abs(g1 - g2).mean().item()
            rel_diff = (torch.abs(g1 - g2) / (torch.abs(g1) + 1e-8)).mean().item()

            is_close = torch.allclose(g1, g2, rtol=rtol, atol=atol)

            status = "✓" if is_close else "✗"
            print(f"{status} {name:50s} | max_diff: {max_diff:.6e} | mean_diff: {mean_diff:.6e} | rel_diff: {rel_diff:.6f}")

            if not is_close:
                all_close = False

        print("="*80)
        if all_close:
            print("✓ All gradients match within tolerance!")
        else:
            print("✗ Some gradients differ beyond tolerance!")
        print("="*80 + "\n")

        return all_close

    # Test parameters
    batch_size = 40
    seq_len = 40
    dim = 256
    num_heads = 8
    trunk_depth = 3
    n_iter = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}, dim={dim}\n")

    # ========================================================================
    # Test 1: WITH torch.compile on trunk_fn
    # ========================================================================
    print("="*80)
    print("TEST 1: WITH torch.compile (disable=False)")
    print("="*80)

    set_seed(309)

    # Create model with compile enabled and nested tensors
    model_compiled = CameraHead(
        dim_in=dim,
        trunk_depth=trunk_depth,
        num_heads=num_heads,
        predict_visibility=True,
        compile=True,
        layernorm=nn.RMSNorm,
        use_nested_tensor=True,  # Enable nested tensors
    ).to(device)

    # Create dummy input - nested tensor format
    set_seed(309)
    # For nested tensors, we need varying sequence lengths
    seq_lens = torch.tensor([seq_len, seq_len], device=device)  # Same length for simplicity
    total_seq_len = seq_lens.sum().item()

    # Create flat tensor for nested tensor
    tokens_flat = torch.randn(total_seq_len, dim, dtype=torch.bfloat16, device=device, requires_grad=True)

    # Create nested metadata
    from layers.njt_utils.nested_metadata import NestedTensorMetadata
    offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)])

    # Convert to nested tensor
    tokens = torch.nested.nested_tensor_from_jagged(
        tokens_flat,
        offsets=offsets,
        min_seqlen=seq_len,
        max_seqlen=seq_len
    )
    nested_metadata = NestedTensorMetadata.from_tensor(tokens, 1)

    mask = None  # No mask for simplicity

    # Forward pass with autocast
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model_compiled(tokens, mask, num_iterations=n_iter, nested_metadata=nested_metadata)

    # Sum one of the outputs for gradient computation
    if isinstance(outputs, tuple):
        # predict_visibility=True returns (coords_list, vis_list, dists_list)
        loss = outputs[0][0].sum()
        for i in range(1, n_iter):
            loss = loss + outputs[0][i].sum()
    else:
        loss = outputs[0].sum()
        for i in range(1, n_iter):
            loss = loss + outputs[i].sum()
    # print(outputs)
    print(f"Loss (compiled): {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Extract gradients
    grads_compiled = get_model_gradients(model_compiled)

    print(f"Number of parameters with gradients: {len(grads_compiled)}")

    # ========================================================================
    # Test 2: WITHOUT torch.compile on trunk_fn
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: WITHOUT torch.compile (disable=True)")
    print("="*80)

    set_seed(309)

    # Create model without compile, but with nested tensors
    model_uncompiled = CameraHead(
        dim_in=dim,
        trunk_depth=trunk_depth,
        num_heads=num_heads,
        predict_visibility=True,
        compile=False,
        layernorm=nn.RMSNorm,
        use_nested_tensor=True,  # Enable nested tensors
    ).to(device)

    # Create same dummy input
    set_seed(309)
    tokens_flat_unc = torch.randn(total_seq_len, dim, dtype=torch.bfloat16, device=device, requires_grad=True)

    # Convert to nested tensor (same structure as before)
    tokens_unc = torch.nested.nested_tensor_from_jagged(
        tokens_flat_unc,
        offsets=offsets,
        min_seqlen=seq_len,
        max_seqlen=seq_len
    )

    # Forward pass with autocast
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs_unc = model_uncompiled(tokens_unc, mask, num_iterations=n_iter, nested_metadata=nested_metadata)

    # Sum one of the outputs for gradient computation
    # print(outputs_unc)
    if isinstance(outputs_unc, tuple):
        loss_unc = outputs_unc[0][0].sum()
        for i in range(1, n_iter):
            loss_unc = loss_unc + outputs_unc[0][i].sum()
    else:
        loss_unc = outputs_unc[0].sum()
        for i in range(1, n_iter):
            loss_unc = loss_unc + outputs_unc[i].sum()

    print(f"Loss (uncompiled): {loss_unc.item():.6f}")

    # Backward pass
    loss_unc.backward()

    # Extract gradients
    grads_uncompiled = get_model_gradients(model_uncompiled)

    print(f"Number of parameters with gradients: {len(grads_uncompiled)}")

    # ========================================================================
    # Compare gradients
    # ========================================================================
    gradients_match = compare_gradients(grads_compiled, grads_uncompiled, rtol=1e-3, atol=1e-5)

    if not gradients_match:
        print("\n⚠️  GRADIENT INSTABILITY DETECTED!")
        print("The compiled version produces different gradients than the uncompiled version.")
        print("This confirms the issue with torch.compile + AMP bfloat16.\n")
    else:
        print("\n✓ Gradients are stable across compiled and uncompiled versions.\n")


if __name__ == "__main__":
    main()
