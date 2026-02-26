import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class NestedTensorMetadata:
    """
    A dataclass to encapsulate metadata for torch.nested.NestedTensor operations.
    """
    offsets: torch.Tensor
    lengths: torch.Tensor
    patched_offsets: torch.Tensor
    min_seq_len: int
    max_seq_len: int
    min_seq_len_patched: int
    max_seq_len_patched: int

    @classmethod
    def from_seqlens(cls, tensor, num_patches, seqlens):
        """
        Create a NestedTensorMetadata instance from sequence lengths. This avoids calling _get_min_seqlen.
        Input shape [B, S, P, C] or [B, S, P_X, P_Y, C] where:

        Usage example:
            >>> metadata = NestedTensorMetadata.from_seqlens(nested_tensor, num_patches)
        """
        if not tensor.is_nested:
            return cls(
                offsets=torch.tensor([]),
                lengths=torch.tensor([]),
                patched_offsets=torch.tensor([]),
                min_seq_len=None,
                max_seq_len=None,
                min_seq_len_patched=None,
                max_seq_len_patched=None
            )
        min_seq_len = torch.min(seqlens).item()
        max_seq_len = torch.max(seqlens).item()
        return cls(
            offsets=tensor._offsets,
            lengths=tensor.lengths(),
            patched_offsets=tensor._offsets * num_patches,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            min_seq_len_patched=min_seq_len * num_patches,
            max_seq_len_patched=max_seq_len * num_patches
        )

    @classmethod
    def from_tensor(cls, tensor, num_patches):
        """
        Create a NestedTensorMetadata instance from a NestedTensor.
        Input shape [B, S, P, C] or [B, S, P_X, P_Y, C] where:

        Usage example:
            >>> metadata = NestedTensorMetadata.from_tensor(nested_tensor)
        """
        if not tensor.is_nested:
            return cls(
                offsets=torch.tensor([]),
                lengths=torch.tensor([]),
                patched_offsets=torch.tensor([]),
                min_seq_len=None,
                max_seq_len=None,
                min_seq_len_patched=None,
                max_seq_len_patched=None
            )
        min_seq_len = tensor._get_min_seqlen()
        max_seq_len = tensor._get_max_seqlen()
        return cls(
            offsets=tensor._offsets,
            lengths=tensor.lengths(),
            patched_offsets=tensor._offsets * num_patches,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            min_seq_len_patched=min_seq_len * num_patches,
            max_seq_len_patched=max_seq_len * num_patches
        )
    
