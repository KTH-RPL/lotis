"""Slicing utilities for nested tensors."""

import torch
from typing import Tuple, Optional


def slice_njt(nested_tensor, start, end, dim=0, offsets=None, return_metadata=False):
    """
    Slices a NestedTensor from start to end along the first dimension.

    Args:
        nested_tensor (torch.nested.NestedTensor): The NestedTensor to slice.
        start (int): The starting index for slicing.
        end (int): The ending index for slicing.
        dim (int, optional): The dimension along which to slice. Default is 0.
        offsets (torch.Tensor, optional): Offsets for the NestedTensor. If None,
            will use the offsets from the NestedTensor.
        return_metadata (bool, optional): Whether to return metadata along with
            the sliced NestedTensor. Default is False.

    Returns:
        torch.nested.NestedTensor: The sliced NestedTensor.
        metadata (offsets, min_seq_len, max_seq_len) if return_metadata is True.
    """
    if nested_tensor.is_nested:
        if offsets is None:
            offsets = nested_tensor.offsets()

        tensor_values = nested_tensor.values()
        if dim == 1:
            raise ValueError("Cannot slice along the dynamic dimension (dim=1) of a NestedTensor.")

        if dim == 0:
            offset_0 = offsets[start]
            offset_1 = offsets[end]
            sliced_values = tensor_values[offset_0:offset_1]
            sliced_tensor = torch.nested.nested_tensor_from_jagged(
                sliced_values,
                offsets[start:end + 1] - offsets[start],
            )
            if return_metadata:
                sliced_offsets = offsets[start:end + 1] - offsets[start]
                min_seq_len = (sliced_offsets[1:] - sliced_offsets[:-1]).min().item()
                max_seq_len = (sliced_offsets[1:] - sliced_offsets[:-1]).max().item()
                return sliced_tensor, (sliced_offsets, min_seq_len, max_seq_len)
            return sliced_tensor
        else:
            slice_dim = dim - 1
            slices = [slice(None)] * tensor_values.ndim
            slices[slice_dim] = slice(start, end)
            sliced_values = tensor_values[tuple(slices)]
            sliced_tensor = torch.nested.nested_tensor_from_jagged(
                sliced_values,
                offsets,
            )
            if return_metadata:
                sliced_offsets = offsets
                min_seq_len = (sliced_offsets[1:] - sliced_offsets[:-1]).min().item()
                max_seq_len = (sliced_offsets[1:] - sliced_offsets[:-1]).max().item()
                return sliced_tensor, (sliced_offsets, min_seq_len, max_seq_len)
            return sliced_tensor

    else:
        return nested_tensor[start:end] if isinstance(nested_tensor, torch.Tensor) else None
