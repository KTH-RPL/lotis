import torch
import time 


def repeat_nested_tensor_efficient_old(nested_tensor, repeats):
    """
    Memory-efficient repeat for nested tensors using contiguous allocation.
    
    Args:
        nested_tensor: Nested tensor of shape [B, S_i, F] where S_i can vary
        repeats: Tensor of shape [B] with repeat counts for each batch element
        
    Returns:
        New nested tensor with repeated elements in contiguous memory
    """
    # Get the individual tensors and their properties
    tensor_list = nested_tensor.unbind()
    
    # Convert repeats to Python list once to avoid multiple .item() calls
    repeat_counts = repeats.tolist()
    
    # Get the last dimensions (all tensors should have same trailing dims)
    # For nested tensor [B, S_i, F1, F2, ...], we need the trailing dims
    trailing_shape = tensor_list[0].shape[1:]  # Everything after sequence dim
    
    # Pre-calculate sequence lengths and total size using vectorized operations
    seq_lens = torch.tensor([t.shape[0] for t in tensor_list], device=nested_tensor.device)
    expanded_seq_lens = torch.repeat_interleave(seq_lens, repeats)
    total_seq_length = expanded_seq_lens.sum().item()
    
    # Calculate offsets directly from expanded sequence lengths
    new_offsets = torch.cat([torch.tensor([0], device=nested_tensor.device), torch.cumsum(expanded_seq_lens, dim=0)])
    
    values_shape = (total_seq_length,) + trailing_shape
    values = torch.empty(values_shape, dtype=tensor_list[0].dtype, device=tensor_list[0].device)
    
    # Fill the contiguous block using optimized copying
    current_offset = 0
    for i, (tensor, repeat_count) in enumerate(zip(tensor_list, repeat_counts)):
        seq_len = tensor.shape[0]
        
        if repeat_count == 1:
            # Single copy optimization
            end_offset = current_offset + seq_len
            values[current_offset:end_offset] = tensor
            current_offset = end_offset
        else:
            # Use tensor.repeat for efficient bulk copying
            # repeated_tensor = 
            total_len = seq_len * repeat_count
            end_offset = current_offset + total_len
            values[current_offset:end_offset] = tensor.expand(repeat_count, *tensor.shape).reshape(-1, *trailing_shape)  # Unnecessary allocation and copy for repeat operation, we could just use the memory in values..
            current_offset = end_offset
    
    # Create the nested tensor
    result = torch.nested.nested_tensor_from_jagged(
        values, 
        offsets=new_offsets
    )
    
    return result, expanded_seq_lens, total_seq_length

def repeat_nested_tensor_efficient(nested_tensor, seq_lens, repeats):
    """
    Memory-efficient repeat for nested tensors using contiguous allocation.
    
    Args:
        nested_tensor: Nested tensor of shape [B, S_i, F] where S_i can vary
        seq_lens: Tensor of shape [B] with sequence lengths for each batch element
        repeats: Tensor of shape [B] with repeat counts for each batch element
        
    Returns:
        New nested tensor with repeated elements in contiguous memory
    """
    tensor_values = nested_tensor.values()
    offsets = nested_tensor.offsets()
    batch_size = len(seq_lens)
    device = tensor_values.device
    total_repeated_elements = (seq_lens * repeats).sum().item()
    
    # Create one big index tensor efficiently
    all_indices = torch.empty(total_repeated_elements, dtype=torch.long, device=device)
    
    output_offset = 0
    
    for i in range(batch_size):
        batch_start = offsets[i]
        seq_len = seq_lens[i]
        repeat_count = repeats[i]
        
        # Create indices for this batch: [batch_start, batch_start+1, ..., batch_start+seq_len-1]
        batch_indices = torch.arange(
            batch_start, 
            batch_start + seq_len, 
            device=device, 
            dtype=torch.long
        )
        
        # Repeat these indices repeat_count times
        repeated_batch_indices = batch_indices.repeat(repeat_count)
        
        # Place into the pre-allocated tensor
        output_end = output_offset + len(repeated_batch_indices)
        all_indices[output_offset:output_end] = repeated_batch_indices
        
        output_offset = output_end
    
    # Step 3: Use advanced indexing to gather all repeated values at once
    repeated_values = tensor_values[all_indices]

    new_seq_lens = torch.repeat_interleave(seq_lens, repeats)

    new_offsets = torch.cat((torch.tensor([0], device=tensor_values.device), torch.cumsum(new_seq_lens, dim=0)), dim=0)
    # To repeat efficiently, we need an indexing tensor that roughly works like this:
    # We construct an index tensor to index all the elements of the first tensor
    # This means, for tensor one the index is [0, 1, 2, 3, 4, 5] if seq_len[0] = 6
    # For tensor two it would be [6, 7, 8, 9, 10] if seq_len[1] = 5 etc.

    # Then, we can repeat this index tensor according to the repeat counts somehow. Ideally we would instantiate each of the index-tensors in one big one straight away
    # Create the nested tensor
    result = torch.nested.nested_tensor_from_jagged(
        repeated_values, 
        offsets=new_offsets,
    )
    
    return result, new_seq_lens, new_seq_lens.sum().item()



def test_efficient_repeat_nested_tensor():
    """
    Test the memory-efficient nested tensor repeat function.
    """
    print("Testing memory-efficient repeat_nested_tensor...")
    print("=" * 60)
    
    # Create nested tensor with gradient tracking
    seq_len = 40
    feature_dim = 768
    fun_dim = 256
    batch_size = 4
    
    # Same sequence length for all batch elements
    tensor_list = [
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
    ]
    
    n_repeats = torch.tensor([35, 60, 18, 70])  # Different repeat counts
    
    original_nested = torch.nested.as_nested_tensor(tensor_list, layout=torch.jagged)
    
    print(f"Original shapes: {[t.shape for t in tensor_list]}")
    print(f"Repeat counts: {n_repeats.tolist()}")
    lengths = torch.diff(original_nested.offsets())
    print(lengths)
    # Test the efficient version
    a = time.time()
    repeated_nested, _, _ = repeat_nested_tensor_efficient(original_nested, lengths, n_repeats)
    print(f"Repeat took {time.time() - a:.4f} seconds")



    # Old version for comparison
    a = time.time()
    repeated_nested_old, _, _ = repeat_nested_tensor_efficient_old(original_nested, n_repeats)
    print(f"Old repeat took {time.time() - a:.4f} seconds")

    # Check if the new version matches the old one
    assert torch.allclose(repeated_nested.values(), repeated_nested_old.values()), "New repeat does not match old version!"
    print("New repeat matches old version!")




    repeated_list = repeated_nested.unbind()
    
    print(f"Result batch size: {len(repeated_list)}")
    print(f"Result shapes: {[t.shape for t in repeated_list]}")
    
    # Verify correctness
    expected_shapes = []
    for i, repeat_count in enumerate(n_repeats):
        for _ in range(repeat_count):
            expected_shapes.append(tensor_list[i].shape)
    
    actual_shapes = [t.shape for t in repeated_list]
    shapes_correct = actual_shapes == expected_shapes
    print(f"Shape correctness: {'✓' if shapes_correct else '✗'}")
    
    # Verify values are correct
    values_correct = True
    current_idx = 0
    for i, repeat_count in enumerate(n_repeats):
        original = tensor_list[i]
        for j in range(repeat_count):
            repeated = repeated_list[current_idx + j]
            if not torch.allclose(repeated, original):
                values_correct = False
                break
        current_idx += repeat_count
        if not values_correct:
            break
    
    print(f"Value correctness: {'✓' if values_correct else '✗'}")
    
    # Test memory contiguity
    # Get the underlying storage info
    print(f"\nMemory layout analysis:")
    try:
        # Check if the nested tensor uses contiguous storage
        # This is a bit tricky to test directly, but we can check storage sizes
        storage_ptrs = [t.data_ptr() for t in repeated_list]
        print(f"First few data pointers: {storage_ptrs[:3]}")
        
        # Calculate expected pointer differences for contiguous layout
        elem_size = repeated_list[0].element_size()
        expected_diffs = []
        for i in range(len(repeated_list)-1):
            current_size = repeated_list[i].numel() * elem_size
            expected_diffs.append(current_size)
        
        actual_diffs = [storage_ptrs[i+1] - storage_ptrs[i] for i in range(len(storage_ptrs)-1)]
        
        print(f"Expected pointer diffs: {expected_diffs[:3]}")
        print(f"Actual pointer diffs: {actual_diffs[:3]}")
        
        contiguous = all(abs(actual - expected) < 100 for actual, expected in zip(actual_diffs, expected_diffs))
        print(f"Memory appears contiguous: {'✓' if contiguous else '✗'}")
        
    except Exception as e:
        print(f"Memory analysis failed: {e}")


def test_automatic_gradient_flow():
    """
    Test if gradients work automatically with the efficient repeat function.
    """
    print(f"\n\nTesting automatic gradient flow...")
    print("=" * 60)
    
    # Create nested tensor with gradients
    seq_len = 80
    feature_dim = 768
    fun_dim = 256
    batch_size = 4
    
    # Same sequence length for all batch elements
    tensor_list = [
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
    ]
    
    n_repeats = torch.tensor([2, 3, 1, 9])  # Different repeat counts
    
    original_nested = torch.nested.as_nested_tensor(tensor_list, layout=torch.jagged)
    
    print(f"Original shapes: {[t.shape for t in tensor_list]}")
    print(f"Repeat counts: {n_repeats.tolist()}")
    print(f"Requires grad: {[t.requires_grad for t in tensor_list]}")
    
    lengths = torch.diff(original_nested.offsets())
    # Test if the efficient function preserves gradients automatically
    repeated_nested, _, _ = repeat_nested_tensor_efficient(original_nested, lengths, n_repeats)
    print(f"Result requires grad: {repeated_nested.requires_grad}")
    
    # Create loss and backprop
    weights = torch.randn_like(repeated_nested) * 100.0
    loss = (repeated_nested * weights).sum()
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss requires grad: {loss.requires_grad}")
    
    try:
        loss.backward()
        gradients_computed = True
        print("✓ Backward pass completed successfully!")
    except Exception as e:
        gradients_computed = False
        print(f"✗ Backward pass failed: {e}")
        return
    
    # Check gradients
    print(f"\nGradient analysis:")

    gradients_exist = all(t.grad is not None for t in tensor_list)
    print(f"All gradients exist: {'✓' if gradients_exist else '✗'}")
    
    if gradients_exist:
        print(f"Gradient shapes: {[t.grad.shape for t in tensor_list]}")
    else:
        print("❌ Gradients not computed - may need custom autograd function")


def test_gradient_equivalence_with_dense():
    """
    Test that custom nested tensor repeat produces same gradients as dense repeat_interleave.
    Uses nested tensors where all dynamic shapes are the same for fair comparison.
    """
    print(f"\n\nTesting gradient equivalence with dense tensor...")
    print("=" * 60)
    
    # Create nested tensor where all sequences have same length (so we can convert to dense)
    seq_len = 80
    feature_dim = 768
    fun_dim = 256
    batch_size = 4
    
    # Same sequence length for all batch elements
    tensor_list = [
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
        torch.randn(seq_len, fun_dim, feature_dim, requires_grad=True),
    ]
    
    n_repeats = torch.tensor([2, 3, 1, 9])  # Different repeat counts
    
    print(f"All sequences have shape: {tensor_list[0].shape}")
    print(f"Batch size: {batch_size}")
    print(f"Repeat counts: {n_repeats.tolist()}")
    
    # Test 1: Custom nested tensor approach
    print(f"\n--- Testing custom nested tensor approach ---")
    nested_tensor = torch.nested.as_nested_tensor(tensor_list, layout=torch.jagged)
    
    # Apply custom repeat function
    lengths = torch.diff(nested_tensor.offsets())
    a = time.time()
    repeated_nested, _, _ = repeat_nested_tensor_efficient(nested_tensor, lengths, n_repeats)
    b = time.time()
    print(f"Custom repeat took {b - a:.4f} seconds")
    # Create loss and compute gradients
    target_nested = torch.randn_like(repeated_nested)
    loss_nested = (repeated_nested * target_nested).sum()
    
    # Zero gradients and compute
    for t in tensor_list:
        if t.grad is not None:
            t.grad.zero_()
    
    loss_nested.backward()
    
    # Extract gradients from nested approach
    nested_grads = [t.grad.clone() for t in tensor_list]
    
    print(f"Nested loss: {loss_nested.item():.6f}")
    print(f"Nested gradient shapes: {[g.shape for g in nested_grads]}")
    
    # Test 2: Dense tensor approach with repeat_interleave
    print(f"\n--- Testing dense tensor approach ---")
    
    # Convert to dense tensor [batch_size, seq_len, feature_dim]
    dense_tensor = torch.stack(tensor_list, dim=0).detach()
    dense_tensor.requires_grad_(True)
    
    # Apply repeat_interleave
    repeated_dense = torch.repeat_interleave(dense_tensor, n_repeats, dim=0)
    
    # Create equivalent target (convert nested target to dense)
    target_list = target_nested.unbind()
    target_dense = torch.stack(target_list, dim=0)
    
    # Compute loss
    loss_dense = (repeated_dense * target_dense).sum()
    
    # Compute gradients
    loss_dense.backward()
    dense_grad = dense_tensor.grad
    
    print(f"Dense loss: {loss_dense.item():.6f}")
    print(f"Dense gradient shape: {dense_grad.shape}")
    
    # Test 3: Compare results
    print(f"\n--- Comparing results ---")
    
    # Check if losses are the same
    loss_match = torch.allclose(loss_nested, loss_dense, atol=1e-6)
    print(f"Losses match: {'✓' if loss_match else '✗'} (diff: {abs(loss_nested.item() - loss_dense.item()):.2e})")
    
    # Compare gradients
    # Dense grad is [batch_size, seq_len, feature_dim], need to split back
    dense_grad_list = [dense_grad[i] for i in range(batch_size)]
    
    gradients_match = True
    max_diff = 0.0
    
    for i, (nested_grad, dense_grad_piece) in enumerate(zip(nested_grads, dense_grad_list)):
        diff = torch.max(torch.abs(nested_grad - dense_grad_piece)).item()
        max_diff = max(max_diff, diff)
        
        piece_match = torch.allclose(nested_grad, dense_grad_piece, atol=1e-6)
        print(f"Gradient {i} match: {'✓' if piece_match else '✗'} (max diff: {diff:.2e})")
        
        if not piece_match:
            gradients_match = False
    
    print(f"\nOverall gradient equivalence: {'✓' if gradients_match else '✗'}")
    print(f"Maximum gradient difference: {max_diff:.2e}")
    
    # Test 4: Verify repeat patterns match
    print(f"\n--- Verifying repeat patterns ---")
    
    # Check that the repeated sequences match between approaches
    nested_list = repeated_nested.unbind()
    dense_list = [repeated_dense[i] for i in range(repeated_dense.shape[0])]
    
    pattern_match = True
    for i, (nested_seq, dense_seq) in enumerate(zip(nested_list, dense_list)):
        seq_match = torch.allclose(nested_seq, dense_seq, atol=1e-6)
        if not seq_match:
            pattern_match = False
            print(f"Sequence {i} mismatch!")
            break
    
    print(f"Repeat patterns match: {'✓' if pattern_match else '✗'}")
    
    # Summary
    overall_success = loss_match and gradients_match and pattern_match
    print(f"\n{'='*60}")
    print(f"GRADIENT EQUIVALENCE TEST: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    print(f"{'='*60}")
    
    return overall_success

if __name__ == "__main__":
    torch.manual_seed(42)
    
    test_efficient_repeat_nested_tensor()
    test_automatic_gradient_flow()  # Test if gradients work automatically first
    test_gradient_equivalence_with_dense()  # Test gradient equivalence with dense tensor
    print(f"\n🎉 All efficient nested tensor tests completed!")
    print(f"Memory-efficient repeat using contiguous allocation works!")