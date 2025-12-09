# Testing for rope layer implementations
import pytest
import torch

from tarp.model.layers.positional.rotational import RotaryPositionalEmbedding


@pytest.fixture
def rope_layer() -> RotaryPositionalEmbedding:
    return RotaryPositionalEmbedding(
        head_dimension=8,
        rotational_fraction=0.5,
        maximum_sequence_length=16,
        base=10000,
    )


@pytest.fixture
def full_rope_layer() -> RotaryPositionalEmbedding:
    """RoPE with full rotation (rotational_fraction=1.0)"""
    return RotaryPositionalEmbedding(
        head_dimension=8,
        rotational_fraction=1.0,
        maximum_sequence_length=16,
        base=10000,
    )


@pytest.fixture
def large_rope_layer() -> RotaryPositionalEmbedding:
    """RoPE with larger dimensions for more realistic testing"""
    return RotaryPositionalEmbedding(
        head_dimension=64,
        rotational_fraction=0.5,
        maximum_sequence_length=512,
        base=10000,
    )


def test_rope_layer_initialization(rope_layer: RotaryPositionalEmbedding):
    assert rope_layer.dimension == 8
    assert rope_layer.rotary_dimension == 4
    assert rope_layer.maximum_sequence_length == 16
    assert rope_layer.base == 10000
    assert rope_layer.inverse_frequencies.shape == (2,)


def test_full_rotation_initialization(full_rope_layer: RotaryPositionalEmbedding):
    """Test initialization with full rotation"""
    assert full_rope_layer.dimension == 8
    assert full_rope_layer.rotary_dimension == 8
    assert full_rope_layer.inverse_frequencies.shape == (4,)


def test_large_dimension_initialization(large_rope_layer: RotaryPositionalEmbedding):
    """Test initialization with realistic large dimensions"""
    assert large_rope_layer.dimension == 64
    assert large_rope_layer.rotary_dimension == 32
    assert large_rope_layer.inverse_frequencies.shape == (16,)


def test_relative_rotational_covariance(rope_layer: RotaryPositionalEmbedding):
    rope_layer.reset_parameters()
    batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 8
    # The RoPE property only holds if the base Q and K vectors are identical at the shifted positions.
    q_base = torch.randn(batch_size, num_heads, 1, head_dim)
    k_base = torch.randn(batch_size, num_heads, 1, head_dim)
    # Expand the base vectors across the sequence length
    query = q_base.expand(-1, -1, seq_length, -1).clone()
    key = k_base.expand(-1, -1, seq_length, -1).clone()
    # Apply RoPE to the Q and K tensors
    rotated_query = rope_layer.rotate_query_or_key(query)
    rotated_key = rope_layer.rotate_query_or_key(key)
    # Relative offset r = 3 (n-m = 3)
    r = 3
    t = 2  # arbitrary shift t
    # 1. Original dot product: RoPE(q_0) * RoPE(k_r)
    dp_original = torch.einsum(
        "bhld,bhld->bh", rotated_query[:, :, 0:1, :], rotated_key[:, :, r : r + 1, :]
    )
    # 2. Shifted dot product: RoPE(q_t) * RoPE(k_{r+t})
    dp_shifted = torch.einsum(
        "bhld,bhld->bh",
        rotated_query[:, :, t : t + 1, :],
        rotated_key[:, :, r + t : r + t + 1, :],
    )
    # Assert that the dot products are approximately equal
    assert torch.allclose(dp_original, dp_shifted, atol=1e-5), (
        f"Dot products not equal: {dp_original} vs {dp_shifted}"
    )


def test_relative_rotational_covariance_multiple_offsets(
    rope_layer: RotaryPositionalEmbedding,
):
    """Test relative invariance with multiple different offsets and shifts"""
    rope_layer.reset_parameters()
    batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 8

    q_base = torch.randn(batch_size, num_heads, 1, head_dim)
    k_base = torch.randn(batch_size, num_heads, 1, head_dim)

    query = q_base.expand(-1, -1, seq_length, -1).clone()
    key = k_base.expand(-1, -1, seq_length, -1).clone()

    rotated_query = rope_layer.rotate_query_or_key(query)
    rotated_key = rope_layer.rotate_query_or_key(key)

    # Test multiple (r, t) combinations
    test_cases = [(1, 1), (2, 3), (3, 2), (4, 1), (1, 4)]

    for r, t in test_cases:
        dp_original = torch.einsum(
            "bhld,bhld->bh",
            rotated_query[:, :, 0:1, :],
            rotated_key[:, :, r : r + 1, :],
        )
        dp_shifted = torch.einsum(
            "bhld,bhld->bh",
            rotated_query[:, :, t : t + 1, :],
            rotated_key[:, :, r + t : r + t + 1, :],
        )
        assert torch.allclose(dp_original, dp_shifted, atol=1e-5), (
            f"Failed for r={r}, t={t}: {dp_original} vs {dp_shifted}"
        )


def test_relative_rotational_covariance_full_rotation(
    full_rope_layer: RotaryPositionalEmbedding,
):
    """Test relative invariance with full rotation (rotational_fraction=1.0)"""
    full_rope_layer.reset_parameters()
    batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 8

    q_base = torch.randn(batch_size, num_heads, 1, head_dim)
    k_base = torch.randn(batch_size, num_heads, 1, head_dim)

    query = q_base.expand(-1, -1, seq_length, -1).clone()
    key = k_base.expand(-1, -1, seq_length, -1).clone()

    rotated_query = full_rope_layer.rotate_query_or_key(query)
    rotated_key = full_rope_layer.rotate_query_or_key(key)

    r, t = 3, 2
    dp_original = torch.einsum(
        "bhld,bhld->bh", rotated_query[:, :, 0:1, :], rotated_key[:, :, r : r + 1, :]
    )
    dp_shifted = torch.einsum(
        "bhld,bhld->bh",
        rotated_query[:, :, t : t + 1, :],
        rotated_key[:, :, r + t : r + t + 1, :],
    )
    assert torch.allclose(dp_original, dp_shifted, atol=1e-5)


def test_rope_layer_output_shape(rope_layer: RotaryPositionalEmbedding):
    batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 8
    input_tensor = torch.randn(batch_size, num_heads, seq_length, head_dim)
    output_tensor = rope_layer.rotate_query_or_key(input_tensor)
    assert output_tensor.shape == input_tensor.shape


def test_rope_preserves_non_rotated_dimensions(rope_layer: RotaryPositionalEmbedding):
    """Test that dimensions beyond rotary_dimension remain unchanged"""
    batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 8
    input_tensor = torch.randn(batch_size, num_heads, seq_length, head_dim)
    output_tensor = rope_layer.rotate_query_or_key(input_tensor)

    # For rotational_fraction=0.5, rotary_dimension=4, so last 4 dimensions should be unchanged
    assert torch.allclose(input_tensor[..., 4:], output_tensor[..., 4:], atol=1e-6), (
        "Non-rotated dimensions should remain unchanged"
    )


def test_rope_at_position_zero(rope_layer: RotaryPositionalEmbedding):
    """Test that RoPE at position 0 should be identity-like (cos=1, sin=0)"""
    rope_layer.reset_parameters()
    batch_size, num_heads, head_dim = 2, 4, 8

    # Single position (position 0)
    input_tensor = torch.randn(batch_size, num_heads, 1, head_dim)
    output_tensor = rope_layer.rotate_query_or_key(input_tensor)

    # At position 0, cos(0)=1 and sin(0)=0, so rotation should be nearly identity
    # for the rotated dimensions
    assert torch.allclose(input_tensor[..., :4], output_tensor[..., :4], atol=1e-5), (
        "Position 0 should have minimal rotation effect"
    )


def test_rope_deterministic(rope_layer: RotaryPositionalEmbedding):
    """Test that RoPE produces deterministic outputs for same inputs"""
    batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 8
    input_tensor = torch.randn(batch_size, num_heads, seq_length, head_dim)

    output1 = rope_layer.rotate_query_or_key(input_tensor)
    output2 = rope_layer.rotate_query_or_key(input_tensor)

    assert torch.allclose(output1, output2, atol=1e-8), "RoPE should be deterministic"


def test_rope_query_and_key_consistency(rope_layer: RotaryPositionalEmbedding):
    """Test that rotate_query_and_key produces same result as individual rotations"""
    batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 8
    query = torch.randn(batch_size, num_heads, seq_length, head_dim)
    key = torch.randn(batch_size, num_heads, seq_length, head_dim)

    # Using combined method
    rotated_q1, rotated_k1 = rope_layer.rotate_query_and_key(query, key)

    # Using individual methods
    rotated_q2 = rope_layer.rotate_query_or_key(query)
    rotated_k2 = rope_layer.rotate_query_or_key(key)

    assert torch.allclose(rotated_q1, rotated_q2, atol=1e-8)
    assert torch.allclose(rotated_k1, rotated_k2, atol=1e-8)


def test_rope_cache_recomputation(rope_layer: RotaryPositionalEmbedding):
    """Test that caches are properly recomputed for longer sequences"""
    batch_size, num_heads, head_dim = 2, 4, 8

    # First with short sequence
    short_seq = torch.randn(batch_size, num_heads, 5, head_dim)
    _ = rope_layer.rotate_query_or_key(short_seq)
    cache_len_short = rope_layer.cosine_cache.shape[2]

    # Then with longer sequence (beyond initial max)
    long_seq = torch.randn(batch_size, num_heads, 20, head_dim)
    _ = rope_layer.rotate_query_or_key(long_seq)
    cache_len_long = rope_layer.cosine_cache.shape[2]

    assert cache_len_long >= 20, "Cache should expand for longer sequences"
    assert cache_len_long > cache_len_short, (
        "Cache should be larger after longer sequence"
    )


def test_rope_rotate_input_shape(rope_layer: RotaryPositionalEmbedding):
    """Test rotate_input method with 3D tensors (no head dimension)"""
    batch_size, seq_length, dimension = 4, 10, 8
    input_tensor = torch.randn(batch_size, seq_length, dimension)
    output_tensor = rope_layer.rotate_input(input_tensor)

    assert output_tensor.shape == input_tensor.shape


def test_rope_rotate_input_consistency(rope_layer: RotaryPositionalEmbedding):
    """Test that rotate_input and forward are equivalent"""
    batch_size, seq_length, dimension = 4, 10, 8
    input_tensor = torch.randn(batch_size, seq_length, dimension)

    output1 = rope_layer.rotate_input(input_tensor)
    output2 = rope_layer.forward(input_tensor)

    assert torch.allclose(output1, output2, atol=1e-8)


def test_rope_different_batch_sizes(rope_layer: RotaryPositionalEmbedding):
    """Test RoPE works correctly with different batch sizes"""
    num_heads, seq_length, head_dim = 4, 10, 8

    for batch_size in [1, 2, 4, 8]:
        input_tensor = torch.randn(batch_size, num_heads, seq_length, head_dim)
        output_tensor = rope_layer.rotate_query_or_key(input_tensor)
        assert output_tensor.shape == input_tensor.shape


def test_rope_gradient_flow(rope_layer: RotaryPositionalEmbedding):
    """Test that gradients flow properly through RoPE"""
    batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 8
    input_tensor = torch.randn(
        batch_size, num_heads, seq_length, head_dim, requires_grad=True
    )

    output_tensor = rope_layer.rotate_query_or_key(input_tensor)
    loss = output_tensor.sum()
    loss.backward()

    assert input_tensor.grad is not None, "Gradients should flow through RoPE"
    assert not torch.isnan(input_tensor.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(input_tensor.grad).any(), "Gradients should not contain Inf"


def test_rope_inverse_frequencies_scaling(rope_layer: RotaryPositionalEmbedding):
    """Test that inverse frequencies follow the expected 1/base^(2i/d) pattern"""
    rope_layer.reset_parameters()
    inv_freq = rope_layer.inverse_frequencies

    # Assert inv_freq is a tensor
    assert isinstance(inv_freq, torch.Tensor), "Inverse frequencies should be a tensor"

    # Check that frequencies are decreasing
    for i in range(len(inv_freq) - 1):
        assert inv_freq[i] > inv_freq[i + 1], "Inverse frequencies should be decreasing"

    # Check the first frequency matches formula: 1 / base^(0/d) = 1
    expected_first = 1.0 / (rope_layer.base ** (0.0 / rope_layer.rotary_dimension))
    assert torch.allclose(inv_freq[0], torch.tensor(expected_first), atol=1e-6)


def test_rope_attention_pattern_symmetry(rope_layer: RotaryPositionalEmbedding):
    """Test that attention patterns show relative position symmetry"""
    rope_layer.reset_parameters()
    batch_size, num_heads, seq_length, head_dim = 1, 1, 8, 8

    # Create queries and keys that are the same
    qk_base = torch.randn(batch_size, num_heads, 1, head_dim)
    query = qk_base.expand(-1, -1, seq_length, -1).clone()
    key = qk_base.expand(-1, -1, seq_length, -1).clone()

    rotated_query = rope_layer.rotate_query_or_key(query)
    rotated_key = rope_layer.rotate_query_or_key(key)

    # Compute attention scores (simplified, without softmax)
    # Shape: (batch, heads, seq_q, seq_k)
    scores = torch.einsum("bhqd,bhkd->bhqk", rotated_query, rotated_key)

    # For same base vectors, scores should only depend on relative distance
    # Check that scores at same relative distance are similar
    scores_squeezed = scores[0, 0]  # (seq_q, seq_k)

    # Check diagonal (distance 0)
    diagonal_scores = torch.diag(scores_squeezed)
    assert torch.allclose(
        diagonal_scores, diagonal_scores[0].expand_as(diagonal_scores), atol=1e-5
    ), "Scores at distance 0 should be the same"


def test_rope_large_dimensions(large_rope_layer: RotaryPositionalEmbedding):
    """Test RoPE with realistic large dimensions"""
    batch_size, num_heads, seq_length, head_dim = 4, 8, 128, 64
    input_tensor = torch.randn(batch_size, num_heads, seq_length, head_dim)
    output_tensor = large_rope_layer.rotate_query_or_key(input_tensor)

    assert output_tensor.shape == input_tensor.shape
    assert not torch.isnan(output_tensor).any()
    assert not torch.isinf(output_tensor).any()


def test_rope_reset_parameters(rope_layer: RotaryPositionalEmbedding):
    """Test that reset_parameters properly reinitializes the layer"""
    # Assert that inverse frequencies is a tensor
    assert isinstance(rope_layer.inverse_frequencies, torch.Tensor)

    original_inv_freq = rope_layer.inverse_frequencies.clone()
    original_cos_cache = (
        rope_layer.cosine_cache.clone() if rope_layer.cosine_cache is not None else None
    )

    rope_layer.reset_parameters()

    # Inverse frequencies should be recomputed (should be same values)
    assert torch.allclose(rope_layer.inverse_frequencies, original_inv_freq, atol=1e-8)

    # Caches should be recomputed
    if original_cos_cache is not None:
        assert torch.allclose(rope_layer.cosine_cache, original_cos_cache, atol=1e-8)
