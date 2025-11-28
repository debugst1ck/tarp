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


def test_rope_layer_initialization(rope_layer: RotaryPositionalEmbedding):
    assert rope_layer.dimension == 8
    assert rope_layer.rotary_dimension == 4
    assert rope_layer.maximum_sequence_length == 16
    assert rope_layer.base == 10000
    assert rope_layer.inverse_frequencies.shape == (2,)


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
    assert torch.allclose(dp_original, dp_shifted), (
        f"Dot products not equal: {dp_original} vs {dp_shifted}"
    )


def test_rope_layer_output_shape(rope_layer: RotaryPositionalEmbedding):
    batch_size, num_heads, seq_length, head_dim = 2, 4, 10, 8
    input_tensor = torch.randn(batch_size, num_heads, seq_length, head_dim)
    output_tensor = rope_layer.rotate_query_or_key(input_tensor)
    assert output_tensor.shape == input_tensor.shape
