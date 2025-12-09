import pytest
import torch

from tarp.model.layers.pooling.learned import (
    LearnedPositionPooling,
    MultiHeadGatedSelfAttentionPooling,
    SelfAttentionPooling,
)


@pytest.mark.parametrize(
    "module_class, kwargs",
    [
        (LearnedPositionPooling, {"maximum_sequence_length": 10}),
        (SelfAttentionPooling, {"feature_dimension": 32}),
        (
            MultiHeadGatedSelfAttentionPooling,
            {"hidden_dimension": 32, "number_of_heads": 4},
        ),
    ],
)
def test_pooling_modules_forward_and_shapes(module_class, kwargs):
    batch, seq_len, hidden_dim = 4, 10, 32
    x = torch.randn(batch, seq_len, hidden_dim, requires_grad=True)
    mask = torch.ones(batch, seq_len)
    mask[:, -2:] = 0  # mask last two tokens

    module = module_class(**kwargs)
    out, att = module(x, mask, return_attention=True)

    # Output shape
    assert out.shape == (batch, hidden_dim)

    # Attention shape depends on module type
    if isinstance(module, MultiHeadGatedSelfAttentionPooling):
        assert att.shape == (batch, module.number_of_heads, seq_len)
    else:
        assert att.shape == (batch, seq_len, 1)


@pytest.mark.parametrize(
    "module_class, kwargs",
    [
        (LearnedPositionPooling, {"maximum_sequence_length": 10}),
        (SelfAttentionPooling, {"feature_dimension": 32}),
        (
            MultiHeadGatedSelfAttentionPooling,
            {"hidden_dimension": 32, "number_of_heads": 4},
        ),
    ],
)
def test_pooling_attention_masking(module_class, kwargs):
    batch, seq_len, hidden_dim = 4, 10, 32
    x = torch.randn(batch, seq_len, hidden_dim)
    mask = torch.ones(batch, seq_len)
    mask[:, -2:] = 0

    module = module_class(**kwargs)
    _, att = module(x, mask, return_attention=True)

    # Extract masked positions (final 2 tokens)
    if isinstance(module, MultiHeadGatedSelfAttentionPooling):
        masked_attention_sum = att[:, :, -2:].sum().item()
    else:
        masked_attention_sum = att[:, -2:, :].sum().item()

    # Masked positions should receive ~zero attention
    assert masked_attention_sum < 1e-5


@pytest.mark.parametrize(
    "module_class, kwargs",
    [
        (LearnedPositionPooling, {"maximum_sequence_length": 10}),
        (SelfAttentionPooling, {"feature_dimension": 32}),
        (
            MultiHeadGatedSelfAttentionPooling,
            {"hidden_dimension": 32, "number_of_heads": 4},
        ),
    ],
)
def test_pooling_backprop(module_class, kwargs):
    batch, seq_len, hidden_dim = 4, 10, 32
    x = torch.randn(batch, seq_len, hidden_dim, requires_grad=True)
    mask = torch.ones(batch, seq_len)
    mask[:, -2:] = 0

    module = module_class(**kwargs)
    out = module(x, mask, return_attention=False)

    # Backpropagate a simple loss
    out.sum().backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
