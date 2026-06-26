from collections.abc import Callable
from typing import final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.functional.activations.smooth import inverse_softplus
from tarp.functional.kernels.log import log_gaussian
from tarp.model.layers.attention.windowed import WindowedCrossAttention


@final
class ChunkedEncoder1D(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        window_radius: int,
        bias: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = 2 * window_radius + 1
        self.radius = window_radius

        self.score_projection = nn.Linear(model_dimension, 1, bias=bias)
        self.value_projection = nn.Linear(model_dimension, model_dimension, bias=bias)
        self.output_projection = nn.Linear(model_dimension, model_dimension, bias=bias)
        self.normalization = nn.RMSNorm(model_dimension)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(
        self, features: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        _, sequence_length, _ = features.shape
        dtype = features.dtype
        device = features.device

        padded_features = F.pad(
            features,
            (0, 0, self.radius, self.radius),
            value=0.0,
        )  # [B, L + 2R, D]

        padded_mask = F.pad(
            attention_mask,
            (self.radius, self.radius),
            value=0.0,
        )  # [B, L + 2R]

        windows = padded_features.unfold(
            dimension=1, size=self.window_size, step=self.radius
        ).permute(0, 1, 3, 2)  # [B, T, W, D]

        window_mask = padded_mask.unfold(
            dimension=1,
            size=self.window_size,
            step=self.radius,
        )  # [B, T, W]

        latent_mask = window_mask.any(dim=-1)  # [B, T]

        scores = self.score_projection(windows).squeeze(-1)  # [B, T, W]
        scores = scores.masked_fill(
            ~window_mask.bool(), torch.finfo(scores.dtype).min
        )  # [B, T, W]

        weights = F.softmax(scores, dim=-1)  # [B, T, W]
        weights = weights * window_mask.to(dtype=weights.dtype)  # [B, T, W]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )  # [B, T, W]

        values = self.value_projection(windows)  # [B, T, W, D]

        latents = torch.einsum("btw,btwd->btd", weights, values)  # [B, T, D]

        latents = self.output_projection(latents)  # [B, T, D]

        latents = latents.masked_fill(
            ~latent_mask.unsqueeze(-1).bool(), 0.0
        )  # TO prevent normalization skewing due to all-zero latents

        latents = self.dropout(self.normalization(latents))  # [B, T, D]

        latents = latents * latent_mask.unsqueeze(-1).to(
            dtype=latents.dtype
        )  # [B, T, D]

        padded_positions = torch.arange(
            -self.radius,
            sequence_length + self.radius,
            device=device,
            dtype=dtype,
        )

        position_windows = padded_positions.unfold(
            dimension=0,
            size=self.window_size,
            step=self.radius,
        )  # [T, W]

        position_windows = position_windows.clamp(
            min=0,
            max=max(0, sequence_length - 1),
        )

        latent_positions = torch.einsum(
            "btw,tw->bt",
            weights,
            position_windows,
        )  # [B, T]

        latent_positions = latent_positions * latent_mask.to(dtype)

        return latents, latent_mask, latent_positions


@final
class ChunkedDecoder1D(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        window_radius: int,
        bias: bool = False,
        dropout: float = 0.1,
        kernel_function: Callable[[Tensor, Tensor], Tensor] = log_gaussian,
    ):
        super().__init__()
        self.window_radius = window_radius
        self.stride = window_radius
        self.kernel_function = kernel_function

        self.cross_attention = WindowedCrossAttention(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            bias=bias,
            dropout=dropout,
        )
        self.normalization = nn.RMSNorm(model_dimension)
        self.bandwidth_parameter = nn.Parameter(
            inverse_softplus(torch.tensor(float(window_radius)))
        )

    @override
    def forward(
        self,
        sequence_features: Tensor,  # [B, L, D]
        latent_tokens: Tensor,  # [B, T, D]
        attention_mask: Tensor,  # [B, L]
        latent_mask: Tensor,  # [B, T]
        latent_positions: Tensor,  # [B, T]
    ) -> Tensor:
        batch_size, sequence_length, _ = sequence_features.shape
        _, latent_length, _ = latent_tokens.shape

        device = sequence_features.device
        dtype = sequence_features.dtype
        epsilon = torch.finfo(dtype).eps

        sequence_positions = torch.arange(
            sequence_length,
            device=device,
        )  # [L]

        center_indices = torch.div(
            sequence_positions + self.stride // 2,
            self.stride,
            rounding_mode="floor",
        )  # [L]

        offsets = torch.tensor([-1, 0, 1], device=device)

        raw_indices = center_indices.unsqueeze(-1) + offsets
        # [L, 3]

        geometric_mask = (raw_indices >= 0) & (raw_indices < latent_length)
        # [L, 3]

        routing_indices = raw_indices.clamp(0, latent_length - 1)
        routing_indices = routing_indices.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )  # [B, L, 3]

        geometric_mask = geometric_mask.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )  # [B, L, 3]

        gathered_latent_mask = (
            latent_mask.bool()
            .gather(
                dim=1,
                index=routing_indices.reshape(batch_size, -1),
            )
            .reshape(batch_size, sequence_length, 3)
        )

        window_mask = (
            geometric_mask & gathered_latent_mask & attention_mask.bool().unsqueeze(-1)
        )  # [B, L, 3]

        routed_positions = latent_positions.gather(
            dim=1,
            index=routing_indices.reshape(batch_size, -1),
        ).reshape(batch_size, sequence_length, 3)

        distance = routed_positions - sequence_positions.to(dtype).reshape(
            1, sequence_length, 1
        )  # [B, L, 3]

        bandwidth = F.softplus(self.bandwidth_parameter).to(dtype).clamp_min(epsilon)
        attention_bias = self.kernel_function(distance, bandwidth)

        context = self.cross_attention(
            query=sequence_features,
            key=latent_tokens,
            value=None,
            attention_bias=attention_bias,
            routing_indices=routing_indices,
            window_mask=window_mask,
        )  # [B, L, D]

        output = self.normalization(sequence_features + context)
        return output * attention_mask.unsqueeze(-1).to(dtype)
