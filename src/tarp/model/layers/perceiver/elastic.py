from collections.abc import Callable
from typing import NamedTuple, final, override

import torch
from torch import Tensor, nn

from tarp.functional.kernels.log import log_gaussian


class ElasticOptimalTransportPerceiverOutput(NamedTuple):
    latent_tokens: Tensor
    latent_mask: Tensor
    latent_positions: Tensor
    latent_density: Tensor
    transport_plan: Tensor
    log_transport_plan: Tensor
    window_indices: Tensor


@final
class ElasticOptimalTransportPerceiver(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        window_radius: int,
        temperature: float = 0.1,
        overflow_threshold: float = 1.0,
        iterations: int = 6,
        bias: bool = False,
        dropout: float = 0.1,
        kernel_function: Callable[[Tensor, Tensor], Tensor] = log_gaussian,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.window_radius = window_radius
        self.temperature = temperature
        self.overflow_threshold = overflow_threshold
        self.iterations = iterations
        self.dropout = dropout
        self.window_width = 2 * window_radius + 1
        self.kernel_function = kernel_function

        # For drift and bandwidth hyperparameters
        self.hyperparameter_projection = nn.Linear(self.model_dimension, 2, bias=bias)

    @override
    def forward(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        latent_size: int,
        latent_lengths: Tensor,
    ) -> ElasticOptimalTransportPerceiverOutput:
        batch_size, sequence_length, _ = sequence_embeddings.size()
        dtype = sequence_embeddings.dtype
        device = sequence_embeddings.device
        epsilon = torch.finfo(dtype).eps
        negative_infinity = torch.finfo(dtype).min

        content_lengths = (
            attention_mask.sum(dim=1, keepdim=True).to(dtype).clamp_min(1)
        )  # [B, 1]
        latent_lengths = latent_lengths.clamp_max(latent_size).reshape(batch_size, 1)

        bandwidth_score, coordinate_score = self.hyperparameter_projection(
            sequence_embeddings
        ).split(1, dim=-1)  # [B, L, 1], [B, L, 1]

        # Spatial positions for each token
        latent_span = (latent_lengths - 1).clamp_min(1)
        source_span = (content_lengths - 1).clamp_min(1)
        step_size = (latent_span / source_span).unsqueeze(-1)  # [B, 1, 1]

        bandwidths = torch.sigmoid(bandwidth_score) * self.window_radius  # [B, L, 1]
        coordinate_shifts = step_size * torch.tanh(coordinate_score)  # [B, L, 1]

        source_coordinates = (
            (
                torch.arange(sequence_length, device=device, dtype=dtype)
                * step_size.squeeze(-1)
                + coordinate_shifts.squeeze(-1)
            )
            .clamp_min(0)
            .minimum(latent_lengths - 1)
            .unsqueeze(-1)
        )  # [B, L, 1]

        source_indices = source_coordinates.round().long()  # [B, L, 1]

        window_indices = source_indices + torch.arange(
            -self.window_radius, self.window_radius + 1, device=device
        )  # [B, L, W]

        boundary_mask = (window_indices >= 0) & (
            window_indices < latent_lengths.unsqueeze(-1)
        )  # (B, L, W)

        window_indices = window_indices.clamp(0, latent_size - 1)  # [B, L, W]

        distance_to_window = window_indices - source_coordinates  # [B, L, W]

        log_transport_cost = self.kernel_function(distance_to_window, bandwidths)

        routing_mask = boundary_mask.to(dtype) * attention_mask.unsqueeze(
            -1
        )  # [B, L, W]

        transport_mask = routing_mask.bool() & (
            log_transport_cost > negative_infinity
        )  # [B, L, W]

        log_potentials = (log_transport_cost / self.temperature).masked_fill(
            ~routing_mask.bool(), negative_infinity
        )

        log_sink_budgets = (content_lengths / latent_lengths).log()  # [B, 1]

        sink_dual_potentials = torch.zeros(
            batch_size,
            latent_size,
            device=device,
            dtype=dtype,
        )  # [B, T]
        source_dual_potentials = torch.zeros(
            batch_size,
            sequence_length,
            1,
            device=device,
            dtype=dtype,
        )  # [B, L, 1]

        overflow_buffer = torch.full(
            (batch_size, sequence_length, 1),
            fill_value=-self.overflow_threshold / self.temperature,
            device=device,
            dtype=dtype,
        ).masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), negative_infinity
        )  # [B, L, 1]

        batch_indices = torch.arange(batch_size, device=device).reshape(
            batch_size, 1, 1
        )

        latent_capacity_mask = (
            torch.arange(latent_size, device=device, dtype=dtype).unsqueeze(0)
            < latent_lengths
        )

        for _ in range(self.iterations):
            windowed_sink_potentials = sink_dual_potentials[
                batch_indices, window_indices
            ]  # [B, L, W]

            source_dual_potentials = (
                -torch.cat(
                    [log_potentials + windowed_sink_potentials, overflow_buffer],
                    dim=-1,
                )
                .logsumexp(dim=-1, keepdim=True)
                .masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
            )  # [B, L, 1]

            window_potentials = source_dual_potentials + log_potentials  # [B, L, W]

            sink_maximum = torch.full_like(
                sink_dual_potentials, negative_infinity
            ).scatter_reduce_(
                1,
                window_indices.reshape(batch_size, -1),
                window_potentials.reshape(batch_size, -1),
                reduce="amax",
                include_self=True,
            )  # [B, T]

            gathered_sink_maximum = sink_maximum[
                batch_indices, window_indices
            ]  # [B, L, W]
            shifted_log_mass = (window_potentials - gathered_sink_maximum).masked_fill(
                ~transport_mask, negative_infinity
            )
            shifted_mass = shifted_log_mass.exp()
            aggregated_mass = torch.zeros_like(sink_dual_potentials).scatter_add_(
                1,
                window_indices.reshape(batch_size, -1),
                shifted_mass.reshape(batch_size, -1),
            )  # [B, T]
            log_sink_marginal = (
                aggregated_mass.clamp_min(epsilon).log() + sink_maximum
            )  # [B, T]
            sink_dual_potentials += torch.where(
                (aggregated_mass > 0) & latent_capacity_mask,
                log_sink_budgets - log_sink_marginal,
                torch.zeros_like(sink_dual_potentials),
            )

        windowed_sink_potentials = sink_dual_potentials[
            batch_indices, window_indices
        ]  # [B, L, W]

        source_dual_potentials = (
            -torch.cat(
                [log_potentials + windowed_sink_potentials, overflow_buffer],
                dim=-1,
            )
            .logsumexp(dim=-1, keepdim=True)
            .masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
        )  # [B, L, 1]

        log_transport_plan = (
            source_dual_potentials + log_potentials + windowed_sink_potentials
        ).masked_fill(~transport_mask, negative_infinity)  # [B, L, W]

        transport_plan = log_transport_plan.exp()  # [B, L, W]

        weighted_embeddings = torch.einsum(
            "blw,bld->blwd", transport_plan, sequence_embeddings
        ).to(dtype)  # [B, L, W, D]

        latent_features = torch.zeros(
            batch_size, latent_size, self.model_dimension, device=device, dtype=dtype
        ).scatter_add_(
            1,
            window_indices.unsqueeze(-1)
            .expand(-1, -1, -1, self.model_dimension)
            .reshape(batch_size, -1, self.model_dimension),
            weighted_embeddings.reshape(batch_size, -1, self.model_dimension),
        )  # [B, T, D]

        latent_density = torch.zeros_like(sink_dual_potentials).scatter_add_(
            1,
            window_indices.reshape(batch_size, -1),
            transport_plan.reshape(batch_size, -1),
        )

        latent_anchor = latent_features / latent_density.unsqueeze(-1).clamp_min(
            epsilon
        )

        latent_mask = ((latent_density > epsilon) & latent_capacity_mask).to(dtype)

        latent_positions = (
            torch.arange(
                latent_size,
                device=device,
                dtype=dtype,
            ).unsqueeze(0)
            * (source_span / latent_span)
            * latent_mask
        )

        return ElasticOptimalTransportPerceiverOutput(
            latent_tokens=latent_anchor,
            latent_mask=latent_mask,
            latent_positions=latent_positions,
            latent_density=latent_density,
            transport_plan=transport_plan,
            log_transport_plan=log_transport_plan,
            window_indices=window_indices,
        )

    def reconstruct(
        self,
        latent_tokens: Tensor,
        transport_plan: Tensor,
        window_indices: Tensor,
    ) -> Tensor:
        """
        Reconstruct sequence embeddings from latent tokens using the transport plan.
        :param Tensor latent_tokens: (B, T, D)
        :param Tensor transport_plan: (B, L, W)
        :param Tensor window_indices: (B, L, W)
        :param Tensor routing_mask: (B, L, W)
        :return Tensor: Reconstructed sequence embeddings (B, L, D)
        """
        batch_size, _, _ = transport_plan.shape
        epsilon = torch.finfo(transport_plan.dtype).eps

        transport_plan_normalized = transport_plan / (
            transport_plan.sum(dim=-1, keepdim=True) + epsilon
        )  # (B, L, W)

        batch_indices = torch.arange(batch_size, device=latent_tokens.device).reshape(
            batch_size, 1, 1
        )
        localized_latent = latent_tokens[batch_indices, window_indices]  # (B, L, W, D)

        reconstructed_sequence = (
            transport_plan_normalized.unsqueeze(-1) * localized_latent
        ).sum(dim=2)  # (B, L, D)

        return reconstructed_sequence

    def pool(
        self, latent_tokens: Tensor, latent_density: Tensor, latent_mask: Tensor
    ) -> Tensor:
        weights = latent_density * latent_mask  # (B, T)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )  # (B, T)
        return (latent_tokens * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
