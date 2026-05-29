from collections.abc import Callable
from typing import NamedTuple, final, override

import torch
from torch import Tensor, nn

from tarp.functional.kernels.log import log_quartic


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
        hidden_dimension: int | None = None,
        kernel_function: Callable[[Tensor, Tensor], Tensor] = log_quartic,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.window_radius = window_radius
        self.temperature = temperature
        self.overflow_threshold = overflow_threshold
        self.iterations = iterations
        self.dropout = dropout
        self.window_width = 2 * window_radius + 1
        self.hidden_dimension = hidden_dimension or model_dimension
        self.kernel_function = kernel_function

        # For drift and bandwidth hyperparameters
        self.hyperparameter_projection = nn.Sequential(
            nn.Linear(model_dimension, self.hidden_dimension, bias=bias),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dimension, 2, bias=bias),
        )

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
        )  # (B, 1)
        attention_mask_expanded = attention_mask.unsqueeze(-1)  # (B, L, 1)
        latent_lengths = latent_lengths.clamp_max(latent_size).reshape(
            batch_size, 1
        )  # (B, 1)

        bandwidth_score, coordinate_score = self.hyperparameter_projection(
            sequence_embeddings
        ).split(1, dim=-1)  # (B, L, 1) x 2

        bandwidths = (
            torch.sigmoid(bandwidth_score) * self.window_radius + 0.5
        )  # (B, L, 1)

        latent_span = (latent_lengths - 1).clamp_min(1)
        source_span = (content_lengths - 1).clamp_min(1)
        step_size = (latent_span / source_span).unsqueeze(-1)  # (B, 1, 1)

        coordinate_shifts = 0.25 * step_size * torch.tanh(coordinate_score)  # (B, L, 1)
        original_coordinates = (attention_mask.cumsum(dim=1) - 1).unsqueeze(
            -1
        ) * step_size  # (B, L, 1)

        source_coordinates = torch.minimum(
            (original_coordinates + coordinate_shifts).clamp_min(0),
            (latent_lengths - 1).to(dtype).unsqueeze(-1),
        )

        source_indices = source_coordinates.round().long()

        window_offsets = torch.arange(
            -self.window_radius,
            self.window_radius + 1,
            device=device,
        ).reshape(1, 1, self.window_width)  # (1, 1, W)

        window_indices = source_indices + window_offsets  # (B, L, W)
        boundary_mask = (window_indices >= 0) & (
            window_indices < latent_lengths.unsqueeze(-1)
        )  # (B, L, W)
        window_indices = window_indices.clamp(0, latent_size - 1)  # (B, L, W)

        # Distance from window indices to source coordinates
        distance_to_window = window_indices.to(dtype) - source_coordinates

        # This is how much each token contributes to each window position, based on distance and bandwidth
        log_transport_cost = self.kernel_function(
            distance_to_window, bandwidths
        )  # (B, L, W)

        routing_mask = boundary_mask.to(dtype) * attention_mask_expanded  # (B, L, W)

        log_potentials = (
            (log_transport_cost / self.temperature)
            .clamp_min(-16.0)  # Logit floor
            .masked_fill(~routing_mask.bool(), negative_infinity)
        )  # (B, L, W)

        log_source_budgets = torch.where(
            attention_mask > 0, 0.0, negative_infinity
        ).unsqueeze(-1)  # (B, L, 1)
        log_sink_budgets = (content_lengths / latent_lengths).log()  # (B, 1)

        sink_dual_potentials = torch.zeros(
            batch_size,
            latent_size,
            device=device,
            dtype=dtype,
        )  # (B, T)
        source_dual_potentials = torch.zeros(
            batch_size,
            sequence_length,
            1,
            device=device,
            dtype=dtype,
        )  # (B, L, 1)
        overflow_buffer = torch.full(
            (batch_size, sequence_length, 1),
            fill_value=-self.overflow_threshold / self.temperature,
            device=device,
            dtype=dtype,
        ).masked_fill(~attention_mask_expanded.bool(), negative_infinity)

        latent_capacity_mask = (
            torch.arange(latent_size, device=device, dtype=dtype).unsqueeze(0)
            < latent_lengths
        )

        batch_indices = torch.arange(batch_size, device=device).reshape(
            batch_size, 1, 1
        )  # (B, 1, 1)

        for _ in range(self.iterations):
            windowed_sink_potentials = sink_dual_potentials[
                batch_indices, window_indices
            ]  # (B, L, W)

            # Row step: update source dual potentials to satisfy source budget constraints
            source_dual_potentials = log_source_budgets - torch.logsumexp(
                torch.cat(
                    [log_potentials + windowed_sink_potentials, overflow_buffer],
                    dim=-1,
                ),
                dim=-1,
                keepdim=True,
            )  # (B, L, 1)

            window_potentials = source_dual_potentials + log_potentials  # (B, L, W)

            sink_max = torch.full_like(
                sink_dual_potentials, negative_infinity
            ).scatter_reduce_(
                1,
                window_indices.reshape(batch_size, -1),
                window_potentials.reshape(batch_size, -1).masked_fill(
                    ~routing_mask.reshape(batch_size, -1).bool(), negative_infinity
                ),
                reduce="amax",
                include_self=True,
            )  # (B, T)
            gathered_sink_max = sink_max[batch_indices, window_indices]  # (B, L, W)
            shifted_potentials_exponential = (
                window_potentials - gathered_sink_max
            ).exp() * routing_mask  # (B, L, W)
            aggregated_mass = torch.zeros_like(sink_dual_potentials).scatter_add_(
                1,
                window_indices.reshape(batch_size, -1),
                shifted_potentials_exponential.reshape(batch_size, -1),
            )  # (B, T)
            sink_marginal_log = torch.where(
                aggregated_mass > 0,
                aggregated_mass.clamp_min(epsilon).log() + sink_max,
                torch.zeros_like(sink_max),
            )
            sink_dual_potentials = torch.where(
                aggregated_mass > 0,
                log_sink_budgets - sink_marginal_log,
                torch.zeros_like(sink_max),
            ).masked_fill(~latent_capacity_mask, 0.0)  # (B, T)

        windowed_sink_potentials = sink_dual_potentials[
            batch_indices, window_indices
        ]  # (B, L, W)

        source_dual_potentials = log_source_budgets - torch.logsumexp(
            torch.cat(
                [log_potentials + windowed_sink_potentials, overflow_buffer],
                dim=-1,
            ),
            dim=-1,
            keepdim=True,
        )

        log_transport_plan = (
            source_dual_potentials + log_potentials + windowed_sink_potentials
        )  # (B, L, W)

        transport_plan = log_transport_plan.exp() * routing_mask  # (B, L, W)

        weighted_embeddings = sequence_embeddings.unsqueeze(
            2
        ) * transport_plan.unsqueeze(-1)  # (B, L, W, D)

        expanded_window_indices = window_indices.unsqueeze(-1).expand(
            -1, -1, -1, self.model_dimension
        )  # (B, L, W, D)

        latent_features = torch.zeros(
            batch_size,
            latent_size,
            self.model_dimension,
            device=device,
            dtype=dtype,
        ).scatter_add_(
            1,
            expanded_window_indices.reshape(batch_size, -1, self.model_dimension),
            weighted_embeddings.reshape(batch_size, -1, self.model_dimension),
        )  # (B, T, D)

        latent_density = torch.zeros_like(sink_dual_potentials).scatter_add_(
            1,
            window_indices.reshape(batch_size, -1),
            transport_plan.reshape(batch_size, -1),
        )  # (B, T)

        # Normalize the latent features by the density to get the final compressed representation
        latent_tokens = latent_features / latent_density.unsqueeze(-1).clamp_min(
            epsilon
        )  # (B, T, D)

        latent_mask = ((latent_density > epsilon) & latent_capacity_mask).to(
            dtype
        )  # (B, T)

        # Positions used to calculate RoPE
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
            latent_tokens=latent_tokens,
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
