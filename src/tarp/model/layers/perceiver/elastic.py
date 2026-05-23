from typing import NamedTuple, final, override

import torch
from torch import Tensor, nn

from tarp.functional.kernels.linear import quartic


class ElasticOptimalTransportPerceiverOutput(NamedTuple):
    latent_tokens: Tensor
    latent_mask: Tensor
    latent_positions: Tensor
    transport_plan: Tensor
    window_indices: Tensor


@final
class ElasticOptimalTransportPerceiver(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        window_radius: int,
        temperature: float = 0.1,
        overflow_threshold: float = 1.0,
        iterations: int = 4,
        bias: bool = False,
        dropout: float = 0.05,
        hidden_dimension: int | None = None,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.window_radius = window_radius
        self.temperature = temperature
        self.overflow_threshold = overflow_threshold
        self.iterations = iterations
        self.bias = bias
        self.window_width = 2 * window_radius + 1
        self.hidden_dimension = hidden_dimension or model_dimension

        # For drift and bandwidth hyperparameters
        self.hyperparameter_projection = nn.Sequential(
            nn.Linear(model_dimension, self.hidden_dimension, bias=bias),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dimension, 2, bias=bias),
        )

    @override
    def forward(
        self, sequence_embeddings: Tensor, attention_mask: Tensor, latent_size: int
    ) -> ElasticOptimalTransportPerceiverOutput:
        batch_size, sequence_length, _ = sequence_embeddings.size()
        content_lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)  # (B, 1)
        attention_mask_expanded = attention_mask.unsqueeze(-1)  # (B, L, 1)

        epsilon = torch.finfo(sequence_embeddings.dtype).eps
        negative_infinity = torch.finfo(sequence_embeddings.dtype).min

        bandwidth_score, coordinate_score = self.hyperparameter_projection(
            sequence_embeddings
        ).split(1, dim=-1)  # (B, L, 1) x 2

        bandwidths = (
            torch.sigmoid(bandwidth_score) * self.window_radius + 0.5
        )  # (B, L, 1)

        latent_span = max(latent_size - 1, 1)
        source_span = (content_lengths - 1).clamp_min(1)

        step_size = (latent_span / source_span).unsqueeze(-1)  # (B, 1, 1)

        coordinate_shifts = 0.25 * step_size * torch.tanh(coordinate_score)  # (B, L, 1)
        original_coordinates = (attention_mask.cumsum(dim=1) - 1).unsqueeze(
            -1
        ) * step_size  # (B, L, 1)

        source_coordinates = (original_coordinates + coordinate_shifts).clamp(
            0, latent_size - 1
        )  # (B, L, 1)

        source_indices = source_coordinates.round().long()

        window_offsets = torch.arange(
            -self.window_radius,
            self.window_radius + 1,
            device=sequence_embeddings.device,
        ).reshape(1, 1, self.window_width)  # (1, 1, W)

        window_indices = source_indices + window_offsets  # (B, L, W)
        boundary_mask = (window_indices >= 0) & (
            window_indices < latent_size
        )  # (B, L, W)
        window_indices = window_indices.clamp(0, latent_size - 1)  # (B, L, W)

        # Distance from window indices to source coordinates
        distance_to_window = (
            window_indices.to(source_coordinates.dtype) - source_coordinates
        )

        # This is how much each token contributes to each window position, based on distance and bandwidth
        window_density = quartic(distance_to_window, bandwidths)  # (B, L, W)

        # Transport cost is one over the density, with overflow handling
        transport_cost = 1.0 / window_density.clamp_min(epsilon)  # (B, L, W)

        routing_mask = (
            boundary_mask.to(sequence_embeddings.dtype) * attention_mask_expanded
        )  # (B, L, W)

        log_potentials = (-transport_cost / self.temperature).masked_fill(
            ~routing_mask.bool(), negative_infinity
        )  # (B, L, W)

        log_source_budgets = torch.where(
            attention_mask > 0, 0.0, negative_infinity
        ).unsqueeze(-1)  # (B, L, 1)
        log_sink_budgets = (content_lengths / latent_size).log()  # (B, 1)

        sink_dual_potentials = torch.zeros(
            batch_size, latent_size, device=sequence_embeddings.device
        )  # (B, T)
        source_dual_potentials = torch.zeros(
            batch_size, sequence_length, 1, device=sequence_embeddings.device
        )  # (B, L, 1)
        overflow_buffer = torch.full(
            (batch_size, sequence_length, 1),
            fill_value=-self.overflow_threshold / self.temperature,
            device=sequence_embeddings.device,
            dtype=sequence_embeddings.dtype,
        ).masked_fill(~attention_mask_expanded.bool(), negative_infinity)

        batch_indices = torch.arange(
            batch_size, device=sequence_embeddings.device
        ).reshape(batch_size, 1, 1)  # (B, 1, 1)

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
            batch_max = window_potentials.flatten(1).max(dim=1, keepdim=True).values
            shifted_potentials_exponential = (
                window_potentials - batch_max.unsqueeze(-1)
            ).exp() * routing_mask  # (B, L, W)
            aggregated_mass = torch.zeros_like(sink_dual_potentials).scatter_add_(
                1,
                window_indices.reshape(batch_size, -1),
                shifted_potentials_exponential.reshape(batch_size, -1),
            )  # (B, T)
            sink_marginal_log = aggregated_mass.clamp_min(epsilon).log() + batch_max
            sink_dual_potentials = log_sink_budgets - sink_marginal_log  # (B, T)

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
            device=sequence_embeddings.device,
            dtype=sequence_embeddings.dtype,
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

        latent_mask = (latent_density > epsilon).to(sequence_embeddings.dtype)  # (B, T)

        # Positions used to calculate RoPE
        sink_grid = torch.arange(
            latent_size,
            device=sequence_embeddings.device,
            dtype=sequence_embeddings.dtype,
        ).unsqueeze(0)  # (1, T)
        latent_positions = (
            sink_grid * (content_lengths - 1).clamp_min(1) / (latent_size - 1)
        )  # (B, T)
        latent_positions = latent_positions * latent_mask

        return ElasticOptimalTransportPerceiverOutput(
            latent_tokens=latent_tokens,
            latent_mask=latent_mask,
            latent_positions=latent_positions,
            transport_plan=transport_plan,
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
