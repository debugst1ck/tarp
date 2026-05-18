import math
from collections.abc import Callable
from typing import NamedTuple, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.functional.kernels.log import log_gaussian
from tarp.model.layers.attention.windowed import (
    WindowedCrossAttentionWithPositionalEncoding,
)
from tarp.model.layers.convolution.masked import DensityNormalizedConvolution1D
from tarp.model.layers.positional.core import (
    HeterogeneousTransformativePositionalEncoding,
)
from tarp.model.layers.positional.rotational import ContinuousRotaryPositionalEncoding


class ElasticKernelDensityCompressionOutput(NamedTuple):
    tokens: Tensor
    sink_mask: Tensor
    sink_coordinates: Tensor
    window_sink_indices: Tensor
    window_mask: Tensor
    sequence_features: Tensor
    source_coordinates: Tensor
    window_sink_coordinates: Tensor
    reconstruction_bias: Tensor
    sink_mass: Tensor
    spilled_mass: Tensor
    total_coordinate_span: Tensor
    sink_allocation: Tensor

    def overflow_loss(
        self,
        minimum_slack: float = 0.02,
        maximum_overflow: float = 0.20,
    ) -> Tensor:
        # Don't throw away information
        upper = F.relu(self.spilled_mass - maximum_overflow)
        # Don't over-fit to lattice
        lower = F.relu(minimum_slack - self.spilled_mass)
        return lower + upper  # L1 penalty [B]

    def coordinate_budget_loss(
        self,
        sequence_mask: Tensor,
        minimum_resolution: float = 0.15,
        maximum_resolution: float = 0.35,
    ) -> Tensor:
        epsilon = torch.finfo(self.total_coordinate_span.dtype).eps
        valid_length = sequence_mask.sum(dim=-1).to(self.total_coordinate_span.dtype)
        actual_resolution = self.total_coordinate_span / valid_length.clamp_min(
            epsilon
        )  # [B]

        lower = F.relu(minimum_resolution - actual_resolution)
        upper = F.relu(actual_resolution - maximum_resolution)
        # L2 hinge loss
        return lower.square() + upper.square()  # [B]

    def entropy_loss(self) -> Tensor:
        epsilon = torch.finfo(self.tokens.dtype).eps
        window_mass = self.sink_allocation.sum(dim=-1)  # [B, L]
        window_mask = window_mass > epsilon

        distribution = self.sink_allocation / window_mass.unsqueeze(-1).clamp_min(
            epsilon
        )
        distribution = distribution.clamp_min(epsilon)

        entropy = -(distribution * distribution.log()).sum(dim=-1)  # [B, L]
        entropy = entropy * window_mask.to(entropy.dtype)

        return entropy.sum(dim=-1) / window_mask.sum(dim=-1).clamp_min(1)

    def spatial_dispersion_loss(self) -> Tensor:
        epsilon = torch.finfo(self.tokens.dtype).eps
        squared_distances = (
            self.source_coordinates.unsqueeze(-1) - self.window_sink_coordinates
        ).square()  # [B, L, W]

        window_mass = self.sink_allocation.sum(dim=-1)  # [B, L]
        window_mask = window_mass > epsilon

        expected_distance = (self.sink_allocation * squared_distances).sum(
            dim=-1
        ) / window_mass.clamp_min(epsilon)

        expected_distance = expected_distance * window_mask.to(expected_distance.dtype)
        return expected_distance.sum(dim=-1) / window_mask.sum(dim=-1).clamp_min(1)

    def sink_balance_loss(self) -> Tensor:
        epsilon = torch.finfo(self.sink_mass.dtype).eps
        masked_mass = self.sink_mass * self.sink_mask.to(self.sink_mass.dtype)
        total_mass = masked_mass.sum(dim=-1, keepdim=True).clamp_min(epsilon)
        distribution = masked_mass / total_mass
        return distribution.square().sum(dim=-1)


class ElasticKernelDensityCompression1D(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        resolution: float = 0.5,
        locality_radius: int = 6,
        positional_weight: float = 0.4,
        background_cost_payload: float = 2.0,
        minimum_budget_usage: float = 0.5,
        hidden_dimension: int | None = None,
        kernel_function: Callable[[Tensor, float | Tensor], Tensor] = log_gaussian,
        number_of_heads: int = 6,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert 0.0 <= minimum_budget_usage <= 1.0, (
            "minimum_budget_usage must be between 0 and 1."
        )
        assert 0.0 <= positional_weight <= resolution <= 1.0, (
            "positional_weight must be non-negative and less than resolution, and resolution must be at most 1 to ensure the model has a meaningful budget to assign to sinks and that the coordinate mass grows with sequence length in a reasonable way."
        )

        self.embedding_dimension = embedding_dimension
        self.resolution = resolution
        self.positional_weight = positional_weight
        self.background_cost_payload = background_cost_payload
        self.locality_radius = locality_radius
        self.minimum_budget_usage = minimum_budget_usage
        self.dropout = dropout
        self.kernel_function = kernel_function

        self.hidden_dimension = hidden_dimension or embedding_dimension
        self.window_size = 2 * locality_radius + 1

        self.mixing = DensityNormalizedConvolution1D(
            in_channels=self.embedding_dimension,
            out_channels=self.embedding_dimension,
            kernel_size=self.window_size,
            padding=self.locality_radius,
            bias=bias,
        )
        self.decoder = WindowedCrossAttentionWithPositionalEncoding(
            model_dimension=self.embedding_dimension,
            number_of_heads=number_of_heads,
            positional_encoder=HeterogeneousTransformativePositionalEncoding(
                ContinuousRotaryPositionalEncoding(
                    embedding_dimension // number_of_heads
                ),
                ContinuousRotaryPositionalEncoding(
                    embedding_dimension // number_of_heads
                ),
            ),
            dropout=dropout,
            bias=bias,
        )

        self.feature_head = nn.Sequential(
            nn.RMSNorm(self.embedding_dimension),
            nn.Linear(self.embedding_dimension, self.hidden_dimension, bias=bias),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )

        self.refining_head = nn.Sequential(
            nn.RMSNorm(self.embedding_dimension + 2),
            nn.Linear(self.embedding_dimension + 2, self.hidden_dimension, bias=bias),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )

        # source_potential (1), coordinate_density (1), assignment_sharpness (1)
        self.source_projections = nn.Linear(self.hidden_dimension, 3, bias=bias)

        self.budget_head = nn.Linear(self.hidden_dimension, 1)  # Summary

        self.affinity_head = nn.Linear(self.hidden_dimension + 1, 1)

        # embedding_dimension (D), vitality (1), reconstruction_sharpness (1)
        self.sink_projections = nn.Linear(
            self.hidden_dimension, self.embedding_dimension + 2, bias=bias
        )

    @override
    def forward(
        self, sequence: Tensor, sequence_mask: Tensor, payload_mask: Tensor
    ) -> ElasticKernelDensityCompressionOutput:
        batch_size, sequence_length, _ = sequence.shape
        device = sequence.device

        # Use neighborhood to better able to identify local patterns and assign content to sinks based on local context
        sequence_features: Tensor = sequence + self.mixing(
            sequence.transpose(1, 2), payload_mask.unsqueeze(1)
        ).transpose(1, 2)  # [B, L, D]

        payload = sequence_features * payload_mask.unsqueeze(-1)  # [B, L, D]
        payload_features: Tensor = self.feature_head(payload)  # [B, L, H]

        dtype = payload_features.dtype
        negative_infinity = torch.finfo(dtype).min
        epsilon = torch.finfo(dtype).eps

        (
            source_potential_scores,
            coordinate_density_scores,
            assignment_sharpness_scores,
        ) = self.source_projections(payload_features).unbind(dim=-1)  # 3 x [B, L]

        # How much budget do we have for content after considering the baseline positional cost?
        valid_sequence_lengths = sequence_mask.sum(dim=1)  # [B]
        adaptive_capacity = (
            (valid_sequence_lengths * self.resolution)
            - (valid_sequence_lengths * self.positional_weight)
        ).clamp_min(0.0)  # [B]

        # Controls local spatial scaling: higher scores stretch, lower scores compress.
        # Spatial priority scores based on coordinate densities
        spatial_priority = F.softmax(
            coordinate_density_scores.masked_fill(~payload_mask, negative_infinity),
            dim=-1,
        )  # [B, L]

        # How much content is there per token on average?
        summary = payload_features.sum(dim=1) / payload_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)  # [B, H]

        # How much of the available budget do we actually use?
        budget_usage_fraction = self.minimum_budget_usage + (
            1 - self.minimum_budget_usage
        ) * torch.sigmoid(self.budget_head(summary)).squeeze(-1)  # [B]

        # Total adaptive coordinate budget available beyond the baseline positional mass.
        adaptive_coordinate_budget = adaptive_capacity * budget_usage_fraction  # [B]

        # The per-token allocation of the coordinate budget
        # Controls how much each token contributes to the coordinates of the sinks.
        adaptive_coordinates = spatial_priority * adaptive_coordinate_budget.unsqueeze(
            -1
        )  # [B, L]

        # Combines the base line positional cost and the adaptive coordinate allocation.
        # Density of sinks at a position is the sum of a fixed positional bias and the adaptive allocation.
        lattice_density = (
            self.positional_weight * sequence_mask.to(adaptive_coordinates.dtype)
        ) + adaptive_coordinates  # [B, L]

        # Center of each token coordinate interval
        source_coordinates = torch.cumsum(lattice_density, dim=1) - (
            lattice_density / 2
        )

        # How much total coordinate mass do we have across the sequence?
        # This controls how many sinks we can fill and is used to normalize the token coordinates to a [0, 1] range.
        total_coordinate_span = lattice_density.sum(dim=1, keepdim=True)  # [B, 1]

        # Make sure at least one sink is always available and that the number of sinks grows.
        lattice_size = max(1, math.ceil(sequence_length * self.resolution))

        # Lattice of sink centers based on the maximum number of sinks.
        sink_coordinates = (
            torch.arange(
                lattice_size, device=device, dtype=adaptive_coordinates.dtype
            ).unsqueeze(0)
            + 0.5
        ).expand(batch_size, -1)  # [B, T]

        # Masks to prune unused sink if the total coordinate mass is small.
        sink_mask = sink_coordinates < total_coordinate_span  # [B, T]
        # Ensure at least the first sink is always active if the sequence is not empty.
        sink_mask[:, :1] = sink_mask[:, :1] | sequence_mask.any(dim=1, keepdim=True)

        # Define the offsets of the local window around each sink center.
        window_offsets = torch.arange(
            -self.locality_radius,
            self.locality_radius + 1,
            device=device,
            dtype=torch.long,
        ).reshape(1, 1, -1)

        # Window selection is discrete; gradients still flow through cumulative_position
        # Indices of the tokens that fall into the local window around each sink center.
        window_sink_indices = (
            torch.round(source_coordinates - 0.5).long().unsqueeze(-1) + window_offsets
        )  # [B, L, W]

        # Mask to indicate which tokens are in the local window of each sink.
        within_bounds_mask = (window_sink_indices >= 0) & (
            window_sink_indices < lattice_size
        )  # [B, L, W]

        # Clamp the indices to ensure they are within the valid range of sinks.
        window_sink_indices = window_sink_indices.clamp(
            0, lattice_size - 1
        )  # [B, L, W]

        # Batch offsets for indexing into the flattened sink token tensor.
        batch_indices = torch.arange(batch_size, device=device).view(
            batch_size, 1, 1
        )  # [B, 1, 1]

        # Mask to indicate which tokens can contribute to which sinks based on the local window and the validity of the sinks.
        window_mask = (
            sink_mask[batch_indices, window_sink_indices] & within_bounds_mask
        )  # [B, L, W]

        # Gather the centers of the sinks in each token's local window.
        window_sink_coordinates = sink_coordinates[
            batch_indices, window_sink_indices
        ]  # [B, L, W]

        coordinate_drift = (
            source_coordinates.unsqueeze(-1) - window_sink_coordinates
        )  # [B, L, W]

        # How well do the features of the tokens in the payload match with the sinks in their local window?
        affinity_scores = self.kernel_function(
            coordinate_drift,
            F.softplus(
                assignment_sharpness_scores.masked_fill(~payload_mask, 0.0)
            ).unsqueeze(-1)
            + epsilon,
        ).masked_fill(
            ~(window_mask & payload_mask.bool().unsqueeze(-1)), negative_infinity
        )  # [B, L, W]

        # Background is an extra write target. It absorbs payload mass that should not be represented by any sink
        # Non payload tokens are going to go to background by default.
        overflow_scores = torch.full(
            (batch_size, sequence_length, 1),
            -self.background_cost_payload,
            device=device,
            dtype=affinity_scores.dtype,
        ).masked_fill(~payload_mask.unsqueeze(-1), 0.0)  # [B, L, 1]

        content_scores = (
            self.affinity_head(
                torch.cat(
                    [
                        payload_features.unsqueeze(-2).expand(
                            -1, -1, self.window_size, -1
                        ),  # [B, L, W, H]
                        coordinate_drift.unsqueeze(-1),  # [B, L, W, 1]
                    ],
                    dim=-1,
                )  # [B, L, W, H + 1])
            )
            .squeeze(-1)
            .masked_fill(~(window_mask & payload_mask.unsqueeze(-1)), negative_infinity)
        )  # [B, L, W]

        # Make a distribution for combined assignment to sinks and background
        transport_plan = F.softmax(
            torch.cat([affinity_scores + content_scores, overflow_scores], dim=-1),
            dim=-1,
        )  # [B, L, W + 1]

        sink_allocation = transport_plan[..., :-1]  # [B, L, W]
        overflow_allocation = transport_plan[..., -1]  # [B, L]

        # How much mass is there to write in sinks based on the features of the payload tokens?

        # Representing the proportion of the payload budget that should be assigned to each token.
        source_mass = F.softmax(
            source_potential_scores.masked_fill(~payload_mask, negative_infinity),
            dim=-1,
        )  # [B, L]

        # mass that is going to sinks vs overflow
        flowing_mass = sink_allocation * source_mass.unsqueeze(-1)  # [B, L, W]
        spilled_mass = (overflow_allocation * source_mass).sum(dim=-1)  # [B]

        flattened_window_indices = window_sink_indices.reshape(
            batch_size, sequence_length * self.window_size
        )  # [B, L*W]

        flattened_flowing_mass = flowing_mass.reshape(
            batch_size, sequence_length * self.window_size
        )  # [B, L*W]

        # Accumulate the mass assigned to each sink from the tokens in its local window.
        sink_mass = torch.zeros(
            batch_size, lattice_size, device=device, dtype=flattened_flowing_mass.dtype
        ).scatter_add_(
            dim=1, index=flattened_window_indices, src=flattened_flowing_mass
        )  # [B, T]

        # # Compute the flux (mass * features) from each source token to its assigned sinks.
        # This is the content that will be aggregated into the sinks.
        flowing_flux = torch.einsum("blw, bld -> blwd", flowing_mass, sequence).reshape(
            batch_size,
            sequence_length * self.window_size,
            self.embedding_dimension,
        )  # [B, L*W, D]

        # Aggregate the transported flux into each sink.
        sink_accumulation = torch.zeros(
            batch_size,
            lattice_size,
            self.embedding_dimension,
            device=device,
            dtype=flowing_flux.dtype,
        ).scatter_add_(
            dim=1,
            index=flattened_window_indices.unsqueeze(-1).expand(
                batch_size,
                sequence_length * self.window_size,
                self.embedding_dimension,
            ),
            src=flowing_flux,
        )  # [B, T, D]

        # The latent representations (centroids) derived from the accumulated mass.
        tokens = sink_accumulation / sink_mass.unsqueeze(-1).clamp_min(
            epsilon
        )  # [B, T, D]

        sink_descriptors = torch.cat(
            [
                tokens,
                sink_mass.unsqueeze(-1),
                sink_mass.unsqueeze(-1).clamp_min(epsilon).log(),
            ],
            dim=-1,
        ) * sink_mask.unsqueeze(-1)  # [B, T, D + 2]

        # Project descriptors into a refinement space
        sink_features = self.refining_head(sink_descriptors) * sink_mask.unsqueeze(
            -1
        )  # [B, T, H]

        sink_refinement, vitality_scores, reconstruction_sharpness_scores = torch.split(
            self.sink_projections(sink_features),
            [self.embedding_dimension, 1, 1],
            dim=-1,
        )  # [B, T, D], [B, T, 1], [B, T, 1]

        # Refine the latent tokens using the learned residuals.
        tokens = tokens + sink_refinement  # [B, T, D]
        tokens = tokens * sink_mask.unsqueeze(-1)  # [B, T, D]

        sink_vitality = F.logsigmoid(vitality_scores).squeeze(-1)  # [B, T]

        attention_bias = sink_vitality.masked_fill(
            ~sink_mask, negative_infinity
        )  # [B, T]

        window_occupancy_bias = attention_bias[
            batch_indices, window_sink_indices
        ]  # [B, L, W]

        sink_reconstruction_sharpness = (
            F.softplus(reconstruction_sharpness_scores) + epsilon
        )  # [B, T, 1]

        window_sharpness = sink_reconstruction_sharpness[
            batch_indices, window_sink_indices
        ].squeeze(-1)  # [B, L, W]

        reconstruction_affinity = self.kernel_function(
            coordinate_drift,
            window_sharpness,
        )  # [B, L, W]

        reconstruction_bias = (
            window_occupancy_bias + reconstruction_affinity
        ).masked_fill(~(window_mask & sequence_mask.unsqueeze(-1)), negative_infinity)

        # For positional embeddings for later transformer model we expose where sinks are in sequence
        weighted_source_contributions = (
            flowing_mass * source_coordinates.unsqueeze(-1)
        ).reshape(batch_size, sequence_length * self.window_size)

        centroid_positions = torch.zeros(
            batch_size,
            lattice_size,
            device=device,
            dtype=weighted_source_contributions.dtype,
        ).scatter_add_(
            dim=1,
            index=flattened_window_indices,
            src=weighted_source_contributions,
        ) / sink_mass.clamp_min(epsilon)  # [B, T]

        return ElasticKernelDensityCompressionOutput(
            tokens,  # [B, T, D]
            sink_mask,  # [B, T]
            centroid_positions,  # [B, T]
            window_sink_indices,  # [B, L, W]
            window_mask,  # [B, L, W]
            sequence_features,  # [B, L, D]
            source_coordinates,  # [B, L]
            window_sink_coordinates,  # [B, L, W]
            reconstruction_bias,  # [B, L, W]
            sink_mass,  # [B, T]
            spilled_mass,  # [B]
            total_coordinate_span.squeeze(-1),  # [B]
            sink_allocation,  # [B, L, W]
        )

    def reconstruct(
        self,
        attended_tokens: Tensor,
        sequence_features: Tensor,
        window_sink_indices: Tensor,
        window_mask: Tensor,
        reconstruction_bias: Tensor,
        source_positions: Tensor,
        sink_positions: Tensor,
    ) -> Tensor:
        """
        Reconstructs the original sequence from the globally attended tokens using windowed cross attention.
        :param Tensor attended_tokens: The attended tokens representing the sinks, of shape [B, T, D].
        :param Tensor sequence_features: The original sequence features of shape [B, L, D].
        :param Tensor window_sink_indices: The indices of the sinks in the local window for each token, of shape [B, L, W].
        :param Tensor window_mask: The mask indicating which sinks are in the local window for each token, of shape [B, L, W].
        :param Tensor reconstruction_bias: The attention bias for reconstruction, of shape [B, L, W].
        :param Tensor source_positions: The positional coordinates of the source tokens, of shape [B, L].
        :param Tensor sink_positions: The positional coordinates of the sink tokens, of shape [B, T].
        :return Tensor: The reconstructed sequence features of shape [B, L, D].
        """
        if self.training:
            dropout_mask = (
                torch.rand(reconstruction_bias.shape, device=reconstruction_bias.device)
                < self.dropout
            ) & window_mask
            reconstruction_bias = reconstruction_bias.masked_fill(
                dropout_mask, torch.finfo(reconstruction_bias.dtype).min
            )
        return self.decoder(
            query=sequence_features,
            key=attended_tokens,
            value=None,
            attention_bias=reconstruction_bias,
            routing_indices=window_sink_indices,
            window_mask=window_mask,
            query_positions=source_positions,
            key_positions=sink_positions,
        )

    def pooling(self, tokens: Tensor, sink_mass: Tensor, sink_mask: Tensor) -> Tensor:
        dtype = tokens.dtype
        epsilon = torch.finfo(dtype).eps
        masked_tokens = tokens * sink_mask.unsqueeze(-1).to(dtype)  # [B, T, D]
        masked_mass = sink_mass * sink_mask.to(dtype)  # [B, T]
        pooled = (masked_tokens * masked_mass.unsqueeze(-1)).sum(dim=1) / (
            masked_mass.sum(dim=1, keepdim=True).clamp_min(epsilon)
        )  # [B, D]
        return pooled
