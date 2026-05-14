import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RadialBasisFunction(Enum):
    """
    Common radial basis functions for computing assignment scores based on distances between tokens and sink centers.
    The choice of function and its sharpness parameter can affect how mass is assigned to sinks based on distance,
    which in turn can influence the quality of the compression and reconstruction.
    """

    CAUCHY = "cauchy"
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    EPANECHNIKOV = "epanechnikov"
    RATIONAL_POWER = "rational_power"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    TRIWEIGHT = "triweight"

    def __call__(self, distances: Tensor, sharpness: Union[float, Tensor]) -> Tensor:
        """
        :param Tensor distances: Tensor of shape [B, L, W] representing the distances from each token to the centers of the sinks in its local window.
        :param Union[float, Tensor] sharpness: Positive decay factor, either a scalar or a tensor of shape [B, L, 1].
        :return Tensor: Tensor of shape [B, L, W] representing the log assignment scores for each token to each sink in its local window.
        """
        epsilon = torch.finfo(distances.dtype).eps
        match self:
            case RadialBasisFunction.CAUCHY:
                return -torch.log1p(sharpness * distances.square())
            case RadialBasisFunction.GAUSSIAN:
                return -sharpness * distances.square()
            case RadialBasisFunction.LAPLACE:
                return -sharpness * distances.abs()
            case RadialBasisFunction.EPANECHNIKOV:
                return torch.clamp(
                    1 - sharpness * distances.square(), min=epsilon
                ).log()
            case RadialBasisFunction.RATIONAL_POWER:
                return -sharpness * torch.log1p(distances.abs())
            case RadialBasisFunction.INVERSE_MULTIQUADRIC:
                return -0.5 * torch.log1p(sharpness * distances.square())
            case RadialBasisFunction.TRIWEIGHT:
                return (
                    3
                    * (
                        torch.clamp(1 - sharpness * distances.square(), min=epsilon)
                    ).log()
                )


class WindowedSinkCrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int = 8,
        bias: bool = False,
    ):
        super().__init__()
        assert embedding_dimension % number_of_heads == 0
        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        self.scale = math.sqrt(self.head_dimension)

        self.query_projection = nn.Linear(
            embedding_dimension, embedding_dimension, bias=bias
        )
        self.key_value_projection = nn.Linear(
            embedding_dimension, 2 * embedding_dimension, bias=bias
        )
        self.output_projection = nn.Linear(
            embedding_dimension, embedding_dimension, bias=bias
        )

    def forward(
        self,
        sequence_queries: Tensor,  # [B, L, D]
        latent_tokens: Tensor,  # [B, T, D]
        window_sink_indices: Tensor,  # [B, L, W]
        window_mask: Tensor,  # [B, L, W]
        reconstruction_bias: Optional[Tensor] = None,  # [B, L, W]
    ) -> Tensor:
        batch_size, sequence_length, embedding_dimension = sequence_queries.shape
        _, lattice_size, _ = latent_tokens.shape
        _, _, local_window = window_sink_indices.shape

        batch_offsets = (
            torch.arange(
                batch_size,
                device=sequence_queries.device,
                dtype=window_sink_indices.dtype,
            ).reshape(batch_size, 1, 1)
            * lattice_size
        )

        flattened_indices = (window_sink_indices + batch_offsets).flatten()

        flattened_tokens = latent_tokens.reshape(
            batch_size * lattice_size,
            embedding_dimension,
        )

        sink_features = flattened_tokens[flattened_indices].reshape(
            batch_size, sequence_length, local_window, embedding_dimension
        )  # [B, L, W, D]

        queries = self.query_projection(sequence_queries).reshape(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        )  # [B, L, H, Dh]

        keys, values = (
            self.key_value_projection(sink_features)
            .reshape(
                batch_size,
                sequence_length,
                local_window,
                2,
                self.number_of_heads,
                self.head_dimension,
            )
            .unbind(dim=3)
        )  # each [B, L, W, H, Dh]

        scores = (
            torch.einsum("blhd,blwhd->blhw", queries, keys) / self.scale
        )  # [B, L, H, W]

        epsilon = torch.finfo(scores.dtype).eps
        negative_infinity = torch.finfo(scores.dtype).min

        if reconstruction_bias is not None:
            scores = scores + reconstruction_bias.unsqueeze(2)  # [B, L, H, W]

        scores = scores.masked_fill(
            ~window_mask.unsqueeze(2), negative_infinity
        )  # [B, L, H, W]

        attention_weights = F.softmax(scores, dim=-1) * window_mask.unsqueeze(2).to(
            scores.dtype
        )

        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(epsilon)

        context = torch.einsum("blhw,blwhd->blhd", attention_weights, values).reshape(
            batch_size, sequence_length, embedding_dimension
        )  # [B, L, D]

        return self.output_projection(context)


class MaskedConvolution1D(nn.Module):
    weight_ones: Tensor

    def __init__(
        self,
        embedding_dimension: int,
        hidden_dimension: Optional[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()

        hidden_dimension = hidden_dimension or embedding_dimension
        self.padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv1d(
                embedding_dimension,
                hidden_dimension,
                kernel_size=kernel_size,
                padding=self.padding,
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv1d(
                hidden_dimension,
                embedding_dimension,
                kernel_size=kernel_size,
                padding=self.padding,
                bias=False,
            ),
        )

        self.register_buffer(
            "weight_ones", torch.ones(1, 1, kernel_size), persistent=False
        )

    def forward(self, sequence: Tensor, content_mask: Tensor) -> Tensor:
        """
        :param Tensor sequence: [B, L, D] input token features.
        :param Tensor content_mask: [B, L] positions allowed to contribute content.
        :return Tensor: [B, L, D] residual local features.
        """
        dtype = sequence.dtype
        mask = content_mask.to(dtype).unsqueeze(1)  # [B, 1, L]

        # Remove hidden-token content before convolution so it cannot leak locally.
        masked_sequence = sequence * content_mask.unsqueeze(-1).to(dtype)

        features = self.feature_conv(masked_sequence.transpose(1, 2))  # [B, D, L]

        # Count how much visible content was present in each convolution window.
        density = F.conv1d(
            mask,
            self.weight_ones.to(dtype=dtype, device=sequence.device),
            padding=self.padding,
        ).clamp_min(1.0)

        features = features / density  # [B, D, L]

        return features.transpose(1, 2)  # [B, L, D]


@dataclass
class LocalCumulativeKernelDensitySequenceCompressionOutput:
    tokens: Tensor
    sink_mask: Tensor
    sink_mass: Tensor
    window_sink_indices: Tensor
    window_mask: Tensor
    reconstruction_bias: Tensor
    spilled_mass: Tensor
    budget_usage_fraction: Tensor
    total_coordinate_span: Tensor
    sink_allocation: Tensor
    source_coordinates: Tensor
    window_sink_coordinates: Tensor
    sink_positions: Tensor

    def overflow_loss(
        self,
        minimum_slack: float = 0.02,
        maximum_overflow: float = 0.20,
    ) -> Tensor:
        """
        Computes a loss term based on the spilled mass that encourages the model to keep the spilled mass within a reasonable range.
        :param float minimum_spilled_mass: The minimum acceptable fraction of mass that can be overflowed.
        :param float maximum_spilled_mass: The maximum acceptable fraction of mass that can be overflowed.
        :return Tensor: A per batch loss term that penalizes spilled mass outside the specified range.
        """
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
        """
        Encourages the mass to be more evenly distributed across active sinks to prevent degenerate solutions where one sink dominates.
        """
        epsilon = torch.finfo(self.sink_mass.dtype).eps
        masked_mass = self.sink_mass * self.sink_mask.to(self.sink_mass.dtype)
        total_mass = masked_mass.sum(dim=-1, keepdim=True).clamp_min(epsilon)
        distribution = masked_mass / total_mass
        return distribution.square().sum(dim=-1)

    def auxiliary_loss(
        self,
        sequence_mask: Tensor,
        overflow_weight: float = 0.01,
        coordinate_budget_weight: float = 0.1,
        entropy_weight: float = 0.0,
        spatial_dispersion_weight: float = 0.0,
        sink_balance_weight: float = 0.005,
    ) -> Tensor:
        return (
            overflow_weight * self.overflow_loss()
            + coordinate_budget_weight * self.coordinate_budget_loss(sequence_mask)
            + entropy_weight * self.entropy_loss()
            + spatial_dispersion_weight * self.spatial_dispersion_loss()
            + sink_balance_weight * self.sink_balance_loss()
        ).mean()


class LocalCumulativeKernelDensitySequenceCompression(nn.Module):
    """ """

    def __init__(
        self,
        embedding_dimension: int,
        resolution: float = 0.5,  # Number of sinks per input token. For example, 0.5 means on average 1 sink for every 2 tokens.
        locality_radius: int = 6,
        positional_weight: float = 0.4,
        background_cost_payload: float = 2.0,
        minimum_budget_usage: float = 0.5,
        hidden_dimension: Optional[int] = None,
        kernel_density_function: RadialBasisFunction = RadialBasisFunction.CAUCHY,
        dropout: float = 0.1,
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
        self.kernel_density_function = kernel_density_function

        self.hidden_dimension = hidden_dimension or embedding_dimension

        self.window_size = 2 * locality_radius + 1

        self.frontend_convolution = MaskedConvolution1D(
            embedding_dimension=self.embedding_dimension,
            hidden_dimension=self.hidden_dimension,
            kernel_size=3,
        )

        self.feature_head = nn.Sequential(
            nn.RMSNorm(self.embedding_dimension),
            nn.Linear(self.embedding_dimension, self.hidden_dimension),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )

        self.refining_head = nn.Sequential(
            nn.RMSNorm(self.embedding_dimension + 2),
            nn.Linear(self.embedding_dimension + 2, self.hidden_dimension),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )

        self.source_potential_head = nn.Linear(self.hidden_dimension, 1)
        self.coordinate_density_head = nn.Linear(self.hidden_dimension, 1)
        self.budget_head = nn.Linear(self.hidden_dimension, 1)
        self.assignment_sharpness_head = nn.Linear(self.hidden_dimension, 1)

        self.affinity_head = nn.Linear(self.hidden_dimension + 1, 1)

        self.sink_refinement_head = nn.Linear(
            self.hidden_dimension, self.embedding_dimension
        )
        self.vitality_head = nn.Linear(self.hidden_dimension, 1)
        self.reconstruction_sharpness_head = nn.Linear(self.hidden_dimension, 1)

        self.decoder = WindowedSinkCrossAttentionDecoder(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=8,
            bias=False,
        )

    def forward(
        self, sequence: Tensor, sequence_mask: Tensor, payload_mask: Tensor
    ) -> LocalCumulativeKernelDensitySequenceCompressionOutput:
        """
        :param Tensor sequence: [B, L, D] input token features.
        :param Tensor sequence_mask: [B, L] binary mask indicating valid token positions (1 for valid tokens, 0 for padding).
        :param Tensor payload_mask: [B, L] binary mask indicating positions that can carry real content.
        """
        batch_size, sequence_length, _ = sequence.shape
        device = sequence.device

        # Use neighborhood to better able to identify local patterns and assign content to sinks based on local context
        sequence = sequence + self.frontend_convolution(
            sequence, payload_mask
        )  # [B, L, D]

        payload = sequence * payload_mask.unsqueeze(-1)  # [B, L, D]
        payload_features = self.feature_head(payload)  # [B, L, H]

        dtype = payload_features.dtype
        negative_infinity = torch.finfo(dtype).min
        epsilon = torch.finfo(dtype).eps

        # How much budget do we have for content after considering the baseline positional cost?
        valid_sequence_lengths = sequence_mask.sum(dim=1)  # [B]
        adaptive_capacity = (
            (valid_sequence_lengths * self.resolution)
            - (valid_sequence_lengths * self.positional_weight)
        ).clamp_min(0.0)  # [B]

        # Controls local spatial scaling: higher scores stretch, lower scores compress.
        coordinate_scores = (
            self.coordinate_density_head(payload_features)
            .squeeze(-1)
            .masked_fill(~payload_mask, negative_infinity)
        )  # [B, L]
        # How much of the coordinate budget should be assigned to each token based on its features.
        spatial_priority = F.softmax(coordinate_scores, dim=-1)  # [B, L]

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

        # Combines the baseline positional cost and the adaptive coordinate allocation.
        # Density of sinks at a position is the sum of a fixed positional bias and the adaptive allocation.
        lattice_density = (
            self.positional_weight * sequence_mask.to(adaptive_coordinates.dtype)
        ) + adaptive_coordinates  # [B, L]

        # Center of each token coordinate interval
        source_coordinates = torch.cumsum(lattice_density, dim=1) - (
            lattice_density / 2
        )  # [B, L]

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
        affinity_scores = self.kernel_density_function(
            coordinate_drift,
            F.softplus(self.assignment_sharpness_head(payload_features)) + epsilon,
        ).masked_fill(
            ~(window_mask & payload_mask.unsqueeze(-1)), negative_infinity
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
        source_potential = (
            self.source_potential_head(payload_features).squeeze(-1)
        ).masked_fill(~payload_mask, negative_infinity)  # [B, L]

        # Representing the proportion of the payload budget that should be assigned to each token.
        source_mass = F.softmax(source_potential, dim=-1)  # [B, L]

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

        # Refine the latent tokens using the learned residuals.
        tokens = tokens + self.sink_refinement_head(sink_features)  # [B, T, D]
        tokens = tokens * sink_mask.unsqueeze(-1)  # [B, T, D]

        sink_vitality_scores = F.logsigmoid(
            self.vitality_head(sink_features).squeeze(-1)
        )  # [B, T]

        attention_bias = sink_vitality_scores.masked_fill(
            ~sink_mask, negative_infinity
        )  # [B, T]

        window_occupancy_bias = attention_bias[
            batch_indices, window_sink_indices
        ]  # [B, L, W]

        sink_reconstruction_sharpness = (
            F.softplus(self.reconstruction_sharpness_head(sink_features)) + epsilon
        )  # [B, T, 1]

        window_sharpness = sink_reconstruction_sharpness[
            batch_indices, window_sink_indices
        ].squeeze(-1)  # [B, L, W]

        reconstruction_affinity = self.kernel_density_function(
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

        sink_positions = torch.zeros(
            batch_size,
            lattice_size,
            device=device,
            dtype=weighted_source_contributions.dtype,
        ).scatter_add_(
            dim=1,
            index=flattened_window_indices,
            src=weighted_source_contributions,
        ) / sink_mass.clamp_min(epsilon)  # [B, T]

        return LocalCumulativeKernelDensitySequenceCompressionOutput(
            tokens=tokens,  # [B, T, D]
            sink_mask=sink_mask,  # [B, T]
            sink_mass=sink_mass,  # [B, T]
            window_sink_indices=window_sink_indices,  # [B, L, W]
            window_mask=window_mask,  # [B, L, W]
            reconstruction_bias=reconstruction_bias,  # [B, L, W]
            spilled_mass=spilled_mass,  # [B]
            budget_usage_fraction=budget_usage_fraction,  # [B]
            total_coordinate_span=total_coordinate_span.squeeze(-1),  # [B]
            sink_allocation=sink_allocation,  # [B, L, W]
            source_coordinates=source_coordinates,  # [B, L]
            window_sink_coordinates=window_sink_coordinates,  # [B, L, W]
            sink_positions=sink_positions,  # [B, T]
        )

    def reconstruct(
        self,
        transformed_tokens: Tensor,
        sequence: Tensor,
        window_sink_indices: Tensor,
        window_mask: Tensor,
        reconstruction_bias: Tensor,
    ) -> Tensor:
        if self.training:
            dropout_mask = (
                torch.rand(reconstruction_bias.shape, device=reconstruction_bias.device)
                < self.dropout
            ) & window_mask
            reconstruction_bias = reconstruction_bias.masked_fill(
                dropout_mask, torch.finfo(reconstruction_bias.dtype).min
            )
        return self.decoder(
            sequence_queries=sequence,
            latent_tokens=transformed_tokens,
            window_sink_indices=window_sink_indices,
            window_mask=window_mask,
            reconstruction_bias=reconstruction_bias,
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
