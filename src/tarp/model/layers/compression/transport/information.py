import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RadialBasisFunctions(Enum):
    """
    Common radial basis functions for computing assignment scores based on distances between tokens and slot centers.
    The choice of function and its sharpness parameter can affect how mass is assigned to slots based on distance,
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
        :param Tensor distances: Tensor of shape [B, L, W] representing the distances from each token to the centers of the slots in its local window.
        :param Union[float, Tensor] sharpness: Positive decay factor, either a scalar or a tensor of shape [B, L, 1].
        :return Tensor: Tensor of shape [B, L, W] representing the log assignment scores for each token to each slot in its local window.
        """
        match self:
            case RadialBasisFunctions.CAUCHY:
                return -torch.log1p(sharpness * distances.square())
            case RadialBasisFunctions.GAUSSIAN:
                return -sharpness * distances.square()
            case RadialBasisFunctions.LAPLACE:
                return -sharpness * distances.abs()
            case RadialBasisFunctions.EPANECHNIKOV:
                return torch.clamp(1 - sharpness * distances.square(), min=1e-6).log()
            case RadialBasisFunctions.RATIONAL_POWER:
                return -sharpness * torch.log1p(distances.abs())
            case RadialBasisFunctions.INVERSE_MULTIQUADRIC:
                return -0.5 * torch.log1p(sharpness * distances.square())
            case RadialBasisFunctions.TRIWEIGHT:
                return (
                    3
                    * (torch.clamp(1 - sharpness * distances.square(), min=1e-6)).log()
                )


@dataclass
class LocalCumulativeKernelDensitySequenceCompressionOutput:
    tokens: Tensor
    attention_bias: Tensor
    slot_mask: Tensor
    slot_mass: Tensor
    background_mass: Tensor
    assignment_plan: Tensor
    window_slot_indices: Tensor
    payload_density: Tensor
    coordinate_rate: Tensor
    total_coordinate_mass: Tensor
    slot_centers: Tensor
    budget_usage_fraction: Tensor
    adaptive_payload_budget: Tensor
    cumulative_position: Tensor
    window_slot_mask: Tensor
    reconstruction_scores: Tensor

    def background_loss(
        self,
        minimum_background_mass: float = 0.02,
        maximum_background_mass: float = 0.10,
    ) -> Tensor:
        """
        Encourage a small but nonzero amount of payload mass to route to background.

        This prevents:
        1. routing almost everything into slots, even when some information should be dropped
        2. routing too much payload into background and underusing the slot bottleneck

        :param float minimum_background_mass: Lower bound for acceptable background mass.
        :param float maximum_background_mass: Upper bound for acceptable background mass.
        :return Tensor: [B] penalty for background usage outside the target range.
        """
        lower = F.relu(minimum_background_mass - self.background_mass)
        upper = F.relu(self.background_mass - maximum_background_mass)
        return lower + upper  # L1 penalty [B]

    def density_budget_loss(
        self,
        sequence_mask: Tensor,
        minimum_resolution: float = 0.40,
        maximum_resolution: float = 0.55,
    ) -> Tensor:
        """
        Encourage the compressor to use a target range of coordinate mass per real sequence token.

        Resolution here means:
            total_coordinate_mass / number_of_valid_tokens

        This is the main loss that controls how much budget the compressor actually spends.
        Use bounds that match the intended operating regime of the model.

        :param Tensor sequence_mask: [B, L] boolean mask of valid sequence positions.
        :param float minimum_resolution: Minimum desired coordinate mass per valid token.
        :param float maximum_resolution: Maximum desired coordinate mass per valid token.
        :return Tensor: [B] penalty for using too little or too much coordinate budget.
        """
        valid_length = (
            sequence_mask.bool().sum(dim=-1).to(self.total_coordinate_mass.dtype)
        )

        actual_resolution = self.total_coordinate_mass / valid_length.clamp_min(1e-6)

        lower = F.relu(minimum_resolution - actual_resolution)
        upper = F.relu(actual_resolution - maximum_resolution)

        return lower.square() + upper.square()  # L2 penalty [B]

    def entropy_loss(self) -> Tensor:
        """
        Shannon entropy of the transport plan to encourage more confident (lower entropy) assignments.

        This penalizes diffuse transport from tokens to multiple nearby slots and
        encourages more confident slot assignments. Only positions that actually
        assign nontrivial mass contribute to the loss.

        :return Tensor: [B] average assignment entropy over active payload positions.
        """
        window_mass = self.assignment_plan.sum(dim=-1)  # [B, L]
        window_mask = window_mass > 1e-8  # [B, L]

        distribution = self.assignment_plan / window_mass.unsqueeze(-1).clamp_min(1e-8)
        distribution = distribution.clamp_min(1e-8)

        entropy = -(distribution * distribution.log()).sum(dim=-1)  # [B, L]
        entropy = entropy * window_mask.to(entropy.dtype)

        return entropy.sum(dim=-1) / window_mask.sum(dim=-1).clamp_min(1)

    def spatial_dispersion_loss(self) -> Tensor:
        """
        Encourage local, geometrically compact assignments within each token's slot window.

        Penalizes the expected squared distance between each token's coordinate and
        the slot centers it writes to, conditioned on the mass assigned to slots.

        :return Tensor: [B] average spatial dispersion of token-to-slot assignments.
        """
        batch_size, _, _ = self.window_slot_indices.shape

        batch_indices = torch.arange(
            batch_size, device=self.slot_centers.device
        ).reshape(batch_size, 1, 1)  # [B, 1, 1]
        window_slot_centers = self.slot_centers[
            batch_indices, self.window_slot_indices
        ]  # [B, L, W]

        squared_positional_distances = (
            self.cumulative_position.unsqueeze(-1) - window_slot_centers
        ).square()  # [B, L, W]

        window_mass = self.assignment_plan.sum(dim=-1)  # [B, L]
        window_mask = window_mass > 1e-8  # [B, L]

        expected_dispersion = (self.assignment_plan * squared_positional_distances).sum(
            dim=-1
        ) / window_mass.clamp_min(1e-8)  # [B, L]

        expected_dispersion = expected_dispersion * window_mask.to(
            expected_dispersion.dtype
        )

        return expected_dispersion.sum(dim=-1) / window_mask.sum(dim=-1).clamp_min(1)

    def auxiliary_losses(self, sequence_mask: Tensor) -> Tensor:
        """
        Combine multiple auxillary losses into a single scalar.

        :param Tensor sequence_mask: [B, L] boolean mask of valid sequence positions.
        :return Tensor: scalar auxillary loss.
        """
        background_penalty = self.background_loss()
        density_budget_penalty = self.density_budget_loss(sequence_mask)
        entropy_penalty = self.entropy_loss()
        spatial_dispersion_penalty = self.spatial_dispersion_loss()

        return (
            0.5 * background_penalty
            + 1.0 * density_budget_penalty
            + 0.0 * entropy_penalty
            + 0.0 * spatial_dispersion_penalty
        ).mean()  # Average over batch


class WindowedSlotCrossAttentionDecoder(nn.Module):
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
        latent_slot_tokens: Tensor,  # [B, T, D]
        window_slot_indices: Tensor,  # [B, L, W]
        window_slot_mask: Tensor,  # [B, L, W]
        reconstruction_bias: Optional[Tensor] = None,  # [B, L, W]
    ) -> Tensor:
        batch_size, sequence_length, embedding_dimension = sequence_queries.shape
        _, maximum_slot_count, _ = latent_slot_tokens.shape
        _, _, local_window = window_slot_indices.shape

        batch_offsets = (
            torch.arange(
                batch_size,
                device=sequence_queries.device,
                dtype=window_slot_indices.dtype,
            ).reshape(batch_size, 1, 1)
            * maximum_slot_count
        )

        flattened_indices = (window_slot_indices + batch_offsets).flatten()

        flattened_tokens = latent_slot_tokens.reshape(
            batch_size * maximum_slot_count,
            embedding_dimension,
        )

        slot_features = flattened_tokens[flattened_indices].reshape(
            batch_size, sequence_length, local_window, embedding_dimension
        )  # [B, L, W, D]

        queries = self.query_projection(sequence_queries).reshape(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        )  # [B, L, H, Dh]

        keys, values = (
            self.key_value_projection(slot_features)
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

        if reconstruction_bias is not None:
            scores = scores + reconstruction_bias.unsqueeze(2)  # [B, L, 1, W]

        scores = scores.masked_fill(
            ~window_slot_mask.unsqueeze(2), -1e4
        )  # [B, L, H, W]

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = attention_weights * window_slot_mask.unsqueeze(2).to(
            attention_weights.dtype
        )
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        context = torch.einsum(
            "blhw,blwhd->blhd", attention_weights, values
        )  # [B, L, H, Dh]

        context = context.reshape(
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


class LocalCumulativeKernelDensitySequenceCompression(nn.Module):
    """ """

    def __init__(
        self,
        embedding_dimension: int,
        resolution: float = 0.5,  # Number of slots per input token. For example, 0.5 means on average 1 slot for every 2 tokens.
        local_slot_radius: int = 6,
        positional_weight: float = 0.4,
        assignment_sharpness: float = 4.0,  # Inverse temperature
        background_cost_payload: float = 2.0,
        minimum_budget_usage: float = 0.5,
        reconstruction_sharpness: Optional[float] = None,
        hidden_dimension: Optional[int] = None,
        kernel_density_function: RadialBasisFunctions = RadialBasisFunctions.CAUCHY,
    ):
        super().__init__()
        assert 0.0 <= minimum_budget_usage <= 1.0, (
            "minimum_budget_usage must be between 0 and 1."
        )
        assert 0.0 <= positional_weight <= resolution <= 1.0, (
            "positional_weight must be non-negative and less than resolution, and resolution must be at most 1 to ensure the model has a meaningful budget to assign to slots and that the coordinate mass grows with sequence length in a reasonable way."
        )

        self.embedding_dimension = embedding_dimension
        self.resolution = resolution
        self.positional_weight = positional_weight
        self.background_cost_payload = background_cost_payload
        self.local_slot_radius = local_slot_radius
        self.minimum_budget_usage = minimum_budget_usage

        self.kernel_density_function = kernel_density_function

        self.reconstruction_sharpness = reconstruction_sharpness or assignment_sharpness
        self.hidden_dimension = hidden_dimension or embedding_dimension

        self.local_window = 2 * local_slot_radius + 1

        self.frontend_convolution = MaskedConvolution1D(
            embedding_dimension=self.embedding_dimension,
            hidden_dimension=self.hidden_dimension,
            kernel_size=3,
        )

        self.payload_feature_head = nn.Sequential(
            nn.RMSNorm(self.embedding_dimension),
            nn.Linear(self.embedding_dimension, self.hidden_dimension),
            nn.SiLU(),
        )
        self.density_head = nn.Linear(self.hidden_dimension, 1)
        self.assignment_sharpness_head = nn.Linear(self.hidden_dimension, 1)

        self.budget_head = nn.Sequential(
            nn.RMSNorm(self.embedding_dimension),
            nn.Linear(self.embedding_dimension, 1),
        )

        self.token_refining_feature_head = nn.Sequential(
            nn.RMSNorm(self.embedding_dimension + 2),
            nn.Linear(self.embedding_dimension + 2, self.hidden_dimension),
            nn.SiLU(),
        )

        self.slot_refiner = nn.Linear(self.hidden_dimension, self.embedding_dimension)
        self.existence_gate = nn.Linear(self.hidden_dimension, 1)

        self.decoder = WindowedSlotCrossAttentionDecoder(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=8,
        )

    def forward(
        self, sequence: Tensor, sequence_mask: Tensor, payload_mask: Tensor
    ) -> LocalCumulativeKernelDensitySequenceCompressionOutput:
        batch_size, sequence_length, _ = sequence.shape
        device = sequence.device
        dtype = sequence.dtype

        negative_infinity = -1e4

        sequence_mask, payload_mask = sequence_mask.bool(), payload_mask.bool()

        sequence = (
            sequence + self.frontend_convolution(sequence, payload_mask)
        ) * sequence_mask.unsqueeze(-1).to(dtype)  # [B, L, D]

        # Budgeting: Determine how many slots sequence should fill
        valid_sequence_length = sequence_mask.sum(dim=-1)  # [B]
        maximum_payload_budget = (
            (valid_sequence_length * self.resolution)
            - (valid_sequence_length * self.positional_weight)
        ).clamp_min(0.0)  # [B]

        # Mask input sequence with payload mask to get only the payload tokens
        payload = sequence * payload_mask.unsqueeze(-1).to(dtype)  # [B, L, D]
        payload_features = self.payload_feature_head(payload)  # [B, L, H]

        # Compute density scores for each token in the payload
        density_scores = (
            self.density_head(payload_features)
            .squeeze(-1)
            .masked_fill(~payload_mask, negative_infinity)
        )  # [B, L]
        # Convert density scores to a distribution over the payload
        density_distribution = F.softmax(density_scores, dim=-1) * payload_mask.to(
            dtype
        )  # [B, L]
        density_distribution = density_distribution / density_distribution.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)  # [B, L]

        pooled_sequence = sequence.sum(dim=1) / sequence_mask.sum(
            dim=1, keepdim=True
        ).to(dtype).clamp_min(1.0)

        budget_usage_fraction = self.minimum_budget_usage + (
            1 - self.minimum_budget_usage
        ) * torch.sigmoid(self.budget_head(pooled_sequence)).squeeze(-1)  # [B]

        adaptive_payload_budget = maximum_payload_budget * budget_usage_fraction  # [B]
        payload_density = density_distribution * adaptive_payload_budget.unsqueeze(
            -1
        )  # [B, L]

        # # Coordinate rate includes a small floor for every real sequence position.
        coordinate_rate = (
            self.positional_weight * sequence_mask.to(dtype)
        ) + payload_density  # [B, L]
        #
        # coordinate_rate = self.positional_weight * sequence_mask.to(dtype)

        # Center coordinate of each token's cumulative interval.
        # cumulative_position is in the range [0, L], where L is the sequence length.
        cumulative_position = torch.cumsum(coordinate_rate, dim=1) - (
            coordinate_rate / 2
        )  # [B, L]

        total_coordinate_mass = coordinate_rate.sum(dim=1, keepdim=True)  # [B, 1]

        #  Build batch-shaped slot centers and active slot mask.
        maximum_slot_count = max(
            1, int(sequence_length * self.resolution + 0.99999)
        )  # Round up to ensure at least one slot

        slot_centers = (
            torch.arange(maximum_slot_count, device=device, dtype=dtype).unsqueeze(0)
            + 0.5
        ).expand(batch_size, -1)  # [B, T]

        # Slot is active if its center falls inside the sequence's cumulative mass.
        slot_mask = slot_centers < total_coordinate_mass  # [B, T]

        # Ensure at least one slot is active if there's any token in the sequence
        slot_mask[:, :1] = slot_mask[:, :1] | (sequence_mask.any(dim=1, keepdim=True))

        # Build local windows of slots
        offsets = (
            torch.arange(
                -self.local_slot_radius,
                self.local_slot_radius + 1,
                device=device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1, 1, W]

        # Window selection is discrete; gradients still flow through cumulative_position
        # in the distance-based assignment scores inside the selected window.
        nearest_slot_indices = torch.round(cumulative_position - 0.5).long()  # [B, L]

        # Add local offsets to nearest slot indices to get the indices of slots in each token's local window.
        window_slot_indices = nearest_slot_indices.unsqueeze(-1) + offsets  # [B, L, W]

        # those that fall within the range of available slots are valid, others will be masked out
        valid_window_mask = (window_slot_indices >= 0) & (
            window_slot_indices < maximum_slot_count
        )  # [B, L, W]

        # Clamp window slot indices to be within the valid range of slot indices for gathering slot centers and masks.
        window_slot_indices = window_slot_indices.clamp(
            0, maximum_slot_count - 1
        )  # [B, L, W]

        # Gather the slot masks for the slots in each token's local window. This will be used to mask out invalid slots in the local window.
        batch_indices = torch.arange(batch_size, device=device).reshape(
            batch_size, 1, 1
        )
        window_slot_mask = slot_mask[batch_indices, window_slot_indices]  # [B, L, W]

        # Mask out invalid slots in the local window
        window_slot_mask = window_slot_mask & valid_window_mask  # [B, L, W]

        # Gather the centers of the slots in each token's local window. This will be used to compute distances from the token to the slots in its local window.
        window_slot_centers = slot_centers[batch_indices, window_slot_indices]

        # Distance from each payload token to the centers of its local window of slots.
        positional_distances = (
            cumulative_position.unsqueeze(-1) - window_slot_centers
        )  # [B, L, W]

        # Compute assignment scores using the kernel density function
        assignment_scores = self.kernel_density_function(
            positional_distances,
            F.softplus(self.assignment_sharpness_head(payload_features)) + 1e-4,
        )  # [B, L, W]

        # Masked assignment scores for invalid slots in the local window
        assignment_scores = assignment_scores.masked_fill(
            ~(window_slot_mask & payload_mask.unsqueeze(-1)), negative_infinity
        )  # [B, L, W]

        # Background is an extra write target. It absorbs payload mass that should not be represented by any slot
        background_scores = torch.full(
            (batch_size, sequence_length, 1),
            -self.background_cost_payload,
            device=device,
            dtype=dtype,
        )  # [B, L, 1]

        # Non-payload tokens are entirely background
        background_scores = background_scores.masked_fill(
            ~payload_mask.unsqueeze(-1), 0.0
        )  # [B, L, 1]

        # Combine slot assignment scores with background scores
        transport_plan = F.softmax(
            torch.cat([assignment_scores, background_scores], dim=-1), dim=-1
        )  # [B, L, W+1]

        assignment_plan = transport_plan[..., :-1]  # [B, L, W]
        background_plan = transport_plan[..., -1]  # [B, L]

        # Source mass is determined by payload density, which is the "mass" that needs
        # to be transported from tokens to slots. Non-payload tokens have zero mass and thus do not contribute to any slot.
        # Of the payload being represented, what fraction comes from each token?
        source_mass = payload_density / payload_density.sum(dim=1, keepdim=True).clamp(
            min=1e-8
        )  # [B, L]

        # # Mask source mass to ensure non-payload tokens have zero mass
        source_mass = source_mass * payload_mask.to(dtype)  # [B, L]
        # source_mass = density_distribution * payload_mask.to(dtype)
        # source_mass = source_mass / source_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # Compute the mass for slots
        assignment_mass = assignment_plan * source_mass.unsqueeze(-1)  # [B, L, W]
        background_mass = (background_plan * source_mass).sum(dim=-1)  # [B]

        flattened_window_slot_indices = window_slot_indices.reshape(
            batch_size, sequence_length * self.local_window
        )  # [B, L*W]

        flattened_assignment_mass = assignment_mass.reshape(
            batch_size, sequence_length * self.local_window
        )  # [B, L*W]

        # How much of the normalized payload transport landed in each slot?
        slot_mass = torch.zeros(
            batch_size, maximum_slot_count, device=device, dtype=dtype
        ).scatter_add_(
            1, flattened_window_slot_indices, flattened_assignment_mass
        )  # [B, T]

        slot_contributions = torch.einsum(
            "blw, bld -> blwd", assignment_mass, sequence
        ).reshape(
            batch_size,
            sequence_length * self.local_window,
            self.embedding_dimension,
        )  # [B, L*W, D]

        slot_payload = torch.zeros(
            batch_size,
            maximum_slot_count,
            self.embedding_dimension,
            device=device,
            dtype=dtype,
        ).scatter_add_(
            1,
            flattened_window_slot_indices.unsqueeze(-1).expand(
                batch_size,
                sequence_length * self.local_window,
                self.embedding_dimension,
            ),
            slot_contributions.to(dtype),
        )  # [B, T, D]

        # Normalize slot payload by slot mass to get the final slot representations. Add a small epsilon to the denominator for numerical stability.
        tokens = slot_payload / slot_mass.unsqueeze(-1).clamp_min(1e-6)  # [B, T, D]

        # Refine slot representations with residuals\
        slot_refiner_features = torch.cat(
            [
                tokens,
                slot_mass.unsqueeze(-1),
                torch.log(slot_mass.unsqueeze(-1).clamp_min(1e-6)),
            ],
            dim=-1,
        ) * slot_mask.unsqueeze(-1)  # [B, T, D+2]

        token_refined_features = self.token_refining_feature_head(
            slot_refiner_features
        ) * slot_mask.unsqueeze(-1)  # [B, T, H]

        tokens = tokens + self.slot_refiner(token_refined_features)  # [B, T, D]
        # Mask out slots that are not active
        tokens = tokens * slot_mask.unsqueeze(-1).to(dtype)  # [B, T, D]

        # Soft existence gate to allow the model to learn to ignore certain slots if it wants to
        existence_scores = self.existence_gate(token_refined_features).squeeze(
            -1
        )  # [B, T]
        log_existence_scores = F.logsigmoid(existence_scores)  # [B, T]

        attention_bias = log_existence_scores.masked_fill(
            ~slot_mask, negative_infinity
        )  # [B, T]

        # Add local occupancy bias so dead slots get less reconstruction mass.
        window_log_occupancy = attention_bias[batch_indices, window_slot_indices]

        # Reconstruction
        reconstruction_scores = self.kernel_density_function(
            positional_distances, self.reconstruction_sharpness
        )  # [B, L, W]

        reconstruction_scores = (
            reconstruction_scores + window_log_occupancy
        )  # [B, L, W]

        # All sequence including non-payload tokens can attend to slots for reconstruction
        reconstruction_scores = reconstruction_scores.masked_fill(
            ~(window_slot_mask & sequence_mask.unsqueeze(-1)), negative_infinity
        )

        return LocalCumulativeKernelDensitySequenceCompressionOutput(
            tokens=tokens,  # [B, T, D]
            attention_bias=attention_bias,  # [B, T]
            slot_mask=slot_mask,  # [B, T]
            slot_mass=slot_mass,  # [B, T]
            background_mass=background_mass,  # [B]
            assignment_plan=assignment_plan,  # [B, L, W]
            window_slot_indices=window_slot_indices,  # [B, L, W]
            payload_density=payload_density,  # [B, L]
            coordinate_rate=coordinate_rate,  # [B, L]
            total_coordinate_mass=total_coordinate_mass.squeeze(-1),  # [B]
            slot_centers=slot_centers,  # [B, T]
            budget_usage_fraction=budget_usage_fraction,  # [B]
            adaptive_payload_budget=adaptive_payload_budget,  # [B]
            cumulative_position=cumulative_position,  # [B, L]
            window_slot_mask=window_slot_mask,  # [B, L, W]
            reconstruction_scores=reconstruction_scores,  # [B, L, W]
        )

    def reconstruct(
        self,
        tokens: Tensor,
        sequence: Tensor,
        window_slot_indices: Tensor,
        window_slot_mask: Tensor,
        positional_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Reconstruct the original sequence from the compressed slot representations using the reconstruction plan, with dynamic realignment.

        :param Tensor tokens: [B, T, D] The compressed slot representations.
        :param Tensor reconstruction_plan: [B, L, W] The plan for how to reconstruct each token from the slots in its local window.
        :param Tensor window_slot_indices: [B, L, W] The indices of the slots in each token's local window.
        :param Tensor sequence: [B, L, D] The original input sequence, used for dynamic realignment.
        :param Tensor sequence_mask: [B, L] Mask indicating valid positions in the sequence.
        :return Tensor: [B, L, D] The reconstructed sequence.
        """
        return self.decoder(
            sequence_queries=sequence,
            latent_slot_tokens=tokens,
            window_slot_indices=window_slot_indices,
            window_slot_mask=window_slot_mask,
            reconstruction_bias=positional_bias,
        )

    def pooling(self, tokens: Tensor, slot_mass: Tensor, slot_mask: Tensor) -> Tensor:
        """
        Pool the compressed slot representations into a single vector representation for the whole sequence.

        :param Tensor tokens: [B, T, D] The compressed slot representations.
        :param Tensor slot_mass: [B, T] The mass assigned to each slot, which can be used as weights for pooling.
        :param Tensor slot_mask: [B, T] A boolean mask indicating which slots are active and should be included in the pooling.
        :return Tensor: [B, D] The pooled representation of the sequence.
        """
        masked_tokens = tokens * slot_mask.unsqueeze(-1).to(tokens.dtype)  # [B, T, D]
        masked_mass = slot_mass * slot_mask.to(slot_mass.dtype)  # [B, T]

        pooled = (masked_tokens * masked_mass.unsqueeze(-1)).sum(
            dim=1
        ) / masked_mass.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)  # [B, D]

        return pooled
