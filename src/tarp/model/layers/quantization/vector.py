from typing import NamedTuple, final, override

import torch
from torch import Tensor, nn


class VectorQuantizationOutput(NamedTuple):
    quantized: Tensor
    encoding_indices: Tensor
    auxiliary_loss: Tensor


@final
class VectorQuantization(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dimension: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.quantization_codes_size = codebook_size - 1  # Reserve one code for padding
        self.embedding_dimension = embedding_dimension
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dimension,
            padding_idx=0,  # Reserve index 0 for padding
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        _ = nn.init.uniform_(
            self.codebook.weight[1:],
            -1.0 / self.quantization_codes_size,
            1.0 / self.quantization_codes_size,
        )
        _ = self.codebook.weight[0].zero_()

    @override
    def forward(self, features: Tensor, mask: Tensor) -> VectorQuantizationOutput:
        # Apply the mask to the features to ignore padding in distance calculations
        mask_expanded = mask.to(dtype=features.dtype).unsqueeze(-1)  # [B, L, 1]
        features = features * mask_expanded

        # Get the real codebook (excluding the reserved padding code)
        codebook_weight = self.codebook.weight[1:]  # [K, D]

        # Euclidean distance between features and codebook entries
        distances = (
            features.square().sum(dim=-1, keepdim=True)
            - 2 * features @ codebook_weight.t()
            + codebook_weight.square().sum(dim=-1)
        )  # [B, L, K]

        # Find the nearest codebook entry for each feature vector, while ignoring masked positions
        encoding_indices = (distances.argmin(dim=-1) + 1).masked_fill(
            ~mask.bool(), 0
        )  # [B, L], reserve 0 for padding

        # Quantize the features using the codebook
        quantized = self.codebook(encoding_indices)  # [B, L, D]

        # Calculate Losses (only over valid, unmasked positions)
        codebook_loss = (
            (quantized - features.detach()).square() * mask_expanded
        ).sum() / mask_expanded.sum().clamp_min(1.0)
        commitment_loss = (
            (features - quantized.detach()).square() * mask_expanded
        ).sum() / mask_expanded.sum().clamp_min(1.0)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        # Add the quantization error back to the features for gradient flow
        quantized = features + (quantized - features).detach()
        quantized = quantized * mask_expanded  # 0-pad

        return VectorQuantizationOutput(
            quantized=quantized,
            encoding_indices=encoding_indices,
            auxiliary_loss=loss,
        )
