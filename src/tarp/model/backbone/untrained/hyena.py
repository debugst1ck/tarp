import math
from typing import Optional

import torch
import torch.autograd
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.functional.fft.convolution import fft_cross_correlation_1d
from tarp.model.backbone import Encoder
from tarp.model.layers.pooling.learned import SelfAttentionPooling


class LinearAndSinusoidalPositionalEncoding1D(nn.Module):
    def __init__(self, embedding_dimension: int) -> None:
        """
        :param int embedding_dimension: Dimension of the positional embeddings (must be odd).
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_bands = (self.embedding_dimension - 1) // 2
        self.register_buffer("cached_normalized_positions", None, persistent=False)
        self.register_buffer("cached_positional_encodings", None, persistent=False)
        self.cached_normalized_positions: Optional[Tensor]  # Declare types # (1, K, 1)
        self.cached_positional_encodings: Optional[Tensor]  # Declare types # (1, K, D)

    def forward(
        self, kernel_length: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """
        :param kernel_length: Length of the kernel to retrieve positional encodings for.
        :return: Positional encodings of shape (1, kernel_length, embedding_dimension) and normalized positions of shape (1, kernel_length, 1).
        """
        # Check if we have cached positional encodings for the requested kernel length
        # Usually, the kernel length remains constant during training/inference
        if (
            self.cached_positional_encodings is not None
            and self.cached_normalized_positions is not None
            and self.cached_positional_encodings.shape[1] == kernel_length
        ):
            return (
                self.cached_positional_encodings,
                self.cached_normalized_positions,
            )

        # Normalized time positions from 0 to 1
        normalized_time = torch.arange(0, kernel_length, device=device).unsqueeze(
            0
        ).unsqueeze(-1) / (kernel_length - 1)  # (1, K, 1)

        # Determine the number of frequency bands
        # Each band corresponds to a pair of sine and cosine functions
        linear_positions = (
            torch.arange(0, kernel_length, device=device).unsqueeze(0).unsqueeze(-1)
        )  # (1, K, 1)

        angular_frequencies = (
            2 * math.pi * linear_positions / kernel_length
        )  # (1, K, 1)

        if self.number_of_bands > 1:
            # Compute complex exponentials for each frequency band rather than sine and cosine separately
            frequencies = torch.linspace(
                1e-4, self.number_of_bands - 1, self.number_of_bands, device=device
            )[None, None, :]  # (1, 1, number_of_bands)
        else:
            # Fallback for very small embedding dimensions
            frequencies = torch.tensor([[[1e-4]]], device=device)  # (1, 1, 1)

        complex_exponentials = torch.exp(
            -1j * frequencies * angular_frequencies
        )  # (1, K, number_of_bands)

        # Concatenate normalized positions, real, and imaginary parts of complex exponentials
        # Resulting shape: (1, K, 1 + 2 * number_of_bands) = (1, K, D)
        positional_encodings = torch.cat(
            [
                normalized_time,
                complex_exponentials.real,
                complex_exponentials.imag,
            ],
            dim=-1,
        )

        # Cache the computed positional encodings and normalized positions
        # This optimization assumes that the kernel length remains constant
        self.cached_positional_encodings = positional_encodings
        self.cached_normalized_positions = normalized_time

        return positional_encodings, normalized_time  # (1, K, D), (1, K, 1)


class FixedExponentialPositionalGating1D(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        fast_decay_fraction: float = 0.3,
        slow_decay_fraction: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.0,
    ) -> None:
        """"""
        super().__init__()
        self.shift = shift
        self.maximum_decay = math.log(target) / fast_decay_fraction
        self.minimum_decay = math.log(target) / slow_decay_fraction
        self.model_dimension = model_dimension

    def reset_parameters(self) -> None:
        """No parameters to reset."""
        pass

    def forward(self, normalized_positions: Tensor, input: Tensor) -> Tensor:
        """
        :param normalized_positions: Normalized positions of shape (1, L, 1) or (B, L, 1).
        :param input: Input tensor of shape (..., L, D).
        :return: Modulated input tensor of shape (..., L, D).
        """
        deltas = torch.linspace(
            self.maximum_decay,
            self.minimum_decay,
            self.model_dimension,
            device=input.device,
        )[None, None, :]  # (1, 1, D)
        decay_factors = torch.exp(-normalized_positions * deltas.abs())  # (1, L, D)
        modulated_input = input * (decay_factors + self.shift)  # (..., L, D)
        return modulated_input  # (..., L, D)


class LearnableFrequencySineActivation(nn.Module):
    """
    Sine activation layer with learnable frequency modulation.
    """

    def __init__(
        self,
        dimension: int,
        angular_frequency: float = 1.0,
        learnable: bool = True,
    ) -> None:
        """
        :param dimension: Dimension of the input and output.
        :param scaling: Initial scaling factor for the frequencies.
        :param learnable: Whether the frequencies are learnable parameters.
        """
        super().__init__()
        self.dimension = dimension
        self.angular_frequency = angular_frequency

        self.frequencies = nn.Parameter(
            torch.empty((1, dimension)), requires_grad=learnable
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(
            self.frequencies, a=1e-3, b=self.angular_frequency
        )  # Initialize frequencies uniformly

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: Input tensor of shape (..., D).
        :return: Output tensor of shape (..., D).
        """
        return torch.sin(input * self.frequencies)


class HyenaFilter1D(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        frequency_scaling: float = 10.0,
        positional_embedding_dimension: int = 3,
        mlp_hidden_dimension: int = 16,
        mlp_number_of_layers: int = 2,
        use_exponential_decay: bool = True,
    ) -> None:
        """
        :param model_dimension: Dimension of the model (input/output feature dimension).
        :param positional_embedding_dimension: Dimension of the positional embeddings (must be at least 3).
        :param mlp_hidden_dimension: Hidden dimension of the MLP used to generate filter weights. Also called "order" or "width".
        :param mlp_number_of_layers: Number of layers in the MLP used to generate filter weights.
        :param use_exponential_decay: Whether to apply exponential decay modulation to the generated filters.
        """
        super().__init__()

        assert positional_embedding_dimension >= 3, (
            "Positional embedding dimension must be at least 3."
        )

        self.model_dimension = model_dimension
        self.positional_embedding_dimension = positional_embedding_dimension
        self.use_exponential_decay = use_exponential_decay

        self.positional_encoding = LinearAndSinusoidalPositionalEncoding1D(
            embedding_dimension=positional_embedding_dimension,
        )

        mlp_layers: list[nn.Module] = [
            # First Hidden Layer Block
            nn.Linear(positional_embedding_dimension, mlp_hidden_dimension),
            # nn.GELU(),
            LearnableFrequencySineActivation(mlp_hidden_dimension, frequency_scaling),
            # Subsequent Hidden Layer Blocks (N-1 times)
            *[
                layer
                for _ in range(mlp_number_of_layers - 1)
                # Loop N-1 times for the rest of the layers
                for layer in [
                    nn.Linear(mlp_hidden_dimension, mlp_hidden_dimension),
                    # nn.GELU(),
                    LearnableFrequencySineActivation(
                        mlp_hidden_dimension, frequency_scaling
                    ),
                ]
            ],
            # Final Output Layer
            nn.Linear(mlp_hidden_dimension, model_dimension),
        ]
        self.mlp = nn.Sequential(*mlp_layers)

        if use_exponential_decay:
            self.modulation = FixedExponentialPositionalGating1D(model_dimension)

        # Optional bias for the filter
        self.filter_bias = nn.Parameter(torch.zeros(model_dimension))

    def reset_parameters(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
            if isinstance(layer, LearnableFrequencySineActivation):
                layer.reset_parameters()
        nn.init.zeros_(self.filter_bias)

    def generate_filter(self, length: int) -> Tensor:
        """
        Generate filter weights of shape (d_model, L_k).
        :param length: Length of the filter to generate. If None, uses the maximum kernel length.
        """
        positional_encodings, normalized_positions = self.positional_encoding(
            kernel_length=length, device=self.filter_bias.device
        )  # (1, L, D_pe), (1, L, 1)

        # Pass positional encodings through MLP to get filter weights
        filter_weights = self.mlp(positional_encodings)  # (1, L, D_model)

        if self.use_exponential_decay:
            filter_weights = self.modulation(
                normalized_positions, filter_weights
            )  # (1, L, D_model)

        # Transpose to shape (D_model, L)
        filter_weights = filter_weights.squeeze(0).transpose(0, 1)  # (D_model, L)
        return filter_weights

    def forward(
        self,
        input: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        :param input: Input tensor of shape (B, L_in, d_model).
        :param attention_mask: Optional attention mask of shape (B, L_in), True for valid positions.
        :param is_causal: Whether to apply causal padding.
        :return: Output tensor of shape (B, L_out, d_model).
        """
        batch_size, input_length, _ = input.shape

        kernel = self.generate_filter(length=input_length)  # (d_model, L_k)

        # We need to add a dimension to the kernel to match the expected shape
        # Since FFT cross-correlation is identical to F.conv1d
        # and F.conv1d expects in shape `(B, C_i, L)` and filter shape `(C_o, C_i // G, K)`
        # Transpose input to (B, d_model, L_in)
        input_transposed = input.transpose(1, 2)  # (B, d_model, L_in)
        kernel = kernel.unsqueeze(1)  # (d_model, 1, L_k)

        # Apply attention mask if provided
        if attention_mask is not None:
            input_transposed = input_transposed * attention_mask.unsqueeze(1).to(
                input_transposed.dtype
            )

        # We treat each feature dimension independently (groups=d_model)
        # Make 'same' padding
        pad_total = kernel.shape[-1] - 1
        if is_causal:
            pad_left = pad_total  # ← all padding on the *left*
            pad_right = 0
        else:
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
        input_transposed = F.pad(input_transposed, (pad_left, pad_right))
        output = fft_cross_correlation_1d(
            input_transposed,
            kernel,
            stride=1,
            dilation=1,
            groups=self.model_dimension,
            bias=self.filter_bias,
        )  # (B, d_model, L_out)

        # Add per channel bias
        output = output.transpose(1, 2)  # (B, L_out, d_model)
        # Apply attention mask to output if provided

        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(
                output.dtype
            )  # (B, L_out, d_model)

        return output  # (B, L_out, d_model) # where L_out = L_in due to 'same' padding


class HyenaOperator1D(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        recurrence_depth: int = 2,
        mlp_hidden_dimension: int = 2,
        number_of_heads: int = 1,
        filter_mlp_hidden_dimension: int = 64,
        expansion_factor: int = 1,
        dropout: float = 0.0,
        short_filter_length=3,
        activation=nn.GELU(),
    ) -> None:
        super().__init__()
        assert model_dimension % number_of_heads == 0, (
            "Model dimension must be divisible by number of heads."
        )
        assert recurrence_depth >= 2, "Hyena recurrence depth must be >= 2."

        self.model_dimension = model_dimension
        self.recurrence_depth = recurrence_depth
        self.mlp_hidden_dimension = mlp_hidden_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads
        self.expansion_factor = expansion_factor
        self.short_filter_length = short_filter_length
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        total_width = model_dimension * expansion_factor * (recurrence_depth + 1)

        # Projections to generate inputs for each Hyena layer
        self.input_projection = nn.Linear(
            model_dimension,
            (recurrence_depth + 1) * model_dimension * expansion_factor,
        )
        self.output_projection = nn.Linear(
            model_dimension * expansion_factor, model_dimension
        )

        # Short filter for local context
        # This is a learnable convolutional filter
        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=short_filter_length,
            padding=short_filter_length - 1,
            groups=total_width,
        )

        # Hyena filter for long-range context
        self.hyena_filter = HyenaFilter1D(
            model_dimension=self.head_dimension * self.expansion_factor,  # 16
            positional_embedding_dimension=7,
            mlp_hidden_dimension=filter_mlp_hidden_dimension,
            mlp_number_of_layers=2,
            use_exponential_decay=True,
            frequency_scaling=10.0,
        )

    def forward(
        self, input: Tensor, attention_mask: Optional[Tensor] = None, is_causal=False
    ) -> Tensor:
        """
        :param Tensor input: Input tensor of shape (B, L, D).
        :param Optional[Tensor] attention_mask: Attention mask of shape (B, L), True for valid positions.
        :param bool is_causal: Whether to apply causal masking. Or disable future information flow.
        :return: Output tensor of shape (B, L, D).
        """

        batch_size, sequence_length, _ = input.shape

        if attention_mask is not None:
            input = input * attention_mask.unsqueeze(-1).to(input.dtype)

        # Project input to get initial states for Hyena layers
        projected_input: Tensor = self.input_projection(
            input
        )  # (B, L, D * F * (R + 1))

        # Rearrange b l d -> b d l
        projected_input = projected_input.transpose(1, 2)  # (B, D * F * (R + 1), L)

        # Apply short convolutional filter for local context
        short_filtered: Tensor = self.short_filter(projected_input)[
            ..., :sequence_length
        ]  # (B, N_h * H_d * F * (R + 1), L)

        # (B, N_h, R+1, L, Hd*F)
        reshaped = short_filtered.reshape(
            batch_size,
            self.recurrence_depth + 1,
            self.number_of_heads,
            sequence_length,
            self.head_dimension * self.expansion_factor,
        ).permute(0, 2, 1, 3, 4)  # (B, N_h, R+1, L, Hd*F)

        # Split on recurrence axis
        *intermediate_units, accumulator = torch.unbind(
            reshaped, dim=2
        )  # Each of shape (B, N_h, L, Hd*F)

        units = torch.stack(intermediate_units[1:], dim=2)  # (B, N_h, R-1, L, Hd*F)

        # Reshape for Hyena filter
        units = units.reshape(
            batch_size * self.number_of_heads * (self.recurrence_depth - 1),
            sequence_length,
            self.head_dimension * self.expansion_factor,
        )  # (B * N_h * (R-1), L, Hd*F)

        # Apply Hyena filter for long-range context
        filtered_units = self.hyena_filter(
            units, is_causal=is_causal
        )  # (B * N_h * (R-1), L, Hd*F)

        # Reshape back to (B, N_h, R-1, L, Hd*F)
        filtered_units = filtered_units.reshape(
            batch_size,
            self.number_of_heads,
            self.recurrence_depth - 1,
            sequence_length,
            self.head_dimension * self.expansion_factor,
        )  # (B, N_h, R-1, L, Hd*F)

        # Combine contributions across recurrence depth (parallel sum)
        accumulator = accumulator + filtered_units.sum(dim=2)

        # Final interaction with X_0
        output = self.activation(accumulator + intermediate_units[0])

        # (B, N_h, L, Hd*F) -> (B, L, D*F)
        # Transpose and reshape back to (B, L, D*F)
        output = output.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.model_dimension * self.expansion_factor
        )  # (B, L, D*F)

        output = self.output_projection(output)  # (B, L, D)

        output = self.dropout(output)

        # Apply attention mask to output if provided
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(output.dtype)  # (B, L, D)

        # Trim output to original sequence length if padded
        output = output[:, :sequence_length, :]  # (B, L, D)

        return output  # (B, L, D)


class HyenaEncoder(Encoder):
    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        filter_mlp_hidden_dimension: int = 64,
        number_of_layers: int = 4,
        number_of_heads: int = 1,
        recurrence_depth: int = 3,
        mixing_width: int = 2,
        local_context_size: int = 3,
        dropout: float = 0.1,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=model_dimension,
            padding_idx=padding_id,
        )
        self.hyena_layers = nn.ModuleList(
            [
                HyenaOperator1D(
                    model_dimension=model_dimension,
                    number_of_heads=number_of_heads,
                    dropout=dropout,
                    filter_mlp_hidden_dimension=filter_mlp_hidden_dimension,
                    recurrence_depth=recurrence_depth,
                    expansion_factor=mixing_width,
                    short_filter_length=local_context_size,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.normalization = nn.LayerNorm(model_dimension)
        self.embedding_dimension = model_dimension
        self.pooling = SelfAttentionPooling(model_dimension)
        self.output_dimension = model_dimension

    @property
    def encoding_size(self) -> int:
        return self.output_dimension

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        """
        Encodes a sequence of embeddings.

        :param Tensor sequence: A tensor of shape `(batch_size, sequence_length)` representing the input sequence of token IDs.
        :param Tensor attention_mask: An optional tensor of shape `(batch_size, sequence_length)` representing the attention mask for the input sequence.
        :param Tensor return_sequence: A boolean indicating whether to return the full sequence of encoded embeddings or a single pooled embedding.
        :return Tensor: A tensor of shape `(batch_size, encoding_size)` if return_sequence is `False`, or `(batch_size, sequence_length, encoding_size)` if `True`.
        """
        # Embed input token IDs
        x = self.embedding(sequence)  # (B, L, D)

        # Apply Hyena layers
        for hyena_layer in self.hyena_layers:
            x = hyena_layer(
                x, attention_mask=attention_mask, is_causal=False
            )  # (B, L, D)

        # Apply layer normalization
        x = self.normalization(x)  # (B, L, D)

        if return_sequence:
            return x  # (B, L, D)
        else:
            # Pooling to get a single embedding per sequence
            pooled = self.pooling(x, attention_mask=attention_mask)  # (B, D)
            return pooled  # (B, D)
