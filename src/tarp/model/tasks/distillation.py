from typing import final, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tarp.model.backbone.core import Encoder
from tarp.model.layers.perceptron.gated import SwishGatedLinearUnitFeedForward


@final
class CrossLanguageDistillationModel(nn.Module):
    def __init__(
        self,
        student_embedding: nn.Module,
        teacher_embedding: nn.Module,
        student: Encoder,
        teacher: Encoder,
        number_of_teacher_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.student_embedding = student_embedding
        self.teacher_embedding = teacher_embedding

        self.student = student
        self.teacher = teacher.freeze()

        self.number_of_teacher_heads = number_of_teacher_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dimension = teacher.encoding_size // number_of_teacher_heads

        # Student should be projected to the same dimension as the teacher for cross-attention.
        self.query_projection = nn.Linear(
            student.encoding_size,
            teacher.encoding_size,
            bias=bias,
        )

        self.output_projection = SwishGatedLinearUnitFeedForward(
            teacher.encoding_size,
            student.encoding_size,
            bias=bias,
        )

    @override
    def forward(
        self,
        student_sequence: Tensor,
        student_mask: Tensor,
        teacher_sequence: Tensor,
        teacher_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        student_embeddings = self.student_embedding(student_sequence)  # [B, L_s, D_s]
        student_encoded, student_auxiliary_loss = self.student(
            student_embeddings, student_mask, mode="sequence"
        )  # [B, L_s, D_s]

        with torch.no_grad():
            teacher_embeddings = self.teacher_embedding(
                teacher_sequence
            )  # [B, L_t, D_t]
            teacher_encoded, teacher_auxiliary_loss = self.teacher(
                teacher_embeddings, teacher_mask, mode="sequence"
            )  # [B, L_t, D_t]

            teacher_encoded = teacher_encoded.detach()  # [B, L_t, D_t]

        batch_size, student_sequence_length, _ = student_encoded.size()
        _, teacher_sequence_length, teacher_dimension = teacher_encoded.size()

        student_queries = (
            self.query_projection(student_encoded)
            .reshape(
                batch_size,
                student_sequence_length,
                self.number_of_teacher_heads,
                self.head_dimension,
            )
            .transpose(1, 2)
        )  # [B, H, L_s, D_h]

        kv = teacher_encoded.reshape(
            batch_size,
            teacher_sequence_length,
            self.number_of_teacher_heads,
            self.head_dimension,
        ).transpose(1, 2)  # [B, H, L_t, D_h]

        joint_mask = (
            student_mask.bool().unsqueeze(2) & teacher_mask.bool().unsqueeze(1)
        ).unsqueeze(1)  # [B, 1, L_s, L_t]

        context = F.scaled_dot_product_attention(
            query=student_queries,
            key=kv,
            value=kv,
            attn_mask=joint_mask,
        )  # [B, H, L_s, D_h]

        context = context.transpose(1, 2).reshape(
            batch_size, student_sequence_length, teacher_dimension
        )  # [B, L_s, D_t]

        # Sum auxiliary losses from student and teacher encoders if they are not None
        if student_auxiliary_loss is not None and teacher_auxiliary_loss is not None:
            total_auxiliary_loss = student_auxiliary_loss + teacher_auxiliary_loss
        else:
            total_auxiliary_loss = student_auxiliary_loss or teacher_auxiliary_loss

        oracle = self.output_projection(self.dropout(context))  # [B, L_s, D_s]

        return student_encoded, oracle, teacher_encoded, total_auxiliary_loss


@final
class ContrastiveLanguageDistillationModel(nn.Module):
    def __init__(
        self,
        student_embedding: nn.Module,
        teacher_embedding: nn.Module,
        student: Encoder,
        teacher: Encoder,
        projection_dimension: int = 256,
    ) -> None:
        super().__init__()

        self.student_embedding = student_embedding
        self.teacher_embedding = teacher_embedding

        self.student = student
        self.teacher = teacher.freeze()

        self.student_projection = nn.Linear(
            student.encoding_size,
            projection_dimension,
            bias=False,
        )
        self.teacher_projection = nn.Linear(
            teacher.encoding_size,
            projection_dimension,
            bias=False,
        )

    @override
    def forward(
        self,
        student_sequence: Tensor,
        student_mask: Tensor,
        teacher_sequence: Tensor,
        teacher_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        student_embeddings = self.student_embedding(student_sequence)

        student_encoded, student_pooled, student_auxiliary = self.student(
            student_embeddings,
            student_mask,
            mode="both",
        )

        with torch.no_grad():
            teacher_embeddings = self.teacher_embedding(teacher_sequence)
            teacher_pooled, teacher_auxiliary = self.teacher(
                teacher_embeddings,
                teacher_mask,
                mode="pooled",
            )

        student_representation = self.student_projection(student_pooled)
        teacher_representation = self.teacher_projection(teacher_pooled.detach())

        if student_auxiliary is not None and teacher_auxiliary is not None:
            auxiliary = student_auxiliary + teacher_auxiliary
        else:
            auxiliary = student_auxiliary or teacher_auxiliary

        return (
            student_encoded,
            student_representation,
            teacher_representation,
            auxiliary,
        )
