import torch
import torch.nn.functional as F


def bidirectional_contrastive_loss(
    student_representations, teacher_representations, temperature=0.1
):
    """
    Compute the contrastive loss between student and teacher representations.

    :param student_representations: Tensor of shape [B, D] - pooled representations from the student encoder.
    :param teacher_representations: Tensor of shape [B, D] - pooled representations from the teacher encoder.
    :param temperature: Scaling factor for the contrastive loss.
    :return: Contrastive loss value.
    """
    student_normal = F.normalize(student_representations, p=2, dim=-1)
    teacher_normal = F.normalize(teacher_representations, p=2, dim=-1)

    sim = student_normal @ teacher_normal.T / temperature
    identity = torch.arange(sim.size(0), device=sim.device)

    return (F.cross_entropy(sim, identity) + F.cross_entropy(sim.T, identity)) / 2
