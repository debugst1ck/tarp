import torch
from torch import Tensor


def accuracy(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction == target).float().mean()


def subset_accuracy(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction == target).all(dim=1).float().mean()


def precision(prediction: Tensor, target: Tensor) -> Tensor:
    true_positives = ((prediction == 1) & (target == 1)).sum().float()
    predicted_positives = (prediction == 1).sum().float()
    return true_positives / (
        predicted_positives + torch.finfo(predicted_positives.dtype).eps
    )


def recall(prediction: Tensor, target: Tensor) -> Tensor:
    true_positives = ((prediction == 1) & (target == 1)).sum().float()
    actual_positives = (target == 1).sum().float()
    return true_positives / (actual_positives + torch.finfo(actual_positives.dtype).eps)


def macro_f1_score(prediction: Tensor, target: Tensor) -> Tensor:
    true_positive = ((prediction == 1) & (target == 1)).sum(dim=0).float()
    false_positive = ((prediction == 1) & (target == 0)).sum(dim=0).float()
    false_negative = ((prediction == 0) & (target == 1)).sum(dim=0).float()

    per_class_f1 = (2 * true_positive) / (
        2 * true_positive
        + false_positive
        + false_negative
        + torch.finfo(true_positive.dtype).eps
    )
    return per_class_f1.mean()


def micro_f1_score(prediction: Tensor, target: Tensor) -> Tensor:
    # Flatten everything to calculate global (micro) F1
    pred_flat = prediction.view(-1)
    target_flat = target.view(-1)

    true_positive = ((pred_flat == 1) & (target_flat == 1)).sum().float()
    false_positive = ((pred_flat == 1) & (target_flat == 0)).sum().float()
    false_negative = ((pred_flat == 0) & (target_flat == 1)).sum().float()

    f1 = (2 * true_positive) / (
        2 * true_positive
        + false_positive
        + false_negative
        + torch.finfo(true_positive.dtype).eps
    )
    return f1


def top_k_accuracy(prediction: Tensor, target: Tensor, k: int) -> Tensor:
    top_k_indices = torch.topk(prediction, k, dim=1).indices
    target_at_top_k = torch.gather(target, dim=1, index=top_k_indices)
    correct = target_at_top_k.any(dim=1).float()
    return correct.mean()
