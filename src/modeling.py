from dataclasses import dataclass
import warnings

import torch
from torch import nn
from torchvision import models
from torchvision.models import AlexNet_Weights

from .config import DEFAULT_LABEL_SMOOTHING, DEFAULT_WEIGHT_DECAY, TrainingStage


@dataclass
class ModelBundle:
    model: nn.Module
    weights: AlexNet_Weights | None


def build_alexnet_model(
    num_classes: int,
    device: torch.device,
    use_pretrained: bool = True,
) -> ModelBundle:
    weights = AlexNet_Weights.IMAGENET1K_V1 if use_pretrained else None

    try:
        model = models.alexnet(weights=weights)
    except Exception as exc:
        warnings.warn(
            f"Could not load pretrained AlexNet weights ({exc}). Falling back to random initialization.",
            stacklevel=2,
        )
        model = models.alexnet(weights=None)
        weights = None

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device)
    return ModelBundle(model=model, weights=weights)


def set_feature_extractor_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def build_criterion(
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
) -> nn.Module:
    weight_tensor = class_weights.to(device) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)


def build_optimizer_and_scheduler(
    model: nn.Module,
    stage: TrainingStage,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
):
    if stage.freeze_features:
        parameter_groups = [
            {
                "params": [parameter for parameter in model.classifier.parameters() if parameter.requires_grad],
                "lr": stage.learning_rate,
            }
        ]
    else:
        parameter_groups = [
            {
                "params": [parameter for parameter in model.features.parameters() if parameter.requires_grad],
                "lr": stage.learning_rate * 0.2,
            },
            {
                "params": [parameter for parameter in model.classifier.parameters() if parameter.requires_grad],
                "lr": stage.learning_rate,
            },
        ]

    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, stage.epochs),
        eta_min=stage.learning_rate * 0.1,
    )
    return optimizer, scheduler
