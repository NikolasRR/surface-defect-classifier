"""Swappable backbone registry for the NEU steel-defect classifier.

Add a new architecture by adding an entry to ARCHITECTURES; everything else
(freezing, classifier-head replacement, stage-2 unfreezing) is generic.
"""

from dataclasses import dataclass
from typing import Callable, List

import torch.nn as nn
from torchvision import models


@dataclass
class ArchitectureSpec:
    builder: Callable[..., nn.Module]
    weights: object
    classifier_attr: str  # dotted path to the final Linear layer, e.g. "fc" or "classifier.1"
    in_features: int
    finetune_blocks: List[str]  # named_parameters() prefixes to unfreeze in stage 2


ARCHITECTURES = {
    "resnet18": ArchitectureSpec(
        builder=models.resnet18,
        weights=models.ResNet18_Weights.DEFAULT,
        classifier_attr="fc",
        in_features=512,
        finetune_blocks=["layer4"],
    ),
    "resnet34": ArchitectureSpec(
        builder=models.resnet34,
        weights=models.ResNet34_Weights.DEFAULT,
        classifier_attr="fc",
        in_features=512,
        finetune_blocks=["layer4"],
    ),
    "efficientnet_b0": ArchitectureSpec(
        builder=models.efficientnet_b0,
        weights=models.EfficientNet_B0_Weights.DEFAULT,
        classifier_attr="classifier.1",
        in_features=1280,
        finetune_blocks=["features.7", "features.8"],
    ),
    "mobilenet_v3_small": ArchitectureSpec(
        builder=models.mobilenet_v3_small,
        weights=models.MobileNet_V3_Small_Weights.DEFAULT,
        classifier_attr="classifier.3",
        in_features=1024,
        finetune_blocks=["features.11", "features.12"],
    ),
}


def _set_classifier(backbone: nn.Module, dotted_path: str, in_features: int, num_classes: int):
    *parent_parts, last = dotted_path.split(".")
    parent = backbone
    for part in parent_parts:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    new_layer = nn.Linear(in_features, num_classes)
    if last.isdigit():
        parent[int(last)] = new_layer
    else:
        setattr(parent, last, new_layer)


class DefectClassifier(nn.Module):
    """Wraps a torchvision backbone for transfer learning, architecture-agnostic.

    Stage 1 (construction): backbone frozen, only the replaced classifier head trains.
    Stage 2 (unfreeze_finetune_block): the architecture's last block(s) become trainable too.
    """

    def __init__(self, architecture: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{architecture}'. Available: {list(ARCHITECTURES)}"
            )
        self.architecture = architecture
        self.spec = ARCHITECTURES[architecture]

        weights = self.spec.weights if pretrained else None
        self.backbone = self.spec.builder(weights=weights)

        for param in self.backbone.parameters():
            param.requires_grad = False

        _set_classifier(self.backbone, self.spec.classifier_attr, self.spec.in_features, num_classes)

    def unfreeze_finetune_block(self):
        prefixes = tuple(self.spec.finetune_blocks)
        for name, param in self.backbone.named_parameters():
            if name.startswith(prefixes):
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
