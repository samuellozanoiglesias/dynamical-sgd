
from __future__ import annotations

from math import prod
from typing import Sequence

import torch
import torch.nn as nn
import torchvision.models as tv_models


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        num_classes: int,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=use_bias))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        h = self.feature_extractor(x)
        return self.fc(h)


def build_model(
    model_cfg: dict,
    input_shape: Sequence[int],
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    architecture = str(model_cfg.get("architecture", "mlp")).strip().lower()
    use_bias = bool(model_cfg.get("use_bias", True))

    if architecture == "mlp":
        input_dim = int(prod(input_shape))
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("nn_width", 1000)),
            num_hidden_layers=int(model_cfg.get("num_hidden_layers", 1)),
            num_classes=num_classes,
            use_bias=use_bias,
        )
        # Replace ReLU with Identity
        for i, layer in enumerate(model.feature_extractor):
            if isinstance(layer, nn.ReLU):
                model.feature_extractor[i] = nn.Identity()
        return model.to(device)
    
    if architecture == "mlp_relu":
        input_dim = int(prod(input_shape))
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("nn_width", 1000)),
            num_hidden_layers=int(model_cfg.get("num_hidden_layers", 1)),
            num_classes=num_classes,
            use_bias=use_bias,
        )
        return model.to(device)

    if architecture == "mlp_tanh":
        input_dim = int(prod(input_shape))
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("nn_width", 1000)),
            num_hidden_layers=int(model_cfg.get("num_hidden_layers", 1)),
            num_classes=num_classes,
            use_bias=use_bias,
        )
        # Replace ReLU with Tanh
        for i, layer in enumerate(model.feature_extractor):
            if isinstance(layer, nn.ReLU):
                model.feature_extractor[i] = nn.Tanh()
        return model.to(device)

    if architecture == "resnet18":
        if len(input_shape) < 3:
            raise ValueError(
                "ResNet18 requires image-like inputs (C,H,W). "
                f"Received input shape {tuple(input_shape)}."
            )
        model = tv_models.resnet18(weights=None, num_classes=num_classes)
        in_channels = int(input_shape[0]) if len(input_shape) > 0 else 1
        model.conv1 = nn.Conv2d(
            in_channels,
            model.conv1.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        return model.to(device)

    raise ValueError(f"Unsupported model architecture '{architecture}'.")