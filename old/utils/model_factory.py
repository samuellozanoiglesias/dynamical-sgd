import torch
import torch.nn as nn
import torchvision.models as models

from .config_wrapper import RunnerSettings


class FeatureStore:
    value = None


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, num_classes, use_bias=True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=use_bias))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes, bias=use_bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        h = self.feature_extractor(x)
        return self.fc(h)


def create_model(settings: RunnerSettings, device: torch.device):
    if settings.model_architecture == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=settings.num_classes)
        model.conv1 = nn.Conv2d(
            settings.input_ch,
            model.conv1.weight.shape[0],
            3,
            1,
            1,
            bias=False,
        )
        model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    else:
        if settings.dataset_name == "mnist":
            input_dim = settings.input_ch * settings.im_size * settings.im_size
        else:
            input_dim = 2
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=settings.mlp_hidden_dim,
            num_hidden_layers=settings.mlp_num_hidden_layers,
            num_classes=settings.num_classes,
            use_bias=settings.mlp_use_bias,
        )

    model = model.to(device)

    def hook(_module, input_t, _output):
        FeatureStore.value = input_t[0].detach()

    classifier = model.fc
    classifier.register_forward_hook(hook)
    return model, classifier, FeatureStore
