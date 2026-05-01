from typing import List, Union

import torch
import torch.nn as nn

FRONTEND_CFG: List[Union[int, str]] = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512,
]
BACKEND_CFG: List[Union[int, str]] = [512, 512, 512, 256, 128, 64]


class CSRNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frontend = _make_layers(FRONTEND_CFG, in_channels=3, dilation=1)
        self.backend = _make_layers(BACKEND_CFG, in_channels=512, dilation=2)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def _make_layers(
    cfg: List[Union[int, str]],
    in_channels: int,
    dilation: int,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    channels = in_channels
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = int(value)
        layers.append(
            nn.Conv2d(
                channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        channels = out_channels
    return nn.Sequential(*layers)
