from collections import OrderedDict

from torch import nn


def make_layers(block: OrderedDict) -> nn.Sequential:
    """Build the convolutional subnets used by the upstream TrajGRU model."""
    layers = []
    for layer_name, values in block.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=values[0], stride=values[1], padding=values[2])
        elif "deconv" in layer_name:
            layer = nn.ConvTranspose2d(
                values[0], values[1], values[2], stride=values[3], padding=values[4]
            )
        elif "conv" in layer_name:
            layer = nn.Conv2d(
                values[0], values[1], values[2], stride=values[3], padding=values[4]
            )
        else:
            raise ValueError(f"Unsupported TrajGRU layer: {layer_name}")
        layers.append((layer_name, layer))
        if "leaky" in layer_name:
            layers.append((f"{layer_name}_activation", nn.LeakyReLU(0.2, inplace=True)))
        elif "relu" in layer_name:
            layers.append((f"{layer_name}_activation", nn.ReLU(inplace=True)))
        elif "sigmoid" in layer_name:
            layers.append((f"{layer_name}_activation", nn.Sigmoid()))
    return nn.Sequential(OrderedDict(layers))
