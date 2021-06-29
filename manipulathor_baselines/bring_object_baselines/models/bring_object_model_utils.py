from typing import Sequence, Union, Tuple

import torch.nn as nn
import torch.nn.functional as F


def upshufflenorelu(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
    )

def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode='bilinear') + y


def make_decoder(
        input_channels: int,
        decoder_layer_channels: Sequence[int],
        decoder_kernel_sizes: Sequence[Union[int, Tuple[int, int]]],
        decoder_strides: Sequence[Union[int, Tuple[int, int]]],
        decoder_paddings: Sequence[Union[int, Tuple[int, int]]],
        decoder_dilations: Sequence[Union[int, Tuple[int, int]]],
        output_height: int,
        output_width: int,
        output_channels: int,
        output_relu: bool = False,
) -> nn.Module:
    assert (
            len(decoder_layer_channels)
            == len(decoder_kernel_sizes)
            == len(decoder_strides)
            == len(decoder_paddings)
            == len(decoder_dilations)
    ), "Mismatched sizes: layers {} kernels {} strides {} paddings {} dilations {}"

    net = nn.Sequential()

    input_channels_list = [input_channels] + list(encoder_layer_channels)

    for it, current_channels in enumerate(encoder_layer_channels):
        net.add_module(
            "conv_{}".format(it),
            nn.Conv2d(
                in_channels=input_channels_list[it],
                out_channels=current_channels,
                kernel_size=encoder_kernel_sizes[it],
                stride=encoder_strides[it],
                padding=encoder_paddings[it],
                dilation=encoder_dilations[it],
            ),
        )
        if it < len(encoder_layer_channels) - 1:
            net.add_module("relu_{}".format(it), nn.ReLU(inplace=True))

    if output_relu:
        net.add_module("out_relu", nn.ReLU(True))

    return net

def make_encoder(
        input_channels: int,
        encoder_layer_channels: Sequence[int],
        encoder_kernel_sizes: Sequence[Union[int, Tuple[int, int]]],
        encoder_strides: Sequence[Union[int, Tuple[int, int]]],
        encoder_paddings: Sequence[Union[int, Tuple[int, int]]],
        encoder_dilations: Sequence[Union[int, Tuple[int, int]]],
) -> nn.Module:
    assert (
            len(encoder_layer_channels)
            == len(encoder_kernel_sizes)
            == len(encoder_strides)
            == len(encoder_paddings)
            == len(encoder_dilations)
    ), "Mismatched sizes: layers {} kernels {} strides {} paddings {} dilations {}"

    net = nn.Sequential()

    input_channels_list = [input_channels] + list(encoder_layer_channels)

    for it, current_channels in enumerate(encoder_layer_channels):
        net.add_module(
            "conv_{}".format(it),
            nn.Conv2d(
                in_channels=input_channels_list[it],
                out_channels=current_channels,
                kernel_size=encoder_kernel_sizes[it],
                stride=encoder_strides[it],
                padding=encoder_paddings[it],
                dilation=encoder_dilations[it],
            ),
        )
        if it < len(encoder_layer_channels) - 1:
            net.add_module("relu_{}".format(it), nn.ReLU(inplace=True))

    return net

class EncoderDecoderModule(nn.Module):
    def __init__(self,
                 input_channels: int,
                 encoder_layer_channels: Sequence[int],
                 encoder_kernel_sizes: Sequence[Union[int, Tuple[int, int]]],
                 encoder_strides: Sequence[Union[int, Tuple[int, int]]],
                 encoder_paddings: Sequence[Union[int, Tuple[int, int]]],
                 encoder_dilations: Sequence[Union[int, Tuple[int, int]]],
                 decoder_layer_channels: Sequence[int],
                 decoder_kernel_sizes: Sequence[Union[int, Tuple[int, int]]],
                 decoder_strides: Sequence[Union[int, Tuple[int, int]]],
                 decoder_paddings: Sequence[Union[int, Tuple[int, int]]],
                 decoder_dilations: Sequence[Union[int, Tuple[int, int]]],
                 output_height: int,
                 output_width: int,
                 output_channels: int,
                 output_relu: bool = False,
                 ):