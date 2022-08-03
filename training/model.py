"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
# from train import requires_grad

from training.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

import copy


def _repeat_tuple(t, n):
    r"""Repeat each element of `t` for `n` times.
    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in t for _ in range(n))


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class EqualConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        lr_mul=1,
        bias=True,
        bias_init=0,
        conv_transpose2d=False,
        activation=False,
    ):
        super().__init__()

        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size).div_(lr_mul)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))

            self.lr_mul = lr_mul
        else:
            self.lr_mul = None

        self.conv_transpose2d = conv_transpose2d

        if activation:
            self.activation = ScaledLeakyReLU(0.2)
            # self.activation = FusedLeakyReLU(out_channel)
        else:
            self.activation = False

    def forward(self, input):
        if self.lr_mul != None:
            bias = self.bias * self.lr_mul
        else:
            bias = None

        if self.conv_transpose2d:
            # out = F.conv_transpose2d(
            #     input,
            #     self.weight.transpose(0, 1) * self.scale,
            #     bias=bias,
            #     stride=self.stride,
            #     # padding=self.padding,
            #     padding=0,
            # )

            # group version for fast training
            batch, in_channel, height, width = input.shape
            input_temp = input.view(1, batch * in_channel, height, width)
            weight = self.weight.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(
                input_temp,
                weight * self.scale,
                bias=bias,
                padding=self.padding,
                stride=2,
                groups=batch,
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            out = F.conv2d(
                input,
                self.weight * self.scale,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
            )

        if self.activation:
            out = self.activation(out)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualConv1dGroup(nn.Module):  # 1d conv group
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        bias=True,
        bias_init=0,
        lr_mul=1,
        activation=False,
    ):
        super().__init__()

        self.out_channel = out_channel
        self.in_channel = in_channel
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channel, 1, kernel_size).div_(lr_mul)
        )
        self.scale = (1 / math.sqrt(kernel_size)) * lr_mul

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))

            self.lr_mul = lr_mul
        else:
            self.lr_mul = None

        if activation:
            self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, input):
        if self.lr_mul != None:
            bias = self.bias * self.lr_mul
        else:
            bias = None

        out = F.conv1d(
            input, self.weight * self.scale, bias=bias, groups=self.in_channel
        )

        if self.activation:
            out = self.activation(out)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        lr_mul=1,
    ):
        assert not (upsample and downsample)
        layers = []

        if upsample:
            stride = 2
            self.padding = 0
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                    conv_transpose2d=True,
                    lr_mul=lr_mul,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))

        else:

            if downsample:
                factor = 2
                p = (len(blur_kernel) - factor) + (kernel_size - 1)
                pad0 = (p + 1) // 2
                pad1 = p // 2

                layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

                stride = 2
                self.padding = 0

            else:
                stride = 1
                self.padding = kernel_size // 2

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], return_features=False
    ):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )
        self.return_features = return_features

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)

        skip = self.skip(input)
        out = (out2 + skip) / math.sqrt(2)

        if self.return_features:
            return out, out1, out2
        else:
            return out


class Discriminator(nn.Module):
    def __init__(self, is_gray, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        if not is_gray:
            convs = [ConvLayer(3, channels[size], 1)]
        else:
            convs = [ConvLayer(1, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)

        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.view(batch, -1)

        out = self.final_linear(out)
        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        blur_kernel,
        normalize_mode,
        upsample=False,
        activate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )

        if activate:
            self.activate = FusedLeakyReLU(out_channel)
        else:
            self.activate = None

    def forward(self, input, style):
        out = self.conv(input, style)

        if self.activate is not None:
            out = self.activate(out)
        return out


class ModulatedConv2d(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        normalize_mode,
        blur_kernel,
        upsample=False,
        downsample=False,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.normalize_mode = normalize_mode
        if normalize_mode == "InstanceNorm2d":
            self.norm = nn.InstanceNorm2d(in_channel, affine=False)
        elif normalize_mode == "BatchNorm2d":
            self.norm = nn.BatchNorm2d(in_channel, affine=False)

        self.beta = None

        self.gamma = EqualConv2d(
            style_dim,
            in_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            bias_init=1,
        )

        self.beta = EqualConv2d(
            style_dim,
            in_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            bias_init=0,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, stylecode):
        assert stylecode is not None
        batch, in_channel, height, width = input.shape
        repeat_size = input.shape[3] // stylecode.shape[3]

        gamma = self.gamma(stylecode)
        if self.beta:
            beta = self.beta(stylecode)
        else:
            beta = 0

        weight = self.scale * self.weight
        weight = weight.repeat(batch, 1, 1, 1, 1)

        if self.normalize_mode in ["InstanceNorm2d", "BatchNorm2d"]:
            input = self.norm(input)
        elif self.normalize_mode == "LayerNorm":
            input = nn.LayerNorm(input.shape[1:], elementwise_affine=False)(input)
        elif self.normalize_mode == "GroupNorm":
            input = nn.GroupNorm(2 ** 3, input.shape[1:], affine=False)(input)
        elif self.normalize_mode == None:
            pass
        else:
            print("not implemented normalization")

        input = input * gamma + beta

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class StyledResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        style_dim,
        blur_kernel,
        normalize_mode,
        global_feature_channel=None,
    ):
        super().__init__()

        if style_dim is None:
            if global_feature_channel is not None:
                self.conv1 = StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    in_channel + global_feature_channel,
                    blur_kernel=blur_kernel,
                    upsample=True,
                    normalize_mode=normalize_mode,
                )
                self.conv2 = StyledConv(
                    out_channel,
                    out_channel,
                    3,
                    out_channel + global_feature_channel,
                    blur_kernel=blur_kernel,
                    normalize_mode=normalize_mode,
                )
            else:
                self.conv1 = StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    in_channel,
                    blur_kernel=blur_kernel,
                    upsample=True,
                    normalize_mode=normalize_mode,
                )
                self.conv2 = StyledConv(
                    out_channel,
                    out_channel,
                    3,
                    out_channel,
                    blur_kernel=blur_kernel,
                    normalize_mode=normalize_mode,
                )
        else:
            self.conv1 = StyledConv(
                in_channel,
                out_channel,
                3,
                style_dim,
                blur_kernel=blur_kernel,
                upsample=True,
                normalize_mode=normalize_mode,
            )
            self.conv2 = StyledConv(
                out_channel,
                out_channel,
                3,
                style_dim,
                blur_kernel=blur_kernel,
                normalize_mode=normalize_mode,
            )

        self.skip = ConvLayer(
            in_channel, out_channel, 1, upsample=True, activate=False, bias=False
        )

    def forward(self, input, stylecodes):
        out = self.conv1(input, stylecodes[0])
        out = self.conv2(out, stylecodes[1])

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample, blur_kernel):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(
            in_channel, 3, 1, style_dim, blur_kernel=blur_kernel, normalize_mode=None
        )

        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        latent_spatial_size,
        channel_multiplier,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        channels = {
            1: 512,
            2: 512,
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.from_rgb = ConvLayer(3, channels[size], 1)
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2))
        self.log_size = log_size

        in_channel = channels[size]
        end = int(math.log(latent_spatial_size, 2))

        for i in range(self.log_size, end, -1):
            out_channel = channels[2 ** (i - 1)]

            self.convs.append(
                ResBlock(in_channel, out_channel, blur_kernel, return_features=True)
            )

            in_channel = out_channel

        self.final_conv = ConvLayer(in_channel, style_dim, 3)

    def forward(self, input):
        out = self.from_rgb(input)

        for convs in self.convs:
            out, _, _ = convs(out)

        out = self.final_conv(out)

        return out  # spatial style code


class Decoder(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        latent_spatial_size,
        channel_multiplier,
        blur_kernel,
        normalize_mode,
        lr_mul,
        small_generator,
    ):
        super().__init__()

        self.size = size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.log_size = int(math.log(size, 2))

        self.input = ConstantInput(
            channels[latent_spatial_size], size=latent_spatial_size
        )

        if small_generator:
            stylecode_dim = style_dim
        else:
            stylecode_dim = channels[latent_spatial_size]

        self.conv1 = StyledConv(
            channels[latent_spatial_size],
            channels[latent_spatial_size],
            3,
            stylecode_dim,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )
        in_channel = channels[latent_spatial_size]

        self.start_index = int(math.log(latent_spatial_size, 2)) + 1  # if 4x4 -> 3
        self.convs = nn.ModuleList()
        self.convs_latent = nn.ModuleList()

        self.convs_latent.append(
            ConvLayer(
                style_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
            )
        )  # 4x4
        self.convs_latent.append(
            ConvLayer(
                stylecode_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
            )
        )

        for i in range(self.start_index, self.log_size + 1):  # 8x8~ 128x128
            if small_generator:
                stylecode_dim_prev, stylecode_dim_next = style_dim, style_dim
            else:
                stylecode_dim_prev = channels[2 ** (i - 1)]
                stylecode_dim_next = channels[2 ** i]
            self.convs_latent.append(
                ConvLayer(
                    stylecode_dim_prev,
                    stylecode_dim_next,
                    3,
                    upsample=True,
                    bias=True,
                    activate=True,
                    lr_mul=lr_mul,
                )
            )
            self.convs_latent.append(
                ConvLayer(
                    stylecode_dim_next,
                    stylecode_dim_next,
                    3,
                    bias=True,
                    activate=True,
                    lr_mul=lr_mul,
                )
            )

        if small_generator:
            stylecode_dim = style_dim
        else:
            stylecode_dim = None

        for i in range(self.start_index, self.log_size + 1):
            out_channel = channels[2 ** i]
            self.convs.append(
                StyledResBlock(
                    in_channel,
                    out_channel,
                    stylecode_dim,
                    blur_kernel,
                    normalize_mode=normalize_mode,
                )
            )

            in_channel = out_channel

        if small_generator:
            stylecode_dim = style_dim
        else:
            stylecode_dim = channels[size]

        # add adain to to_rgb
        self.to_rgb = StyledConv(
            channels[size],
            3,
            1,
            stylecode_dim,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )

        self.num_stylecodes = self.log_size * 2 - 2 * (
            self.start_index - 2
        )  # the number of AdaIN layer(stylecodes)
        assert len(self.convs) * 2 + 2 == self.num_stylecodes

        self.latent_spatial_size = latent_spatial_size

    def forward(self, style_code, mix_space=None, mask=None):
        if mix_space == None:
            batch = style_code.shape[0]
        elif mix_space == "demo":
            _, _, interpolation_step = mask
            batch = interpolation_step
        else:
            batch = style_code[0].shape[0]

        style_codes = []

        if mix_space == None:
            for i in range(self.num_stylecodes):
                style_code = self.convs_latent[i](style_code)
                style_codes.append(style_code)

        elif mix_space.startswith("stylemixing"):  # style mixing
            layer_position = mix_space.split("_")[1]

            stylecode1 = style_code[0]
            stylecode2 = style_code[1]
            style_codes1 = []
            style_codes2 = []

            for i in range(self.num_stylecodes):
                stylecode1 = self.convs_latent[i](stylecode1)
                stylecode2 = self.convs_latent[i](stylecode2)
                style_codes1.append(stylecode1)
                style_codes2.append(stylecode2)

            style_codes = style_codes1

            if layer_position == "coarse":
                style_codes[2:] = style_codes2[2:]
            elif layer_position == "fine":
                style_codes[:2] = style_codes2[:2]

        elif mix_space == "w":  # mix stylemaps in W space
            _, _, H, W = style_code[0].shape
            ratio = self.size // H
            mask_for_latent = nn.MaxPool2d(kernel_size=ratio, stride=ratio)(mask)

            style_code = torch.where(mask_for_latent > -1, style_code[1], style_code[0])
            for i in range(self.num_stylecodes):
                style_code = self.convs_latent[i](style_code)
                style_codes.append(style_code)

        elif mix_space == "w_plus":  # mix stylemaps in W+ space
            style_code1 = style_code[0]
            style_code2 = style_code[1]
            style_codes1 = []
            style_codes2 = []
            # print('111In model.py, style_code1: ', style_code1.shape)
            # print('222In model.py, style_code2" ', style_code2.shape)

            for up_layer in self.convs_latent:
                style_code1 = up_layer(style_code1)
                style_code2 = up_layer(style_code2)
                style_codes1.append(style_code1)
                style_codes2.append(style_code2)

            for i in range(0, len(style_codes2)):
                _, C, H, W = style_codes2[i].shape
                # ratio = self.size // H # self.size is come from the loaded args of the pretrained checkpoint
                mask_size = mask.shape[1]
                ratio = mask_size // H

                # print('In model.py, mask: ', mask.shape)
                # print('In model.py, size: ', mask_size)
                # print('In model.pt, H: ', H)
                # print('In model.py, ratio: ', ratio)
                mask_for_latent = nn.MaxPool2d(kernel_size=ratio, stride=ratio)(mask)
                mask_for_latent = mask_for_latent.unsqueeze(1).repeat(1, C, 1, 1)
                # print('In model.py, mask_for_latent: ', mask_for_latent.shape)
                # print('In model.py, style_codes2: ', style_codes2[i].shape)
                # print('In model.py, style_codes1: ', style_codes1[i].shape)
                style_codes2[i] = torch.where(
                    mask_for_latent > -1, style_codes2[i], style_codes1[i]
                )

            style_codes = style_codes2

        elif mix_space == "demo":  # mix stylemaps in W+ space using masks
            image_masks, shift_values, interpolation_step = mask
            original_stylemap = style_code[0]
            reference_stylemaps = style_code[1]
            style_codes = []

            for up_layer in self.convs_latent:
                original_stylemap = up_layer(original_stylemap)
                reference_stylemaps = up_layer(reference_stylemaps)

                _, C, H, W = original_stylemap.shape
                ratio = self.size // H
                masks = nn.AvgPool2d(kernel_size=ratio, stride=ratio)(image_masks)

                p_xs, p_ys = shift_values

                mask_moved = torch.empty_like(masks).copy_(masks)
                for i in range(len(reference_stylemaps)):
                    p_y, p_x = int(p_ys[i] / ratio), int(p_xs[i] / ratio)
                    mask_moved[i, 0] = torch.roll(
                        mask_moved[i, 0], shifts=(p_y, p_x), dims=(0, 1)
                    )

                    if p_y >= 0:
                        mask_moved[i, 0, :p_y] = 0
                    else:
                        mask_moved[i, 0, p_y:] = 0

                    if p_x >= 0:
                        mask_moved[i, 0, :, :p_x] = 0
                    else:
                        mask_moved[i, 0, :, p_x:] = 0

                    masks[i, 0] = torch.roll(
                        mask_moved[i, 0], shifts=(-p_y, -p_x), dims=(0, 1)
                    )

                masks = masks.repeat(1, C, 1, 1)
                mask_moved = mask_moved.repeat(1, C, 1, 1)

                original_stylemap_all = original_stylemap.repeat(
                    interpolation_step, 1, 1, 1
                )

                for inter_s in range(interpolation_step):
                    weight = inter_s / (interpolation_step - 1)
                    for i in range(len(reference_stylemaps)):
                        current_mask = masks[i] > 0.5
                        current_mask_moved = mask_moved[i] > 0.5
                        original_stylemap_all[inter_s][current_mask_moved] += weight * (
                            reference_stylemaps[i][current_mask]
                            - original_stylemap_all[inter_s][current_mask_moved]
                        )

                style_codes.append(original_stylemap_all)

        out = self.input(batch)
        out = self.conv1(out, style_codes[0])

        for i in range(len(self.convs)):
            out = self.convs[i](out, [style_codes[2 * i + 1], style_codes[2 * i + 2]])
        image = self.to_rgb(out, style_codes[-1])

        return image


class Generator(nn.Module):
    def __init__(
        self,
        size,
        mapping_layer_num,
        style_dim,
        latent_spatial_size,
        lr_mul,
        channel_multiplier,
        normalize_mode,
        blur_kernel=[1, 3, 3, 1],
        small_generator=False,
    ):
        super().__init__()

        self.latent_spatial_size = latent_spatial_size
        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(mapping_layer_num):
            if i != (mapping_layer_num - 1):
                in_channel = style_dim
                out_channel = style_dim
            else:
                in_channel = style_dim
                out_channel = style_dim * latent_spatial_size * latent_spatial_size

            layers.append(
                EqualLinear(
                    in_channel, out_channel, lr_mul=lr_mul, activation="fused_lrelu"
                )
            )
        self.mapping_z = nn.Sequential(*layers)
        
        # is_own_data = True
        # if is_own_data:
        #     size = 512

        self.decoder = Decoder(
            size,
            style_dim,
            latent_spatial_size,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
            lr_mul=1,
            small_generator=small_generator,
        )  # always 1, always zero padding

    def forward(
        self,
        input,
        return_stylecode=False,
        input_is_stylecode=False,
        mix_space=None,
        mask=None,
        calculate_mean_stylemap=False,
        truncation=None,
        truncation_mean_latent=None,
    ):
        if calculate_mean_stylemap:  # calculate mean_latent
            stylecode = self.mapping_z(input)
            return stylecode.mean(0, keepdim=True)
        else:
            if input_is_stylecode:
                stylecode = input
            else:
                stylecode = self.mapping_z(input)
                if truncation != None and truncation_mean_latent != None:
                    stylecode = truncation_mean_latent + truncation * (
                        stylecode - truncation_mean_latent
                    )
                N, C = stylecode.shape
                stylecode = stylecode.reshape(
                    N, -1, self.latent_spatial_size, self.latent_spatial_size
                )

            image = self.decoder(stylecode, mix_space=mix_space, mask=mask)
            
            if return_stylecode == True:
                return image, stylecode
            else:
                return image, None

from training.fdc_utils import elementwise_mult_cast_int as emci
from training.fdc_layers import *

class LocalPathway(nn.Module):
    def __init__(self, is_gray, use_batchnorm=True, is_skip=False, feature_layer_dim=64, fm_mult=1.0):
        super(LocalPathway, self).__init__()
        n_fm_encoder = [64, 128, 256, 512]
        n_fm_decoder = [256, 128]
        n_fm_encoder = emci(n_fm_encoder, fm_mult)
        n_fm_decoder = emci(n_fm_decoder, fm_mult)
        self.is_skip = is_skip

        # Encoder
        if not is_gray:
            self.conv0 = sequential(conv(3, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        else:
            self.conv0 = sequential(conv(1, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        self.conv1 = sequential(conv(n_fm_encoder[0], n_fm_encoder[1], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[1], activation=nn.LeakyReLU()))
        self.conv2 = sequential(conv(n_fm_encoder[1], n_fm_encoder[2], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[2], activation=nn.LeakyReLU()))
        self.conv3 = sequential(conv(n_fm_encoder[2], n_fm_encoder[3], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[3], activation=nn.LeakyReLU()))

        self.conv0_1 = sequential(conv(3, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        self.conv1_1 = sequential(conv(n_fm_encoder[0], n_fm_encoder[1], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[1], activation=nn.LeakyReLU()))
        self.conv2_1 = sequential(conv(n_fm_encoder[1], n_fm_encoder[2], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[2], activation=nn.LeakyReLU()))
        self.conv3_1 = sequential(conv(n_fm_encoder[2], n_fm_encoder[3], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[3], activation=nn.LeakyReLU()))

        # Decoder
        if is_skip:
            self.deconv0 = deconv(n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select0 = sequential(conv(n_fm_decoder[0] + self.conv2.out_channels, n_fm_decoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[0], activation=nn.LeakyReLU()))

            self.deconv1 = deconv(self.after_select0.out_channels, n_fm_decoder[1], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select1 = sequential(conv(n_fm_decoder[1]+self.conv1.out_channels, n_fm_decoder[1], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[1], activation=nn.LeakyReLU()))

            self.deconv2 = deconv(self.after_select1.out_channels, feature_layer_dim, 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select2 = sequential(conv(feature_layer_dim+self.conv0.out_channels, feature_layer_dim, 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(feature_layer_dim, activation=nn.LeakyReLU()))
        else:
            # self.deconv0 = deconv(2*n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.deconv0 = deconv(n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select0 = sequential(conv(n_fm_decoder[0] + 0, n_fm_decoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[0], activation=nn.LeakyReLU()))

            self.deconv1 = deconv(self.after_select0.out_channels, n_fm_decoder[1], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select1 = sequential(conv(n_fm_decoder[1]+0, n_fm_decoder[1], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[1], activation=nn.LeakyReLU()))

            self.deconv2 = deconv(self.after_select1.out_channels, feature_layer_dim, 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select2 = sequential(conv(feature_layer_dim+0, feature_layer_dim, 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(feature_layer_dim, activation=nn.LeakyReLU()))
        
        if not is_gray:
            self.local_img = conv(feature_layer_dim, 3, 1, 1, 0, None, None, False)
        else:
            self.local_img = conv(feature_layer_dim, 1, 1, 1, 0, None, None, False)

    # def forward(self, x, x_1):
    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
    
        # conv0_1 = self.conv0_1(x_1)
        # conv1_1 = self.conv1_1(conv0_1)
        # conv2_1 = self.conv2_1(conv1_1)
        # conv3_1 = self.conv3_1(conv2_1)

        # out = torch.cat((conv3, conv3_1), dim=1)
        out = conv3

        if self.is_skip:
            deconv0 = self.deconv0(out)
            after_select0 = self.after_select0(torch.cat([deconv0, conv2], 1))
            deconv1 = self.deconv1(after_select0)
            after_select1 = self.after_select1(torch.cat([deconv1, conv1], 1))
            deconv2 = self.deconv2(after_select1)
            after_select2 = self.after_select2(torch.cat([deconv2, conv0], 1))
        else:
            deconv0 = self.deconv0(out)
            after_select0 = self.after_select0(deconv0)
            deconv1 = self.deconv1(after_select0)
            after_select1 = self.after_select1(deconv1)
            deconv2 = self.deconv2(after_select1)
            after_select2 = self.after_select2(deconv2)

        # print('conv0 shape: ', conv0.shape)
        # print('conv1 shape: ', conv1.shape)
        # print('conv2 shape: ', conv2.shape)
        # print('conv3 shape: ', conv3.shape)
        # print('deconv0 shape: ', deconv0.shape)
        # print('after_select0 shape: ', after_select0.shape)
        # print('deconv1 shape: ', deconv1.shape)
        # print('after_select1 shape: ', after_select1.shape)
        # print('deconv2 shape: ', deconv2.shape)
        # print('after_select2 shape: ', after_select2.shape)

        local_img = self.local_img(after_select2)

        assert local_img.shape == x.shape, '{} {}'.format(local_img.shape, x.shape)        
        return local_img, deconv2

# Gloabl fusion for local method
class GlobalPathway_1(nn.Module):
    def __init__(self, is_gray, use_batchnorm=True, is_skip=False, feature_layer_dim=64, fm_mult=1.0):
        super(GlobalPathway_1, self).__init__()
        n_fm_encoder = [64, 128, 256, 512]
        n_fm_decoder = [256, 128]
        n_fm_encoder = emci(n_fm_encoder, fm_mult)
        n_fm_decoder = emci(n_fm_decoder, fm_mult)
        self.is_skip = is_skip

        # Encoder for reference image
        if not is_gray:
            self.conv0 = sequential(conv(3, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        else:
            self.conv0 = sequential(conv(1, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        self.conv1 = sequential(conv(n_fm_encoder[0], n_fm_encoder[1], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[1], activation=nn.LeakyReLU()))
        self.conv2 = sequential(conv(n_fm_encoder[1], n_fm_encoder[2], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[2], activation=nn.LeakyReLU()))
        self.conv3 = sequential(conv(n_fm_encoder[2], n_fm_encoder[3], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[3], activation=nn.LeakyReLU()))

        # Encoder for grayscale prediction image
        self.conv0_1 = sequential(conv(3, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        self.conv1_1 = sequential(conv(n_fm_encoder[0], n_fm_encoder[1], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[1], activation=nn.LeakyReLU()))
        self.conv2_1 = sequential(conv(n_fm_encoder[1], n_fm_encoder[2], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[2], activation=nn.LeakyReLU()))
        self.conv3_1 = sequential(conv(n_fm_encoder[2], n_fm_encoder[3], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[3], activation=nn.LeakyReLU()))

        # Decoder
        if is_skip:
            self.deconv0 = deconv(2*n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select0 = sequential(conv(n_fm_decoder[0] + self.conv2_1.out_channels, n_fm_decoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[0], activation=nn.LeakyReLU()))

            self.deconv1 = deconv(self.after_select0.out_channels, n_fm_decoder[1], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select1 = sequential(conv(n_fm_decoder[1]+self.conv1_1.out_channels, n_fm_decoder[1], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[1], activation=nn.LeakyReLU()))

            self.deconv2 = deconv(self.after_select1.out_channels, feature_layer_dim, 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select2 = sequential(conv(feature_layer_dim+self.conv0_1.out_channels, feature_layer_dim, 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(feature_layer_dim, activation=nn.LeakyReLU()))
        else:
            self.deconv0 = deconv(2*n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            # self.deconv0 = deconv(n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select0 = sequential(conv(n_fm_decoder[0] + 0, n_fm_decoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[0], activation=nn.LeakyReLU()))

            self.deconv1 = deconv(self.after_select0.out_channels, n_fm_decoder[1], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select1 = sequential(conv(n_fm_decoder[1]+0, n_fm_decoder[1], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[1], activation=nn.LeakyReLU()))

            self.deconv2 = deconv(self.after_select1.out_channels, feature_layer_dim, 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select2 = sequential(conv(feature_layer_dim+0, feature_layer_dim, 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(feature_layer_dim, activation=nn.LeakyReLU()))
        
        if not is_gray:
            self.local_img = conv(feature_layer_dim, 3, 1, 1, 0, None, None, False)
        else:
            self.local_img = conv(feature_layer_dim, 1, 1, 1, 0, None, None, False)

    def forward(self, x, x_1): # x: ref image x_1: overlaid
        # Encoding of the ref image
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        # Encoding of the prediction
        conv0_1 = self.conv0_1(x_1)
        conv1_1 = self.conv1_1(conv0_1)
        conv2_1 = self.conv2_1(conv1_1)
        conv3_1 = self.conv3_1(conv2_1)

        out = torch.cat((conv3, conv3_1), dim=1)
        # out = conv3

        if self.is_skip:
            deconv0 = self.deconv0(out)
            after_select0 = self.after_select0(torch.cat([deconv0, conv2_1], 1))
            deconv1 = self.deconv1(after_select0)
            after_select1 = self.after_select1(torch.cat([deconv1, conv1_1], 1))
            deconv2 = self.deconv2(after_select1)
            after_select2 = self.after_select2(torch.cat([deconv2, conv0_1], 1))
        else:
            deconv0 = self.deconv0(out)
            after_select0 = self.after_select0(deconv0)
            deconv1 = self.deconv1(after_select0)
            after_select1 = self.after_select1(deconv1)
            deconv2 = self.deconv2(after_select1)
            after_select2 = self.after_select2(deconv2)

        # print('conv0 shape: ', conv0.shape)
        # print('conv1 shape: ', conv1.shape)
        # print('conv2 shape: ', conv2.shape)
        # print('conv3 shape: ', conv3.shape)
        # print('deconv0 shape: ', deconv0.shape)
        # print('after_select0 shape: ', after_select0.shape)
        # print('deconv1 shape: ', deconv1.shape)
        # print('after_select1 shape: ', after_select1.shape)
        # print('deconv2 shape: ', deconv2.shape)
        # print('after_select2 shape: ', after_select2.shape)

        local_img = self.local_img(after_select2)

        # print('local img shape: ', local_img.shape)
        # print('x shape: ', x.shape)

        # assert local_img.shape == x.shape, '{} {}'.format(local_img.shape, x.shape)        
        return local_img, deconv2

# Global fusion for global method
class GlobalPathway_2(nn.Module):
    def __init__(self, is_gray, use_batchnorm=True, is_skip=False, feature_layer_dim=64, fm_mult=1.0):
        super(GlobalPathway_2, self).__init__()
        n_fm_encoder = [64, 128, 256, 512]
        n_fm_decoder = [256, 128]
        n_fm_encoder = emci(n_fm_encoder, fm_mult)
        n_fm_decoder = emci(n_fm_decoder, fm_mult)
        self.is_skip = is_skip

        # Encoder
        if not is_gray:
            self.conv0 = sequential(conv(3, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        else:
            self.conv0 = sequential(conv(1, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        self.conv1 = sequential(conv(n_fm_encoder[0], n_fm_encoder[1], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[1], activation=nn.LeakyReLU()))
        self.conv2 = sequential(conv(n_fm_encoder[1], n_fm_encoder[2], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[2], activation=nn.LeakyReLU()))
        self.conv3 = sequential(conv(n_fm_encoder[2], n_fm_encoder[3], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[3], activation=nn.LeakyReLU()))

        self.conv0_1 = sequential(conv(3, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        self.conv1_1 = sequential(conv(n_fm_encoder[0], n_fm_encoder[1], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[1], activation=nn.LeakyReLU()))
        self.conv2_1 = sequential(conv(n_fm_encoder[1], n_fm_encoder[2], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[2], activation=nn.LeakyReLU()))
        self.conv3_1 = sequential(conv(n_fm_encoder[2], n_fm_encoder[3], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[3], activation=nn.LeakyReLU()))

        # Decoder
        if is_skip:
            self.deconv0 = deconv(n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select0 = sequential(conv(n_fm_decoder[0] + self.conv2.out_channels, n_fm_decoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[0], activation=nn.LeakyReLU()))

            self.deconv1 = deconv(self.after_select0.out_channels, n_fm_decoder[1], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select1 = sequential(conv(n_fm_decoder[1]+self.conv1.out_channels, n_fm_decoder[1], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[1], activation=nn.LeakyReLU()))

            self.deconv2 = deconv(self.after_select1.out_channels, feature_layer_dim, 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select2 = sequential(conv(feature_layer_dim+self.conv0.out_channels, feature_layer_dim, 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(feature_layer_dim, activation=nn.LeakyReLU()))
        else:
            self.deconv0 = deconv(2*n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            # self.deconv0 = deconv(n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select0 = sequential(conv(n_fm_decoder[0] + 0, n_fm_decoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[0], activation=nn.LeakyReLU()))

            self.deconv1 = deconv(self.after_select0.out_channels, n_fm_decoder[1], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select1 = sequential(conv(n_fm_decoder[1]+0, n_fm_decoder[1], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[1], activation=nn.LeakyReLU()))

            self.deconv2 = deconv(self.after_select1.out_channels, feature_layer_dim, 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select2 = sequential(conv(feature_layer_dim+0, feature_layer_dim, 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(feature_layer_dim, activation=nn.LeakyReLU()))
        
        if not is_gray:
            self.local_img = conv(feature_layer_dim, 3, 1, 1, 0, None, None, False)
        else:
            self.local_img = conv(feature_layer_dim, 1, 1, 1, 0, None, None, False)

    def forward(self, x, x_1): # x: overlaid; x_1: ref_img
    # def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
    
        conv0_1 = self.conv0_1(x_1)
        conv1_1 = self.conv1_1(conv0_1)
        conv2_1 = self.conv2_1(conv1_1)
        conv3_1 = self.conv3_1(conv2_1)

        out = torch.cat((conv3, conv3_1), dim=1)
        # out = conv3

        if self.is_skip:
            deconv0 = self.deconv0(out)
            after_select0 = self.after_select0(torch.cat([deconv0, conv2], 1))
            deconv1 = self.deconv1(after_select0)
            after_select1 = self.after_select1(torch.cat([deconv1, conv1], 1))
            deconv2 = self.deconv2(after_select1)
            after_select2 = self.after_select2(torch.cat([deconv2, conv0], 1))
        else:
            deconv0 = self.deconv0(out)
            after_select0 = self.after_select0(deconv0)
            deconv1 = self.deconv1(after_select0)
            after_select1 = self.after_select1(deconv1)
            deconv2 = self.deconv2(after_select1)
            after_select2 = self.after_select2(deconv2)

        # print('conv0 shape: ', conv0.shape)
        # print('conv1 shape: ', conv1.shape)
        # print('conv2 shape: ', conv2.shape)
        # print('conv3 shape: ', conv3.shape)
        # print('deconv0 shape: ', deconv0.shape)
        # print('after_select0 shape: ', after_select0.shape)
        # print('deconv1 shape: ', deconv1.shape)
        # print('after_select1 shape: ', after_select1.shape)
        # print('deconv2 shape: ', deconv2.shape)
        # print('after_select2 shape: ', after_select2.shape)

        local_img = self.local_img(after_select2)

        # print('local img shape: ', local_img.shape)
        # print('x shape: ', x.shape)

        # assert local_img.shape == x.shape, '{} {}'.format(local_img.shape, x.shape)        
        return local_img, deconv2

class GlobalPathway(nn.Module):
    def __init__(self, is_gray, use_batchnorm=True, is_skip=False, feature_layer_dim=64, fm_mult=1.0):
        super(GlobalPathway, self).__init__()
        n_fm_encoder = [64, 128, 256, 512]
        n_fm_decoder = [256, 128]
        n_fm_encoder = emci(n_fm_encoder, fm_mult)
        n_fm_decoder = emci(n_fm_decoder, fm_mult)
        self.is_skip = is_skip

        # Encoder
        if not is_gray:
            self.conv0 = sequential(conv(4, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        else:
            self.conv0 = sequential(conv(2, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        self.conv1 = sequential(conv(n_fm_encoder[0], n_fm_encoder[1], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[1], activation=nn.LeakyReLU()))
        self.conv2 = sequential(conv(n_fm_encoder[1], n_fm_encoder[2], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[2], activation=nn.LeakyReLU()))
        self.conv3 = sequential(conv(n_fm_encoder[2], n_fm_encoder[3], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[3], activation=nn.LeakyReLU()))

        self.conv0_1 = sequential(conv(3, n_fm_encoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[0], activation=nn.LeakyReLU()))
        self.conv1_1 = sequential(conv(n_fm_encoder[0], n_fm_encoder[1], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[1], activation=nn.LeakyReLU()))
        self.conv2_1 = sequential(conv(n_fm_encoder[1], n_fm_encoder[2], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[2], activation=nn.LeakyReLU()))
        self.conv3_1 = sequential(conv(n_fm_encoder[2], n_fm_encoder[3], 3, 2, 1, 'kaiming', nn.LeakyReLU(1e-2), use_batchnorm), ResidualBlock(n_fm_encoder[3], activation=nn.LeakyReLU()))

        # Decoder
        if is_skip:
            self.deconv0 = deconv(n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select0 = sequential(conv(n_fm_decoder[0] + self.conv2.out_channels, n_fm_decoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[0], activation=nn.LeakyReLU()))

            self.deconv1 = deconv(self.after_select0.out_channels, n_fm_decoder[1], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select1 = sequential(conv(n_fm_decoder[1]+self.conv1.out_channels, n_fm_decoder[1], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[1], activation=nn.LeakyReLU()))

            self.deconv2 = deconv(self.after_select1.out_channels, feature_layer_dim, 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select2 = sequential(conv(feature_layer_dim+self.conv0.out_channels, feature_layer_dim, 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(feature_layer_dim, activation=nn.LeakyReLU()))
        else:
            # self.deconv0 = deconv(2*n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.deconv0 = deconv(n_fm_encoder[3], n_fm_decoder[0], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select0 = sequential(conv(n_fm_decoder[0] + 0, n_fm_decoder[0], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[0], activation=nn.LeakyReLU()))

            self.deconv1 = deconv(self.after_select0.out_channels, n_fm_decoder[1], 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select1 = sequential(conv(n_fm_decoder[1]+0, n_fm_decoder[1], 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(n_fm_decoder[1], activation=nn.LeakyReLU()))

            self.deconv2 = deconv(self.after_select1.out_channels, feature_layer_dim, 3, 2, 1, 1, 'kaiming', nn.ReLU(), use_batchnorm)
            self.after_select2 = sequential(conv(feature_layer_dim+0, feature_layer_dim, 3, 1, 1, 'kaiming', nn.LeakyReLU(), use_batchnorm), ResidualBlock(feature_layer_dim, activation=nn.LeakyReLU()))
        
        if not is_gray:
            self.local_img = conv(feature_layer_dim, 3, 1, 1, 0, None, None, False)
        else:
            self.local_img = conv(feature_layer_dim, 1, 1, 1, 0, None, None, False)

    # def forward(self, x, x_1):
    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
    
        # conv0_1 = self.conv0_1(x_1)
        # conv1_1 = self.conv1_1(conv0_1)
        # conv2_1 = self.conv2_1(conv1_1)
        # conv3_1 = self.conv3_1(conv2_1)

        # out = torch.cat((conv3, conv3_1), dim=1)
        out = conv3

        if self.is_skip:
            deconv0 = self.deconv0(out)
            after_select0 = self.after_select0(torch.cat([deconv0, conv2], 1))
            deconv1 = self.deconv1(after_select0)
            after_select1 = self.after_select1(torch.cat([deconv1, conv1], 1))
            deconv2 = self.deconv2(after_select1)
            after_select2 = self.after_select2(torch.cat([deconv2, conv0], 1))
        else:
            deconv0 = self.deconv0(out)
            after_select0 = self.after_select0(deconv0)
            deconv1 = self.deconv1(after_select0)
            after_select1 = self.after_select1(deconv1)
            deconv2 = self.deconv2(after_select1)
            after_select2 = self.after_select2(deconv2)

        # print('conv0 shape: ', conv0.shape)
        # print('conv1 shape: ', conv1.shape)
        # print('conv2 shape: ', conv2.shape)
        # print('conv3 shape: ', conv3.shape)
        # print('deconv0 shape: ', deconv0.shape)
        # print('after_select0 shape: ', after_select0.shape)
        # print('deconv1 shape: ', deconv1.shape)
        # print('after_select1 shape: ', after_select1.shape)
        # print('deconv2 shape: ', deconv2.shape)
        # print('after_select2 shape: ', after_select2.shape)

        local_img = self.local_img(after_select2)

        # print('local img shape: ', local_img.shape)
        # print('x shape: ', x.shape)

        # assert local_img.shape == x.shape, '{} {}'.format(local_img.shape, x.shape)        
        return local_img, deconv2

class LocalFuser(nn.Module):
    #differs from original code here
    #https://github.com/HRLTY/TP-GAN/blob/master/TP_GAN-Mar6FS.py

    def __init__(self ):
        super(LocalFuser,self).__init__()

    def forward( self , f_left_eye , f_right_eye, f_mouth, mask_input):
        mask = mask_input[0] # Only use the first mask to crop the size
        
        IMG_W = mask.shape[1]
        IMG_H = mask.shape[0]

        mask_left_pos = torch.where(mask==1) 
        left_v_min = torch.min(mask_left_pos[0])
        left_v_max = torch.max(mask_left_pos[0])
        left_u_min = torch.min(mask_left_pos[1])
        left_u_max = torch.max(mask_left_pos[1])

        left_pad_left = left_u_min 
        left_pad_right = IMG_W - left_u_max
        left_pad_top = left_v_min 
        left_pad_bottom = IMG_H - left_v_max

        LEFT_EYE_H = left_v_max - left_v_min 
        LEFT_EYE_W = left_u_max - left_u_min
        new_f_left_eye = F.interpolate(f_left_eye, size=(LEFT_EYE_H, LEFT_EYE_W), mode='bilinear')

        mask_right_pos = torch.where(mask==2)
        right_v_min = torch.min(mask_right_pos[0])
        right_v_max = torch.max(mask_right_pos[0])
        right_u_min = torch.min(mask_right_pos[1])
        right_u_max = torch.max(mask_right_pos[1])

        right_pad_left = right_u_min
        right_pad_right = IMG_W - right_u_max
        right_pad_top = right_v_min
        right_pad_bottom = IMG_H - right_v_max

        RIGHT_EYE_H = right_v_max - right_v_min
        RIGHT_EYE_W = right_u_max - right_u_min
        new_f_right_eye = F.interpolate(f_right_eye, size=(RIGHT_EYE_H, RIGHT_EYE_W), mode='bilinear')

        mask_mouth_pos = torch.where(mask==3)
        mouth_v_min = torch.min(mask_mouth_pos[0])
        mouth_v_max = torch.max(mask_mouth_pos[0])
        mouth_u_min = torch.min(mask_mouth_pos[1])
        mouth_u_max = torch.max(mask_mouth_pos[1])

        mouth_pad_left = mouth_u_min
        mouth_pad_right = IMG_W - mouth_u_max
        mouth_pad_top = mouth_v_min
        mouth_pad_bottom = IMG_H - mouth_v_max

        MOUTH_H = mouth_v_max - mouth_v_min
        MOUTH_W = mouth_u_max - mouth_u_min
        new_f_mouth = F.interpolate(f_mouth, size=(MOUTH_H, MOUTH_W), mode='bilinear')

        new_f_left_eye = torch.nn.functional.pad(new_f_left_eye , (left_pad_left, left_pad_right, left_pad_top, left_pad_bottom))
        new_f_right_eye = torch.nn.functional.pad(new_f_right_eye,(right_pad_left, right_pad_right, right_pad_top, right_pad_bottom))
        new_f_mouth = torch.nn.functional.pad(new_f_mouth,        (mouth_pad_left, mouth_pad_right, mouth_pad_top, mouth_pad_bottom))

        # final_out = torch.cat(( ref_gray, new_f_left_eye , new_f_right_eye, new_f_mouth), 1)
        final_out = torch.cat((new_f_left_eye , new_f_right_eye, new_f_mouth), 1)
        
        return final_out

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian

class CannyFilter(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=True):
        super(CannyFilter, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))
        self.is_gray = True

    def forward(self, img):
        img_r = img[:,0:1]
        if not self.is_gray:
            img_g = img[:,1:2]
            img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        if not self.is_gray:
            blur_horizontal = self.gaussian_filter_horizontal(img_g)
            blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
            blur_horizontal = self.gaussian_filter_horizontal(img_b)
            blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        if self.is_gray:
            blurred_img = blurred_img_r

        else:
            blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        if not self.is_gray:
            grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
            grad_y_g = self.sobel_filter_vertical(blurred_img_g)
            grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
            grad_y_b = self.sobel_filter_vertical(blurred_img_b)


        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        if not self.is_gray:
            grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
            grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        # grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        # grad_orientation += 180.0
        # grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # # THIN EDGES (NON-MAX SUPPRESSION)

        # all_filtered = self.directional_filter(grad_mag)

        # inidices_positive = (grad_orientation / 45) % 8
        # inidices_negative = ((grad_orientation / 45) + 4) % 8

        # height = inidices_positive.size()[2]
        # width = inidices_positive.size()[3]
        # pixel_count = height * width
        # pixel_range = torch.FloatTensor([range(pixel_count)])
        # if self.use_cuda:
        #     pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        # indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        # channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        # indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        # channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        # channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        # is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        # is_max = torch.unsqueeze(is_max, dim=0)

        # thin_edges = grad_mag.clone()
        # thin_edges[is_max==0] = 0.0

        # # THRESHOLD

        # thresholded = thin_edges.clone()
        # thresholded[thin_edges<self.threshold] = 0.0

        # early_threshold = grad_mag.clone()
        # early_threshold[grad_mag<self.threshold] = 0.0

        # assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        # return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold

        max_grad = torch.max(grad_mag)
        min_grad = torch.min(grad_mag)
        grad_range = max_grad - min_grad

        grad_mag -= min_grad
        grad_mag /= grad_range

        print('grad mag shape: ', grad_mag.shape)
        return grad_mag
