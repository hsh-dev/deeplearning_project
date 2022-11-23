import math
from math import sqrt

import torch
import torch.nn as nn
from torch.nn import functional as F

from training_module.augmentation import upfirdn2d


class Discriminator(nn.Module):

    def __init__(self, size=128, channel_multiplier=1):
        super().__init__()

        # channels = {
        #     4: 512,
        #     8: 512,
        #     16: 512,
        #     32: 512,
        #     64: 256 * channel_multiplier,
        #     128: 128 * channel_multiplier,
        #     256: 64 * channel_multiplier,
        #     512: 32 * channel_multiplier,
        #     1024: 16 * channel_multiplier,
        # }

        channels = {
            4: 256,
            8: 256,
            16: 128,
            32: 128,
            64: 64 * channel_multiplier,
            128: 32 * channel_multiplier,
            256: 32 * channel_multiplier,
            512: 16 * channel_multiplier,
            1024: 8 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]

            convs.append(ResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4]),
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
        stddev = stddev.mean([ 2, 3, 4 ], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([ out, stddev ], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ConvLayer(nn.Sequential):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[ 1, 3, 3, 1 ],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=( pad0, pad1 )))

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
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class EqualConv2d(nn.Module):

    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = self.make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def make_kernel(self, k):
        k = torch.tensor(k, dtype=torch.float32)

        if k.ndim == 1:
            k = k[None, :] * k[:, None]

        k /= k.sum()

        return k

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class EqualLR:

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
            ) * scale
        )

    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


###################################################################################################
###################################################################################################


class OldGenerator(nn.Module):

    def __init__(
        self,
        size,
        style_dim,
        n_mlp=8,
        channel_multiplier=2,
        blur_kernel=[ 1, 3, 3, 1 ],
        lr_mlp=0.01
    ):
        """
        :input size: img size to generate
        :style_dim: latent vector size
        :n_mlp: number of mlps for style mapping network
        """
        super().__init__()

        log_size = int(math.log(size, 2))
        self.num_layers = (log_size - 2) * 2 + 1
        self.n_latent = log_size * 2 - 2
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

        style = [PixelNorm()]
        for _ in range(n_mlp):
            style.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*style)

        in_channels = channels[4]

        self.constant_input = ConstantInput(in_channels)
        self.styled_conv1 = StyledConv(
            in_channels, in_channels, 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(in_channels, style_dim, upsample=False)

        self.styled_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [ 1, 1, 2**res, 2**res ]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel
                )
            )
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

    def forward(self, styles):
        #1. prepare latent
        styles = self.style(styles)
        noise = [None] * self.num_layers

        inject_index = self.n_latent
        latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        #2. generator
        out = self.constant_input(latent.size(0))
        out = self.styled_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb(out, latent[:, 1])

        i = 1
        for styled_conv1, styled_conv2, noise1, noise2, to_rgb in zip(
            self.styled_convs[::2], self.styled_convs[1::2], noise[1::2], noise[2::2],
            self.to_rgbs
        ):
            out = styled_conv1(out, latent[:, i], noise=noise1)
            out = styled_conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)    #to_rgb(out) + upsample(skip)

            i += 2

        return skip


class ConstantInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.size(0)
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[ 1, 3, 3, 1 ],
        demodulate=True
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return


class ToRGB(nn.Module):

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[ 1, 3, 3, 1 ]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(
            in_channel, 3, 1, style_dim, demodulate=False
        )    #to RGB (channel=3) via 1x1conv
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)
