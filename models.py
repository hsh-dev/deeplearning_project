import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch.nn import Conv2d, LeakyReLU, Sequential, BatchNorm2d, ReLU, ConvTranspose2d


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_layer = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            LeakyReLU(0.2),
        )

        ## DOWN SCALE
        self.down_1 = Sequential(
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.2),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
        )
        self.down_2 = Sequential(
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.2),
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
        )

        self.out_layer = Sequential(
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2),
            Conv2d(256, 1, kernel_size=7, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.out_layer(x)

        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.in_layer = Sequential(
            Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            BatchNorm2d(64),
            ReLU(0.2),
        )

        ## DOWN SCALE
        self.down_1 = Sequential(
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(0.2),
        )
        self.down_2 = Sequential(
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(0.2),
        )

        res_block_counts = 8

        self.res_blocks = Sequential()
        for i in range(res_block_counts):
            self.res_blocks.append(ResidualBlock())

        ## UP SCALE
        self.up_1 = Sequential(
            ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )

        self.up_2 = Sequential(
            ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.out_layer = Conv2d(64, 3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.down_1(x)
        x = self.down_2(x)

        x = self.res_blocks(x)

        x = self.up_1(x)
        x = self.up_2(x)
        x = self.out_layer(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.res_layer = Sequential(
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
        )
        self.act_func = ReLU()

    def forward(self, x):
        shortcut = x
        x = self.res_layer(x)
        x = x + shortcut
        x = self.act_func(x)

        return x


##############################################
##############################################


class Block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):

    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([
            Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):

    def __init__(self, chs):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)
        ])
        self.dec_blocks = nn.ModuleList([
            Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)
        ])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([ x, enc_ftrs ], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([ H, W ])(enc_ftrs)
        return enc_ftrs


class UNetGenerator(nn.Module):

    def __init__(
        self,
        enc_chs=( 3, 64, 128, 256, 512 ),
        dec_chs=( 512, 256, 128, 64 ),
        num_class=3,
        retain_dim=False,
        out_sz=( 128, 128 )
    ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)

        # for i in range(len(enc_ftrs)):
        #     print(enc_ftrs[i].shape)

        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        out = self.head(out)
        # if self.retain_dim:
        #     out = F.interpolate(out, self.out_sz)
        return out