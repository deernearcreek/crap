import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from layers import SpatialTransformer, VecInt


class _ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, apply_norm=False):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel, stride, padding)
        self.activation = nn.LeakyReLU(0.2)
        self.apply_norm = apply_norm
        if apply_norm:
            self.norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        if self.apply_norm:
            out = self.activation(self.norm(self.main(x)))
        else:
            out = self.activation(self.main(x))
        return out


class _ResidualBlock(nn.Module):
    """
    Residual Block
    https://github.com/hhb072/IntroVAE
    """

    def __init__(self, in_channels=64, out_channels=64, scale=1.0, stride=1, apply_norm=False):
        super(_ResidualBlock, self).__init__()

        midc = int(out_channels * scale)
        self.apply_norm = apply_norm

        if (in_channels is not out_channels) or (stride != 1):
            self.conv_expand = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0,
                                         bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=midc, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = nn.Conv3d(in_channels=midc, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)

        if self.apply_norm:
            self.norm0 = nn.InstanceNorm3d(in_channels)
            self.norm1 = nn.InstanceNorm3d(midc)
            self.norm2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        if self.conv_expand is not None:
            if self.apply_norm:
                identity_data = self.conv_expand(nn.LeakyReLU(0.2)(self.norm0(x)))
            else:
                identity_data = self.conv_expand(nn.LeakyReLU(0.2)(x))
        else:
            identity_data = x

        if self.apply_norm:
            # output = self.relu1(self.norm1(self.conv1(x)))
            # output = self.conv2(output)
            # output = torch.add(output, identity_data)
            # output = self.relu2(self.norm2(output))
            output = self.conv1(self.relu1(self.norm1(x)))
            output = self.conv2(self.relu2(self.norm2(output)))
            output = torch.add(output, identity_data)
        else:
            # output = self.relu1(self.conv1(x))
            # output = self.conv2(output)
            # output = self.relu2(torch.add(output, identity_data))
            output = self.conv1(self.relu1(x))
            output = self.conv2(self.relu2(output))
            output = torch.add(output, identity_data)
        return output


class Encoder(nn.Module):
    def __init__(self, c_dim=1, channels=(16, 32, 64, 128, 128)):
        super(Encoder, self).__init__()

        self.enc0 = nn.Sequential(
            nn.Conv3d(c_dim, channels[0], 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2),
            nn.AvgPool3d(2),
        )

        self.enc1 = nn.Sequential(
            _ConvBlock(channels[0], channels[1], apply_norm=True),
            nn.AvgPool3d(2)
        )

        self.enc2 = nn.Sequential(
            _ConvBlock(channels[1], channels[2], apply_norm=True),
            nn.AvgPool3d(2)
        )

        self.enc3 = nn.Sequential(
            _ConvBlock(channels[2], channels[3], apply_norm=True),
            nn.AvgPool3d(2)
        )

        self.enc4 = nn.Sequential(
            _ConvBlock(channels[3], channels[4], apply_norm=True),
            nn.AvgPool3d(2)
        )

        self.bottleneck = nn.Sequential(
            _ResidualBlock(channels[4], channels[4], scale=1.0, stride=1, apply_norm=True),
        )

    def forward(self, x):
        x4 = self.enc0(x)
        x3 = self.enc1(x4)
        x2 = self.enc2(x3)
        x1 = self.enc3(x2)
        x0 = self.enc4(x1)
        return self.bottleneck(x0), x1, x2, x3, x4


class SynthDecoder(nn.Module):
    def __init__(self, c_dim=1, channels=(16, 32, 64, 128, 128), image_size=128):
        super(SynthDecoder, self).__init__()

        self.dec0 = nn.Sequential(
            _ConvBlock(channels[-1], channels[-2], apply_norm=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.dec1 = nn.Sequential(
            _ConvBlock(channels[-2] * 2, channels[-3], apply_norm=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.dec2 = nn.Sequential(
            _ConvBlock(channels[-3] * 2, channels[-4], apply_norm=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.dec3 = nn.Sequential(
            _ConvBlock(channels[-4] * 2, channels[-5], apply_norm=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.dec4 = nn.Sequential(
            _ConvBlock(channels[0] * 2, channels[0], apply_norm=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(channels[0], c_dim, 5, 1, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x0, x1, x2, x3, x4 = x
        z1 = self.dec0(x0)  # 128x8x10x8
        z2 = self.dec1(torch.cat((z1, x1), dim=1))  # 64x16x20x16
        z3 = self.dec2(torch.cat((z2, x2), dim=1))  # 32x32x40x32
        z4 = self.dec3(torch.cat((z3, x3), dim=1))  # 16x64x80x64
        z = self.dec4(torch.cat((z4, x4), dim=1))  # 1x128x160x128
        return z, z4, z3, z2, z1


class RegDecoder(nn.Module):
    def __init__(self, c_dim=3, channels=(16, 32, 64, 128, 128), skip_connect=True):
        super(RegDecoder, self).__init__()

        self.skip_connect = skip_connect
        if skip_connect:
            self.dec0 = nn.Sequential(
                _ConvBlock(channels[-1] * 2, channels[-1] * 2),
                _ConvBlock(channels[-1] * 2, channels[-2], kernel=1, padding=0),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )
            # self.conv0 = nn.Conv3d(channels[-1] * 2, channels[-2], 1, 1, 0)

            self.dec1 = nn.Sequential(
                _ConvBlock(channels[-2] * 3, channels[-2] * 3),
                _ConvBlock(channels[-2] * 3, channels[-3], kernel=1, padding=0),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )
            # self.conv1 = nn.Conv3d(channels[-2], channels[-3], 1, 1, 0)

            self.dec2 = nn.Sequential(
                _ConvBlock(channels[-3] * 3, channels[-3] * 3),
                _ConvBlock(channels[-3] * 3, channels[-4], kernel=1, padding=0),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )
            # self.conv2 = nn.Conv3d(channels[-3], channels[-4], 1, 1, 0)

            self.dec3 = nn.Sequential(
                _ConvBlock(channels[-4] * 3, channels[-4] * 3),
                _ConvBlock(channels[-4] * 3, channels[-5], kernel=1, padding=0),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )
            # self.conv3 = nn.Conv3d(channels[-4], channels[-5], 1, 1, 0)

            self.dec4 = nn.Sequential(
                _ConvBlock(channels[0] * 3, channels[0] * 3),
                _ConvBlock(channels[0] * 3, channels[0], kernel=1, padding=0),
                nn.Upsample(scale_factor=2, mode='trilinear'),
                # nn.Conv3d(channels[0], c_dim, 5, 1, 2)
            )

        else:
            self.dec0 = nn.Sequential(
                _ConvBlock(channels[-1] * 2, channels[-2]),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

            self.dec1 = nn.Sequential(
                _ConvBlock(channels[-2], channels[-3]),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

            self.dec2 = nn.Sequential(
                _ConvBlock(channels[-3], channels[-4]),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

            self.dec3 = nn.Sequential(
                _ConvBlock(channels[-4], channels[-5]),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

            self.dec4 = nn.Sequential(
                _ConvBlock(channels[-5], channels[0]),
                nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.Conv3d(channels[0], c_dim, 5, 1, 2)
            )

        # init flow layer with small weights and bias
        self.final = _ConvBlock(channels[0], channels[0], apply_norm=False)
        self.flow = nn.Conv3d(channels[0], c_dim, 3, 1, 1)
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, z):
        if self.skip_connect:
            z0, z1, z2, z3, z4 = z
            r1 = self.dec0(z0)  # 128x8x10x8

            r2 = self.dec1(torch.cat((r1, z1), dim=1))  # 64x16x20x16

            r3 = self.dec2(torch.cat((r2, z2), dim=1))  # 32x40x32

            r4 = self.dec3(torch.cat((r3, z3), dim=1))  # 16x64x80x64

            r5 = self.dec4(torch.cat((r4, z4), dim=1))  # 128x160x128

            flow = self.flow(self.final(r5))
            return flow
        else:
            r0 = self.dec0(z)  # 128x8x10x8
            r1 = self.dec1(r0)  # 64x16x20x16
            r2 = self.dec2(r1)  # 32x40x32
            r3 = self.dec3(r2)  # 16x64x80x64
            z4 = self.dec4(r3)  # 128x160x128

            flow = self.flow(self.final(z4))
            return flow


class DualStreamReg(nn.Module):
    def __init__(self, cdim=1, channels=(16, 32, 64, 128, 128), image_size=(128, 160, 128),
                 skip_connect=True):
        """
        Single-Modality Dual-Stream Registration
        cdim: input image dimension
        channels: number of feature channels at each of the encoding / decoding convolutions
        image_size:
        separate_decoders: if true, no weight-sharing between MR and CBCT synthesis decoders;
                           if false, use weight-sharing
        skip_connect: if True, add skip connection from synthesis decoding branch to registration decoding branch
        """
        super(DualStreamReg, self).__init__()

        self.encoder_src = Encoder(cdim, channels)
        self.encoder_tgt = Encoder(cdim, channels)
        self.skip_connect = skip_connect
        self.decoder = RegDecoder(3, channels, skip_connect=skip_connect)
        self.transformer = SpatialTransformer(image_size)
        self.vectint = VecInt(image_size, 7)

    def forward(self, x, y):
        z_src = self.encoder_src(x)
        z_tgt = self.encoder_tgt(y)

        # Registration decoding branch
        # Adding detach to the skip connections from synthesis branch?
        if self.skip_connect:
            flow = self.decoder([torch.cat((z_src[0], z_tgt[0]), dim=1), torch.cat((z_src[1], z_tgt[1]), dim=1),
                                    torch.cat((z_src[2], z_tgt[2]), dim=1),
                                    torch.cat((z_src[3], z_tgt[3]), dim=1),
                                    torch.cat((z_src[4], z_tgt[4]), dim=1)])
        else:
            flow = self.decoder(torch.cat((z_src[0], z_tgt[0]), dim=1))
        flow = self.exp(flow)
        reg = self.warp_image(x, flow)
        return reg, flow

    def warp_image(self, img, flow):
        return self.transformer(img, flow)

    def exp(self, flow):
        return self.vectint(flow)