import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from layers import SpatialTransformer, VecInt, ResizeTransform, GumbelSoftmax


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu and normalization for unet.
    """

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1,
                 apply_norm=False, apply_act=True, norm='IN'):
        super(ConvBlock, self).__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel, stride, padding)
        nn.init.kaiming_normal_(self.main.weight)
        self.activation = nn.LeakyReLU(0.2)
        self.apply_norm = apply_norm
        self.norm = norm
        if apply_norm:
            if norm == 'IN':
                self.norm = nn.InstanceNorm3d(out_channels)
            elif norm == 'BN':
                self.norm = nn.BatchNorm3d(out_channels)
            elif norm == 'SN':
                self.norm = nn.utils.spectral_norm
        self.apply_act = apply_act

    def forward(self, x):
        if self.apply_norm:
            if self.norm == 'SN':
                x = self.norm(self.main)(x)
            else:
                x = self.norm(self.main(x))
        else:
            x = self.main(x)
        if self.apply_act:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block
    https://github.com/hhb072/IntroVAE
    """

    def __init__(self, in_channels=64, out_channels=64, scale=1.0, stride=1, apply_norm=False):
        super(ResidualBlock, self).__init__()

        midc = int(out_channels * scale)
        self.apply_norm = apply_norm

        if (in_channels is not out_channels) or (stride != 1):
            self.conv_expand = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                         stride=stride, padding=0,
                                         bias=False)
            nn.init.kaiming_normal_(self.conv_expand.weight)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=midc, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = nn.Conv3d(in_channels=midc, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        nn.init.kaiming_normal_(self.conv2.weight)
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


class Downsample(nn.Module):
    """
    Downsampling
    """

    def __init__(self, in_channels, out_channels, kernel=3, apply_norm=True, apply_act=True, norm='IN'):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel, stride=2, padding=kernel // 2, bias=False)
        nn.init.kaiming_normal_(self.main.weight)
        self.apply_norm = apply_norm
        self.norm = norm
        if apply_norm:
            if norm == 'IN':
                self.norm = nn.InstanceNorm3d(out_channels)
            elif norm == 'BN':
                self.norm = nn.BatchNorm3d(out_channels)
            else:
                self.apply_norm = False
        self.apply_act = apply_act
        if self.apply_act:
            self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.apply_norm:
            x = self.norm(self.main(x))
        else:
            x = self.main(x)
        if self.apply_act:
            x = self.activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self, c_dim=1, channels=(16, 32, 64, 128, 128),
                 res=False, norm='IN', gumbel=False, scale=1):
        super(Encoder, self).__init__()

        self.conv_init = ConvBlock(c_dim, channels[0], 3, 1, 1, apply_norm=False)
        # self.enc0 = nn.Sequential(
        #     nn.Conv3d(c_dim, channels[0], 5, 1, 2, bias=False),
        #     nn.LeakyReLU(0.2),
        #     nn.AvgPool3d(2),
        # )

        if res:
            self.enc0 = nn.Sequential(
                ResidualBlock(channels[0], channels[1], scale=scale, stride=2, apply_norm=True))
            self.enc1 = nn.Sequential(
                ResidualBlock(channels[1], channels[2], scale=scale, stride=2, apply_norm=True))
            self.enc2 = nn.Sequential(
                ResidualBlock(channels[2], channels[3], scale=scale, stride=2, apply_norm=True))
            self.enc3 = nn.Sequential(
                ResidualBlock(channels[3], channels[4], scale=scale, stride=2, apply_norm=True))
            self.bottleneck = nn.Sequential(
                ResidualBlock(channels[4], channels[4], scale=scale, stride=1, apply_norm=True))
        else:
            self.enc0 = nn.Sequential(
                ConvBlock(channels[0], channels[1], stride=2, apply_norm=True, norm=norm))
            self.enc1 = nn.Sequential(
                ConvBlock(channels[1], channels[1], apply_norm=True, norm=norm),
                ConvBlock(channels[1], channels[2], stride=2, apply_norm=True, norm=norm))
            self.enc2 = nn.Sequential(
                ConvBlock(channels[2], channels[2], apply_norm=True, norm=norm),
                ConvBlock(channels[2], channels[3], stride=2, apply_norm=True, norm=norm))
            self.enc3 = nn.Sequential(
                ConvBlock(channels[3], channels[3], apply_norm=True, norm=norm),
                ConvBlock(channels[3], channels[4], stride=2, apply_norm=True, norm=norm))
            self.bottleneck = nn.Sequential(
                ConvBlock(channels[4], channels[4], apply_norm=True, apply_act=~gumbel))
        if gumbel:
            self.bottleneck.add_module('gumbel', GumbelSoftmax(temperature=0.5))

    def forward(self, x):
        x4 = self.conv_init(x)
        x3 = self.enc0(x4)
        x2 = self.enc1(x3)
        x1 = self.enc2(x2)
        x0 = self.enc3(x1)
        return self.bottleneck(x0), x1, x2, x3, x4


class SynthDecoder(nn.Module):
    def __init__(self, c_dim=1, channels=(16, 32, 64, 128, 128), res=False, scale=1.0):
        super(SynthDecoder, self).__init__()

        if res:
            self.dec1 = nn.Sequential(
                ResidualBlock(channels[-2] * 2, channels[-2], scale=scale, apply_norm=True))
            self.dec2 = nn.Sequential(
                ResidualBlock(channels[-3] + channels[-2], channels[-3], scale=scale, apply_norm=True))
            self.dec3 = nn.Sequential(
                ResidualBlock(channels[-4] + channels[-3], channels[-4], scale=scale, apply_norm=True))
            self.dec4 = nn.Sequential(
                ResidualBlock(channels[-5] + channels[-4], channels[-5], scale=scale, apply_norm=True))
            self.dec_final = nn.Sequential(
                nn.Conv3d(channels[0], c_dim, 3, 1, 1),
                nn.Sigmoid())

        else:
            self.dec1 = nn.Sequential(
                ConvBlock(channels[-1] + channels[-2], channels[-2], apply_norm=True),
                ConvBlock(channels[-2], channels[-2], apply_norm=True),
            )
            self.dec2 = nn.Sequential(
                ConvBlock(channels[-2] + channels[-3], channels[-3], apply_norm=True),
                ConvBlock(channels[-3], channels[-3], apply_norm=True),
            )
            self.dec3 = nn.Sequential(
                ConvBlock(channels[-3] + channels[-4], channels[-4], apply_norm=True),
                ConvBlock(channels[-4], channels[-4], apply_norm=True),
            )
            self.dec4 = nn.Sequential(
                ConvBlock(channels[-4] + channels[-5], channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))
            self.dec_final = nn.Sequential(
                nn.Conv3d(channels[0], c_dim, 3, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x0, x1, x2, x3, x4 = x
        z1 = nn.Upsample(scale_factor=2, mode='trilinear')(x0)
        z1 = self.dec1(torch.cat((z1, x1), dim=1))  # 128x8x10x8
        z2 = nn.Upsample(scale_factor=2, mode='trilinear')(z1)
        z2 = self.dec2(torch.cat((z2, x2), dim=1))
        z3 = nn.Upsample(scale_factor=2, mode='trilinear')(z2)
        z3 = self.dec3(torch.cat((z3, x3), dim=1))
        z4 = nn.Upsample(scale_factor=2, mode='trilinear')(z3)
        z4 = self.dec4(torch.cat((z4, x4), dim=1))
        z = self.dec_final(z4)
        return z, z4, z3, z2, z1

class SynthDecoder_v2(nn.Module):
    def __init__(self, c_dim=1, channels=(16, 32, 64, 128, 128), res=False, scale=1.0):
        super(SynthDecoder_v2, self).__init__()

        if res:
            self.dec1 = nn.Sequential(
                ResidualBlock(channels[-2] * 2, channels[-2], scale=scale, apply_norm=True))
            self.dec2 = nn.Sequential(
                ResidualBlock(channels[-3] + channels[-2], channels[-3], scale=scale, apply_norm=True))
            self.dec3 = nn.Sequential(
                ResidualBlock(channels[-4] + channels[-3], channels[-4], scale=scale, apply_norm=True))
            self.dec4 = nn.Sequential(
                ResidualBlock(channels[-5] + channels[-4], channels[-5], scale=scale, apply_norm=True))
            self.dec_final = nn.Sequential(
                nn.Conv3d(channels[0], c_dim, 3, 1, 1),
                nn.Sigmoid())

        else:
            self.dec1 = nn.Sequential(
                ConvBlock(channels[-1] + channels[-2], channels[-2], apply_norm=True),
                ConvBlock(channels[-2], channels[-2], apply_norm=True),
            )
            self.dec2 = nn.Sequential(
                ConvBlock(channels[-2] + channels[-3], channels[-3], apply_norm=True),
                ConvBlock(channels[-3], channels[-3], apply_norm=True),
            )
            self.dec3 = nn.Sequential(
                ConvBlock(channels[-3] + channels[-4], channels[-4], apply_norm=True),
                ConvBlock(channels[-4], channels[-4], apply_norm=True),
            )
            self.dec4 = nn.Sequential(
                ConvBlock(channels[-4] + channels[-5], channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))
            self.dec_final = nn.Sequential(
                nn.Conv3d(channels[0], c_dim, 3, 1, 1),
                nn.Sigmoid()
            )
        self.ref1 = ConvBlock(channels[-2], channels[-2], apply_norm=True)
        self.ref2 = ConvBlock(channels[-3], channels[-3], apply_norm=True)
        self.ref3 = ConvBlock(channels[-4], channels[-4], apply_norm=True)
        self.ref4 = ConvBlock(channels[-5], channels[-5], apply_norm=True)
    def forward(self, x):
        x0, x1, x2, x3, x4 = x
        z1 = nn.Upsample(scale_factor=2, mode='trilinear')(x0)
        z1 = self.dec1(torch.cat((z1, x1), dim=1))  # 128x8x10x8
        z2 = nn.Upsample(scale_factor=2, mode='trilinear')(z1)
        z2 = self.dec2(torch.cat((z2, x2), dim=1))
        z3 = nn.Upsample(scale_factor=2, mode='trilinear')(z2)
        z3 = self.dec3(torch.cat((z3, x3), dim=1))
        z4 = nn.Upsample(scale_factor=2, mode='trilinear')(z3)
        z4 = self.dec4(torch.cat((z4, x4), dim=1))
        z = self.dec_final(z4)
        return z, self.ref4(z4), self.ref3(z3), self.ref2(z2), self.ref1(z1)
    
class RegDecoder(nn.Module):
    def __init__(self, c_dim=3, channels=(16, 32, 64, 128, 128), skip_connect=True, res=False):
        super(RegDecoder, self).__init__()

        self.skip_connect = skip_connect
        if res:
            if skip_connect:
                self.dec0 = nn.Sequential(
                    ResidualBlock(channels[-1] * 2, channels[-2], scale=0.5),  # channels[-2]/(channels[-1] * 2
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec1 = nn.Sequential(
                    ResidualBlock(channels[-2] * 3, channels[-3], scale=0.5),  # channels[-3]/(channels[-2] * 3)
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec2 = nn.Sequential(
                    ResidualBlock(channels[-3] * 3, channels[-4], scale=1.0),  # channels[-4]/(channels[-3] * 3)
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec3 = nn.Sequential(
                    ResidualBlock(channels[-4] * 3, channels[-5], scale=1.0),  # channels[-5]/(channels[-4] * 3)
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.final = ConvBlock(channels[0] * 3, channels[0], apply_norm=False)
            else:
                self.dec0 = nn.Sequential(
                    ResidualBlock(channels[-1] * 2, channels[-2], scale=channels[-2]/(channels[-1] * 2)),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec1 = nn.Sequential(
                    ResidualBlock(channels[-2], channels[-3], scale=channels[-3]/(channels[-2])),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec2 = nn.Sequential(
                    ResidualBlock(channels[-3], channels[-4], scale=channels[-4]/(channels[-3])),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec3 = nn.Sequential(
                    ResidualBlock(channels[-4], channels[-5], scale=channels[-5]/(channels[-4])),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.final = ConvBlock(channels[0], channels[0], apply_norm=False)
        else:
            if skip_connect:
                self.dec0 = nn.Sequential(
                    ConvBlock(channels[-1] * 2, channels[-2], apply_norm=True),
                    ConvBlock(channels[-2], channels[-2], apply_norm=True),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec1 = nn.Sequential(
                    ConvBlock(channels[-2] * 3, channels[-3], apply_norm=True),
                    ConvBlock(channels[-3], channels[-3], apply_norm=True),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec2 = nn.Sequential(
                    ConvBlock(channels[-3] * 3, channels[-4], apply_norm=True),
                    ConvBlock(channels[-4], channels[-4], apply_norm=True),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec3 = nn.Sequential(
                    ConvBlock(channels[-4] * 3, channels[-5], apply_norm=True),
                    ConvBlock(channels[-5], channels[-5], apply_norm=True),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.final = ConvBlock(channels[0] * 3, channels[0], apply_norm=False)
            else:
                self.dec0 = nn.Sequential(
                    ConvBlock(channels[-1] * 2, channels[-2], apply_norm=True),
                    ConvBlock(channels[-2], channels[-2], apply_norm=True),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec1 = nn.Sequential(
                    ConvBlock(channels[-2], channels[-3], apply_norm=True),
                    ConvBlock(channels[-3], channels[-3], apply_norm=True),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec2 = nn.Sequential(
                    ConvBlock(channels[-3], channels[-4], apply_norm=True),
                    ConvBlock(channels[-4], channels[-4], apply_norm=True),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.dec3 = nn.Sequential(
                    ConvBlock(channels[-4], channels[-5], apply_norm=True),
                    ConvBlock(channels[-5], channels[-5], apply_norm=True),
                    nn.Upsample(scale_factor=2, mode='trilinear'))
                self.final = ConvBlock(channels[0], channels[0], apply_norm=False)

        # init flow layer with small weights and bias
        self.flow = nn.Conv3d(channels[0], c_dim, 3, 1, 1)
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, z):
        if self.skip_connect:
            z0, z1, z2, z3, z4, = z
            r1 = self.dec0(z0)  # 128x8x10x8
            r2 = self.dec1(torch.cat((r1, z1), dim=1))  # 64x16x20x16
            r3 = self.dec2(torch.cat((r2, z2), dim=1))  # 32x40x32
            r4 = self.dec3(torch.cat((r3, z3), dim=1))  # 16x64x80x64
            r = self.final(torch.cat((r4, z4), dim=1))

        else:
            r1 = self.dec0(z)  # 128x8x10x8
            r2 = self.dec1(r1)  # 64x16x20x16
            r3 = self.dec2(r2)  # 32x40x32
            r4 = self.dec3(r3)  # 16x64x80x64
            r = self.final(r4)

        flow = self.flow(r)
        return flow


# class RegCascadedDecoder(nn.Module):
#     def __init__(self, c_dim=3, channels=(16, 32, 64, 128, 128), res=False):
#         super(RegCascadedDecoder, self).__init__()
#
#         self.resize = ResizeTransform(vel_resize=0.5, ndims=3)  # upsample and scale deformation field
#         if res:
#             self.dec0 = ResidualBlock(channels[-1] * 2, channels[-2], scale=0.5)
#             self.dec1 = ResidualBlock(channels[-2] * 3, channels[-3], scale=channels[-3]/channels[-2]/3)
#             self.dec2 = ResidualBlock(channels[-3] * 3, channels[-4], scale=channels[-4]/channels[-3]/3)
#             self.dec3 = ResidualBlock(channels[-4] * 3, channels[-5], scale=channels[-5]/channels[-4]/3)
#             self.dec4 = ResidualBlock(channels[-5] * 3, channels[-5], scale=1/3)
#         else:
#             self.dec0 = nn.Sequential(
#                 ConvBlock(channels[-1] * 2, channels[-2], apply_norm=True),
#                 ConvBlock(channels[-2], channels[-2], apply_norm=True))
#             self.dec1 = nn.Sequential(
#                 ConvBlock(channels[-2] * 3, channels[-3], apply_norm=True),
#                 ConvBlock(channels[-3], channels[-3], apply_norm=True))
#             self.dec2 = nn.Sequential(
#                 ConvBlock(channels[-3] * 3, channels[-4], apply_norm=True),
#                 ConvBlock(channels[-4], channels[-4], apply_norm=True))
#             self.dec3 = nn.Sequential(
#                 ConvBlock(channels[-4] * 3, channels[-5], apply_norm=True),
#                 ConvBlock(channels[-5], channels[-5], apply_norm=True))
#             self.dec4 = nn.Sequential(
#                 ConvBlock(channels[-5] * 3, channels[-5], apply_norm=True),
#                 ConvBlock(channels[-5], channels[-5], apply_norm=True))
#
#         self.flow0 = nn.Conv3d(channels[-2], c_dim, 3, 1, 1)
#         self.flow0.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow0.weight.shape))
#         self.flow0.bias = nn.Parameter(torch.zeros(self.flow0.bias.shape))
#
#         self.flow1 = nn.Conv3d(channels[-3], c_dim, 3, 1, 1)
#         self.flow1.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow1.weight.shape))
#         self.flow1.bias = nn.Parameter(torch.zeros(self.flow1.bias.shape))
#         self.STN1 = SpatialTransformer((16, 20, 16))
#         self.vecint1 = VecInt((16, 20, 16), nsteps=7, transformer=self.STN1)
#
#         self.flow2 = nn.Conv3d(channels[-4], c_dim, 3, 1, 1)
#         self.flow2.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow2.weight.shape))
#         self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))
#         self.STN2 = SpatialTransformer((32, 40, 32))
#         self.vecint2 = VecInt((32, 40, 32), nsteps=7, transformer=self.STN2)
#
#         self.flow3 = nn.Conv3d(channels[-5], c_dim, 3, 1, 1)
#         self.flow3.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow3.weight.shape))
#         self.flow3.bias = nn.Parameter(torch.zeros(self.flow3.bias.shape))
#         self.STN3 = SpatialTransformer((64, 80, 64))
#         self.vecint3 = VecInt((64, 80, 64), nsteps=7, transformer=self.STN3)
#
#         self.flow4 = nn.Conv3d(channels[-5], c_dim, 3, 1, 1)
#         self.flow4.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow4.weight.shape))
#         self.flow4.bias = nn.Parameter(torch.zeros(self.flow4.bias.shape))
#         self.STN4 = SpatialTransformer((128, 160, 128))
#         self.vecint4 = VecInt((128, 160, 128), nsteps=7, transformer=self.STN4)
#
#     def forward(self, z):
#         z0, z1, z2, z3, z4 = z
#
#         r0 = self.dec0(z0)
#         flow1 = self.resize(self.flow0(r0))
#         flow1 = self.vecint1(flow1)
#         r1 = nn.Upsample(scale_factor=2, mode='trilinear')(r0)
#
#         z1_src, z1_tgt = z1
#         r1 = self.dec1(torch.cat((self.STN1(z1_src, flow1), z1_tgt, r1), dim=1))
#         flow1 = flow1 + self.flow1(r1)
#         flow2 = self.resize(flow1)
#         flow2 = self.vecint2(flow2)
#         r2 = nn.Upsample(scale_factor=2, mode='trilinear')(r1)
#
#         z2_src, z2_tgt = z2
#         r2 = self.dec2(torch.cat((self.STN2(z2_src, flow2), z2_tgt, r2), dim=1))
#         flow2 = flow2 + self.flow2(r2)
#         flow3 = self.resize(flow2)
#         flow3 = self.vecint3(flow3)
#         r3 = nn.Upsample(scale_factor=2, mode='trilinear')(r2)
#
#         z3_src, z3_tgt = z3
#         r3 = self.dec3(torch.cat((self.STN3(z3_src, flow3), z3_tgt, r3), dim=1))
#         flow3 = flow3 + self.flow3(r3)
#         flow4 = self.resize(flow3)
#         flow4 = self.vecint4(flow4)
#         r4 = nn.Upsample(scale_factor=2, mode='trilinear')(r3)
#
#         z4_src, z4_tgt = z4
#         r4 = self.dec4(torch.cat((self.STN4(z4_src, flow4), z4_tgt, r4), dim=1))
#         flow4 = flow4 + self.flow4(r4)
#
#         # flow4 = self.vecint(flow4)
#         return flow4, flow3, flow2, flow1


class RegCascadedDecoder(nn.Module):
    def __init__(self, c_dim=3, channels=(16, 32, 64, 128, 128), res=False):
        super(RegCascadedDecoder, self).__init__()

        self.resize = ResizeTransform(vel_resize=0.5, ndims=3)  # upsample and scale deformation field
        if res:
            self.dec0 = ResidualBlock(channels[-1] * 2, channels[-2], scale=0.5)
            self.dec1 = ResidualBlock(channels[-2] * 2, channels[-3], scale=0.5)
            self.dec2 = ResidualBlock(channels[-3] * 2, channels[-4], scale=0.5)
            self.dec3 = ResidualBlock(channels[-4] * 2, channels[-5], scale=0.5)
            self.dec4 = ResidualBlock(channels[-5] * 2, channels[-5], scale=0.5)
        else:
            self.dec0 = nn.Sequential(
                ConvBlock(channels[-1] * 2, channels[-2], apply_norm=True),
                ConvBlock(channels[-2], channels[-2], apply_norm=True))
            self.dec1 = nn.Sequential(
                ConvBlock(channels[-2] * 2, channels[-3], apply_norm=True),
                ConvBlock(channels[-3], channels[-3], apply_norm=True))
            self.dec2 = nn.Sequential(
                ConvBlock(channels[-3] * 2, channels[-4], apply_norm=True),
                ConvBlock(channels[-4], channels[-4], apply_norm=True))
            self.dec3 = nn.Sequential(
                ConvBlock(channels[-4] * 2, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))
            self.dec4 = nn.Sequential(
                ConvBlock(channels[-5] * 2, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))

        self.flow0 = nn.Conv3d(channels[-2], c_dim, 1, 1)
        # self.flow0.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow0.weight.shape))
        self.flow0.bias = nn.Parameter(torch.zeros(self.flow0.bias.shape))

        self.flow1 = nn.Conv3d(channels[-3], c_dim, 1, 1)
        # self.flow1.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow1.weight.shape))
        self.flow1.bias = nn.Parameter(torch.zeros(self.flow1.bias.shape))
        self.STN1 = SpatialTransformer((16, 20, 16))
        self.vecint1 = VecInt((16, 20, 16), nsteps=7, transformer=self.STN1)

        self.flow2 = nn.Conv3d(channels[-4], c_dim, 1, 1)
        # self.flow2.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow2.weight.shape))
        self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))
        self.STN2 = SpatialTransformer((32, 40, 32))
        self.vecint2 = VecInt((32, 40, 32), nsteps=7, transformer=self.STN2)

        self.flow3 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        # self.flow3.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow3.weight.shape))
        self.flow3.bias = nn.Parameter(torch.zeros(self.flow3.bias.shape))
        self.STN3 = SpatialTransformer((64, 80, 64))
        self.vecint3 = VecInt((64, 80, 64), nsteps=7, transformer=self.STN3)

        self.flow4 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        # self.flow4.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow4.weight.shape))
        self.flow4.bias = nn.Parameter(torch.zeros(self.flow4.bias.shape))
        self.STN4 = SpatialTransformer((128, 160, 128))
        self.vecint4 = VecInt((128, 160, 128), nsteps=7, transformer=self.STN4)

    def forward(self, z):
        z0, z1, z2, z3, z4 = z

        r0 = self.dec0(z0)
        flow1 = self.resize(self.flow0(r0))
        flow1 = self.vecint1(flow1)
        # r1 = nn.Upsample(scale_factor=2, mode='trilinear')(r0)

        z1_src, z1_tgt = z1
        r1 = self.dec1(torch.cat((self.STN1(z1_src, flow1), z1_tgt), dim=1))
        flow1 = flow1 + self.STN1(self.flow1(r1), flow1)  # flow composition
        flow2 = self.resize(flow1)
        flow2 = self.vecint2(flow2)

        z2_src, z2_tgt = z2
        r2 = self.dec2(torch.cat((self.STN2(z2_src, flow2), z2_tgt), dim=1))
        flow2 = flow2 + self.STN2(self.flow2(r2), flow2)  # flow composition
        flow3 = self.resize(flow2)
        flow3 = self.vecint3(flow3)

        z3_src, z3_tgt = z3
        r3 = self.dec3(torch.cat((self.STN3(z3_src, flow3), z3_tgt), dim=1))
        flow3 = flow3 + self.STN3(self.flow3(r3), flow3)  # flow composition
        flow4 = self.resize(flow3)
        flow4 = self.vecint4(flow4)

        z4_src, z4_tgt = z4
        r4 = self.dec4(torch.cat((self.STN4(z4_src, flow4), z4_tgt), dim=1))
        flow4 = flow4 + self.STN4(self.flow4(r4), flow4)  # flow composition

        # flow4 = self.vecint(flow4)
        return flow4, flow3, flow2, flow1

class RegCascadedDecoder_v4(nn.Module):
    #  each level, concatenate synthesis fixed/moving features + flow from previous level
    def __init__(self, c_dim=3, channels=(16, 32, 64, 128, 128), res=False):
        super(RegCascadedDecoder_v4, self).__init__()

        self.resize = ResizeTransform(vel_resize=0.5, ndims=3)  # upsample and scale deformation field
        if res:
            self.dec0 = ResidualBlock(channels[-1] * 2, channels[-2], scale=channels[-2]/(channels[-1] * 2))
            self.dec1 = ResidualBlock(channels[-2] * 2 + 3, channels[-3], scale=channels[-3]/(channels[-2] * 2 + 3))
            self.dec2 = ResidualBlock(channels[-3] * 2 + 3, channels[-4], scale=channels[-4]/(channels[-3] * 2 + 3))
            self.dec3 = ResidualBlock(channels[-4] * 2 + 3, channels[-5], scale=channels[-4]/(channels[-5] * 2 + 3))
            self.dec4 = ResidualBlock(channels[-5] * 2 + 3, channels[-5], scale=channels[-5]/(channels[-5] * 2 + 3))
        else:
            self.dec0 = nn.Sequential(
                ConvBlock(channels[-1] * 2, channels[-2], apply_norm=True),
                ConvBlock(channels[-2], channels[-2], apply_norm=True))
            self.dec1 = nn.Sequential(
                ConvBlock(channels[-2] * 2 + 3, channels[-3], apply_norm=True),
                ConvBlock(channels[-3], channels[-3], apply_norm=True))
            self.dec2 = nn.Sequential(
                ConvBlock(channels[-3] * 2 + 3, channels[-4], apply_norm=True),
                ConvBlock(channels[-4], channels[-4], apply_norm=True))
            self.dec3 = nn.Sequential(
                ConvBlock(channels[-4] * 2 + 3, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))
            self.dec4 = nn.Sequential(
                ConvBlock(channels[-5] * 2 + 3, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))

        self.flow0 = nn.Conv3d(channels[-2], c_dim, 1, 1)
        self.flow0.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow0.weight.shape))
        self.flow0.bias = nn.Parameter(torch.zeros(self.flow0.bias.shape))

        self.flow1 = nn.Conv3d(channels[-3], c_dim, 1, 1)
        self.flow1.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow1.weight.shape))
        self.flow1.bias = nn.Parameter(torch.zeros(self.flow1.bias.shape))
        self.STN1 = SpatialTransformer((16, 20, 16))
        self.vecint1 = VecInt((16, 20, 16), nsteps=7, transformer=self.STN1)

        self.flow2 = nn.Conv3d(channels[-4], c_dim, 1, 1)
        self.flow2.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow2.weight.shape))
        self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))
        self.STN2 = SpatialTransformer((32, 40, 32))
        self.vecint2 = VecInt((32, 40, 32), nsteps=7, transformer=self.STN2)

        self.flow3 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        self.flow3.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow3.weight.shape))
        self.flow3.bias = nn.Parameter(torch.zeros(self.flow3.bias.shape))
        self.STN3 = SpatialTransformer((64, 80, 64))
        self.vecint3 = VecInt((64, 80, 64), nsteps=7, transformer=self.STN3)

        self.flow4 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        self.unc = nn.Conv3d(channels[-5], c_dim*3, 1, 1)
        self.flow4.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow4.weight.shape))
        self.flow4.bias = nn.Parameter(torch.zeros(self.flow4.bias.shape))
        self.STN4 = SpatialTransformer((128, 160, 128))
        self.vecint4 = VecInt((128, 160, 128), nsteps=7, transformer=self.STN4)

    def forward(self, z):
        z0, z1, z2, z3, z4 = z

        r0 = self.dec0(z0)
        flow1 = self.resize(self.flow0(r0))
        flow1 = self.vecint1(flow1)

        z1_src, z1_tgt = z1
        r1 = self.dec1(torch.cat((self.STN1(z1_src, flow1.detach()), z1_tgt, flow1), dim=1))
        flow1 = flow1 + self.STN1(self.flow1(r1), flow1)  # flow composition
        flow2 = self.resize(flow1)
        flow2 = self.vecint2(flow2)

        z2_src, z2_tgt = z2
        r2 = self.dec2(torch.cat((self.STN2(z2_src, flow2.detach()), z2_tgt, flow2), dim=1))
        flow2 = flow2 + self.STN2(self.flow2(r2), flow2)  # flow composition
        flow3 = self.resize(flow2)
        flow3 = self.vecint3(flow3)

        z3_src, z3_tgt = z3
        r3 = self.dec3(torch.cat((self.STN3(z3_src, flow3.detach()), z3_tgt, flow3), dim=1))
        flow3 = flow3 + self.STN3(self.flow3(r3), flow3)  # flow composition
        flow4 = self.resize(flow3)
        flow4 = self.vecint4(flow4)
        
        # print('flow4_input:%.4f'%flow4.mean().item())
        z4_src, z4_tgt = z4
        r4 = self.dec4(torch.cat((self.STN4(z4_src, flow4.detach()), z4_tgt, flow4), dim=1))
        unc_flow = self.flow4(r4)
        flow4 = flow4 + self.STN4(unc_flow, flow4) 
        
        # print('STN4_1:%.4f'%self.STN4(z4_src, flow4.detach()).mean().item(),\
        #       'z4_tgt:%.4f'%z4_tgt.mean().item(),'r4:%.4f'%r4.mean().item(),'flow4 weight:%.4f'%self.flow4.weight.mean().item(),\
        #     'unc_flow:%.4f'% unc_flow.mean().item(),'STN4_2:%.4f'%self.STN4(unc_flow, flow4).mean().item(),\
        #     'flow4_output:%.4f'%flow4.mean().item())
        # flow composition

        # flow4 = self.vecint(flow4)
        return flow4, flow3, flow2, flow1, torch.nn.Softplus()(self.unc(r4))

class RegCascadedDecoder_v5(nn.Module):
    #  each level, concatenate synthesis fixed/moving features + flow from previous level
    def __init__(self, c_dim=3, channels=(16, 32, 64, 128, 128), res=False):
        super(RegCascadedDecoder_v5, self).__init__()

        self.resize = ResizeTransform(vel_resize=0.5, ndims=3)  # upsample and scale deformation field
        if res:
            self.dec0 = ResidualBlock(channels[-1] * 2, channels[-2], scale=channels[-2]/(channels[-1] * 2))
            self.dec1 = ResidualBlock(channels[-2] * 2 + 3, channels[-3], scale=channels[-3]/(channels[-2] * 2 + 3))
            self.dec2 = ResidualBlock(channels[-3] * 2 + 3, channels[-4], scale=channels[-4]/(channels[-3] * 2 + 3))
            self.dec3 = ResidualBlock(channels[-4] * 2 + 3, channels[-5], scale=channels[-4]/(channels[-5] * 2 + 3))
            self.dec4 = ResidualBlock(channels[-5] * 2 + 3, channels[-5], scale=channels[-5]/(channels[-5] * 2 + 3))
        else:
            self.dec0 = nn.Sequential(
                ConvBlock(channels[-1] * 2, channels[-2], apply_norm=True),
                ConvBlock(channels[-2], channels[-2], apply_norm=True))
            self.dec1 = nn.Sequential(
                ConvBlock(channels[-2] * 2 + 3, channels[-3], apply_norm=True),
                ConvBlock(channels[-3], channels[-3], apply_norm=True))
            self.dec2 = nn.Sequential(
                ConvBlock(channels[-3] * 2 + 3, channels[-4], apply_norm=True),
                ConvBlock(channels[-4], channels[-4], apply_norm=True))
            self.dec3 = nn.Sequential(
                ConvBlock(channels[-4] * 2 + 3, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))
            self.dec4 = nn.Sequential(
                ConvBlock(channels[-5] * 2 + 3, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))

        self.flow0 = nn.Conv3d(channels[-2], c_dim, 1, 1)
        self.flow0.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow0.weight.shape))
        self.flow0.bias = nn.Parameter(torch.zeros(self.flow0.bias.shape))

        self.flow1 = nn.Conv3d(channels[-3], c_dim, 1, 1)
        self.flow1.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow1.weight.shape))
        self.flow1.bias = nn.Parameter(torch.zeros(self.flow1.bias.shape))
        self.STN1 = SpatialTransformer((16, 20, 16))
        self.vecint1 = VecInt((16, 20, 16), nsteps=7, transformer=self.STN1)

        self.flow2 = nn.Conv3d(channels[-4], c_dim, 1, 1)
        self.flow2.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow2.weight.shape))
        self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))
        self.STN2 = SpatialTransformer((32, 40, 32))
        self.vecint2 = VecInt((32, 40, 32), nsteps=7, transformer=self.STN2)

        self.flow3 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        self.flow3.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow3.weight.shape))
        self.flow3.bias = nn.Parameter(torch.zeros(self.flow3.bias.shape))
        self.STN3 = SpatialTransformer((64, 80, 64))
        self.vecint3 = VecInt((64, 80, 64), nsteps=7, transformer=self.STN3)

        self.flow4 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        self.flow4.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow4.weight.shape))
        self.flow4.bias = nn.Parameter(torch.zeros(self.flow4.bias.shape))
        self.STN4 = SpatialTransformer((128, 160, 128))
        self.vecint4 = VecInt((128, 160, 128), nsteps=7, transformer=self.STN4)

        self.unc = nn.Conv3d(channels[-5], c_dim, 1, 1)

    def forward(self, z):
        z0, z1, z2, z3, z4 = z

        r0 = self.dec0(z0)
        flow1 = self.resize(self.flow0(r0))
        flow1 = self.vecint1(flow1)

        z1_src, z1_tgt = z1
        r1 = self.dec1(torch.cat((self.STN1(z1_src, flow1.detach()), z1_tgt, flow1), dim=1))
        flow1 = flow1 + self.STN1(self.flow1(r1), flow1)  # flow composition
        flow2 = self.resize(flow1)
        flow2 = self.vecint2(flow2)

        z2_src, z2_tgt = z2
        r2 = self.dec2(torch.cat((self.STN2(z2_src, flow2.detach()), z2_tgt, flow2), dim=1))
        flow2 = flow2 + self.STN2(self.flow2(r2), flow2)  # flow composition
        flow3 = self.resize(flow2)
        flow3 = self.vecint3(flow3)

        z3_src, z3_tgt = z3
        r3 = self.dec3(torch.cat((self.STN3(z3_src, flow3.detach()), z3_tgt, flow3), dim=1))
        flow3 = flow3 + self.STN3(self.flow3(r3), flow3)  # flow composition
        flow4 = self.resize(flow3)
        flow4 = self.vecint4(flow4)

        z4_src, z4_tgt = z4
        r4 = self.dec4(torch.cat((self.STN4(z4_src, flow4.detach()), z4_tgt, flow4), dim=1))
        
        abr = self.unc(r4)
        a = abr[:,0:1,:,:,:]
        b = abr[:,1:2,:,:,:]
        r = abr[:,2:3,:,:,:]
        a = self.STN4(a, flow4)
        b = self.STN4(b, flow4)
        r = self.STN4(r, flow4)
        
        flow4 = flow4 + self.STN4(self.flow4(r4), flow4)  # flow composition
        
        unc_flow = torch.stack([a,b,r],dim = 1)
        # flow4 = self.vecint(flow4)
        return flow4, flow3, flow2, flow1, torch.nn.Softplus()(unc_flow)

class RegCascadedDecoder_v3(nn.Module):
    #  each level, concatenate synthesis fixed/moving features + flow from previous level
    def __init__(self, c_dim=3, channels=(16, 32, 64, 128, 128), res=False):
        super(RegCascadedDecoder_v3, self).__init__()

        self.resize = ResizeTransform(vel_resize=0.5, ndims=3)  # upsample and scale deformation field
        if res:
            self.dec0 = ResidualBlock(channels[-1] * 2, channels[-2], scale=channels[-2]/(channels[-1] * 2))
            self.dec1 = ResidualBlock(channels[-2] * 2 + 3, channels[-3], scale=channels[-3]/(channels[-2] * 2 + 3))
            self.dec2 = ResidualBlock(channels[-3] * 2 + 3, channels[-4], scale=channels[-4]/(channels[-3] * 2 + 3))
            self.dec3 = ResidualBlock(channels[-4] * 2 + 3, channels[-5], scale=channels[-4]/(channels[-5] * 2 + 3))
            self.dec4 = ResidualBlock(channels[-5] * 2 + 3, channels[-5], scale=channels[-5]/(channels[-5] * 2 + 3))
        else:
            self.dec0 = nn.Sequential(
                ConvBlock(channels[-1] * 2, channels[-2], apply_norm=True),
                ConvBlock(channels[-2], channels[-2], apply_norm=True))
            self.dec1 = nn.Sequential(
                ConvBlock(channels[-2] * 2 + 3, channels[-3], apply_norm=True),
                ConvBlock(channels[-3], channels[-3], apply_norm=True))
            self.dec2 = nn.Sequential(
                ConvBlock(channels[-3] * 2 + 3, channels[-4], apply_norm=True),
                ConvBlock(channels[-4], channels[-4], apply_norm=True))
            self.dec3 = nn.Sequential(
                ConvBlock(channels[-4] * 2 + 3, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))
            self.dec4 = nn.Sequential(
                ConvBlock(channels[-5] * 2 + 3, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))

        self.flow0 = nn.Conv3d(channels[-2], c_dim, 1, 1)
        self.flow0.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow0.weight.shape))
        self.flow0.bias = nn.Parameter(torch.zeros(self.flow0.bias.shape))

        self.flow1 = nn.Conv3d(channels[-3], c_dim, 1, 1)
        self.flow1.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow1.weight.shape))
        self.flow1.bias = nn.Parameter(torch.zeros(self.flow1.bias.shape))
        self.STN1 = SpatialTransformer((16, 20, 16))
        self.vecint1 = VecInt((16, 20, 16), nsteps=7, transformer=self.STN1)

        self.flow2 = nn.Conv3d(channels[-4], c_dim, 1, 1)
        self.flow2.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow2.weight.shape))
        self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))
        self.STN2 = SpatialTransformer((32, 40, 32))
        self.vecint2 = VecInt((32, 40, 32), nsteps=7, transformer=self.STN2)

        self.flow3 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        self.flow3.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow3.weight.shape))
        self.flow3.bias = nn.Parameter(torch.zeros(self.flow3.bias.shape))
        self.STN3 = SpatialTransformer((64, 80, 64))
        self.vecint3 = VecInt((64, 80, 64), nsteps=7, transformer=self.STN3)

        self.flow4 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        self.flow4.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow4.weight.shape))
        self.flow4.bias = nn.Parameter(torch.zeros(self.flow4.bias.shape))
        self.STN4 = SpatialTransformer((128, 160, 128))
        self.vecint4 = VecInt((128, 160, 128), nsteps=7, transformer=self.STN4)

        #self.unc = nn.Conv3d(channels[-5], c_dim*3, 1, 1)

    def forward(self, z):
        z0, z1, z2, z3, z4 = z

        r0 = self.dec0(z0)
        flow1 = self.resize(self.flow0(r0))
        flow1 = self.vecint1(flow1)

        z1_src, z1_tgt = z1
        r1 = self.dec1(torch.cat((self.STN1(z1_src, flow1.detach()), z1_tgt, flow1), dim=1))
        flow1 = flow1 + self.STN1(self.flow1(r1), flow1)  # flow composition
        flow2 = self.resize(flow1)
        flow2 = self.vecint2(flow2)

        z2_src, z2_tgt = z2
        r2 = self.dec2(torch.cat((self.STN2(z2_src, flow2.detach()), z2_tgt, flow2), dim=1))
        flow2 = flow2 + self.STN2(self.flow2(r2), flow2)  # flow composition
        flow3 = self.resize(flow2)
        flow3 = self.vecint3(flow3)

        z3_src, z3_tgt = z3
        r3 = self.dec3(torch.cat((self.STN3(z3_src, flow3.detach()), z3_tgt, flow3), dim=1))
        flow3 = flow3 + self.STN3(self.flow3(r3), flow3)  # flow composition
        flow4 = self.resize(flow3)
        flow4 = self.vecint4(flow4)

        z4_src, z4_tgt = z4
        r4 = self.dec4(torch.cat((self.STN4(z4_src, flow4.detach()), z4_tgt, flow4), dim=1))
        flow4 = flow4 + self.STN4(self.flow4(r4), flow4)  # flow composition

        # flow4 = self.vecint(flow4)
        return flow4, flow3, flow2, flow1
   
class RegCascadedDecoder_v6(nn.Module):
    #  each level, concatenate synthesis fixed/moving features + flow from previous level
    def __init__(self, c_dim=3, channels=(16, 32, 64, 128, 128), res=False):
        super(RegCascadedDecoder_v6, self).__init__()

        self.resize = ResizeTransform(vel_resize=0.5, ndims=3)  # upsample and scale deformation field
        if res:
            self.dec0 = ResidualBlock(channels[-1] * 2, channels[-2], scale=channels[-2]/(channels[-1] * 2))
            self.dec1 = ResidualBlock(channels[-2] * 2 + 3, channels[-3], scale=channels[-3]/(channels[-2] * 2 + 3))
            self.dec2 = ResidualBlock(channels[-3] * 2 + 3, channels[-4], scale=channels[-4]/(channels[-3] * 2 + 3))
            self.dec3 = ResidualBlock(channels[-4] * 2 + 3, channels[-5], scale=channels[-4]/(channels[-5] * 2 + 3))
            self.dec4 = ResidualBlock(channels[-5] * 2 + 3, channels[-5], scale=channels[-5]/(channels[-5] * 2 + 3))
        else:
            self.dec0 = nn.Sequential(
                ConvBlock(channels[-1] * 2, channels[-2], apply_norm=True),
                ConvBlock(channels[-2], channels[-2], apply_norm=True))
            self.dec1 = nn.Sequential(
                ConvBlock(channels[-2] * 2 + 3, channels[-3], apply_norm=True),
                ConvBlock(channels[-3], channels[-3], apply_norm=True))
            self.dec2 = nn.Sequential(
                ConvBlock(channels[-3] * 2 + 3, channels[-4], apply_norm=True),
                ConvBlock(channels[-4], channels[-4], apply_norm=True))
            self.dec3 = nn.Sequential(
                ConvBlock(channels[-4] * 2 + 3, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))
            self.dec4 = nn.Sequential(
                ConvBlock(channels[-5] * 2 + 3, channels[-5], apply_norm=True),
                ConvBlock(channels[-5], channels[-5], apply_norm=True))

        self.flow0 = nn.Conv3d(channels[-2], c_dim, 1, 1)
        self.flow0.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow0.weight.shape))
        self.flow0.bias = nn.Parameter(torch.zeros(self.flow0.bias.shape))

        self.flow1 = nn.Conv3d(channels[-3], c_dim, 1, 1)
        self.flow1.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow1.weight.shape))
        self.flow1.bias = nn.Parameter(torch.zeros(self.flow1.bias.shape))
        self.STN1 = SpatialTransformer((16, 20, 16))
        self.vecint1 = VecInt((16, 20, 16), nsteps=7, transformer=self.STN1)

        self.flow2 = nn.Conv3d(channels[-4], c_dim, 1, 1)
        self.flow2.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow2.weight.shape))
        self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))
        self.STN2 = SpatialTransformer((32, 40, 32))
        self.vecint2 = VecInt((32, 40, 32), nsteps=7, transformer=self.STN2)

        self.flow3 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        self.flow3.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow3.weight.shape))
        self.flow3.bias = nn.Parameter(torch.zeros(self.flow3.bias.shape))
        self.STN3 = SpatialTransformer((64, 80, 64))
        self.vecint3 = VecInt((64, 80, 64), nsteps=7, transformer=self.STN3)

        self.flow4 = nn.Conv3d(channels[-5], c_dim, 1, 1)
        self.flow4.weight = nn.Parameter(Normal(0, 1e-3).sample(self.flow4.weight.shape))
        self.flow4.bias = nn.Parameter(torch.zeros(self.flow4.bias.shape))
        self.STN4 = SpatialTransformer((128, 160, 128))
        self.vecint4 = VecInt((128, 160, 128), nsteps=7, transformer=self.STN4)
        self.unc = nn.Conv3d(c_dim, c_dim, 1, 1)

    def forward(self, z):
        z0, z1, z2, z3, z4 = z

        r0 = self.dec0(z0)
        flow1 = self.resize(self.flow0(r0))
        flow1 = self.vecint1(flow1)

        z1_src, z1_tgt = z1
        r1 = self.dec1(torch.cat((self.STN1(z1_src, flow1.detach()), z1_tgt, flow1), dim=1))
        flow1 = flow1 + self.STN1(self.flow1(r1), flow1)  # flow composition
        flow2 = self.resize(flow1)
        flow2 = self.vecint2(flow2)

        z2_src, z2_tgt = z2
        r2 = self.dec2(torch.cat((self.STN2(z2_src, flow2.detach()), z2_tgt, flow2), dim=1))
        flow2 = flow2 + self.STN2(self.flow2(r2), flow2)  # flow composition
        flow3 = self.resize(flow2)
        flow3 = self.vecint3(flow3)

        z3_src, z3_tgt = z3
        r3 = self.dec3(torch.cat((self.STN3(z3_src, flow3.detach()), z3_tgt, flow3), dim=1))
        flow3 = flow3 + self.STN3(self.flow3(r3), flow3)  # flow composition
        flow4 = self.resize(flow3)
        flow4 = self.vecint4(flow4)

        z4_src, z4_tgt = z4
        r4 = self.dec4(torch.cat((self.STN4(z4_src, flow4.detach()), z4_tgt, flow4), dim=1))
        flow4 = flow4 + self.STN4(self.flow4(r4), flow4)  # flow composition

        # flow4 = self.vecint(flow4)
        return flow4, flow3, flow2, flow1, self.unc(flow4)

class UNet(nn.Module):
    """
    A basic U-Net with an encoder and decoder module
    """

    def __init__(self, cdim=1, channels=(16, 32, 64, 128, 128), res=False, gumbel=False):
        """
        res: if True, use residual blocks; otherwise use two convolutions
        gumbel: apply Gumbel Softmax to the latent variable to squeeze into categorical vector
                compress bottleneck information
        """
        super(UNet, self).__init__()

        self.encoder = Encoder(cdim, channels, res=res, gumbel=gumbel)
        self.decoder = SynthDecoder(cdim, channels, res=res)

    def forward(self, x):
        # Encoding
        z = self.encoder(x)

        # Decoding + Skip-Connections
        y, _, _, _, _ = self.decoder(z)
        return y

    def set_required_grad(self, level=-1):
        """
        level=0: freeze the bottleneck layer (in the encoder)
        else: freeze the n low-level layers
        """
        if level == -1:
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
        if level == -2: #freeze entire encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = True
        else:
            for i, layer in enumerate(reversed(list(self.encoder.children()))):
                if i <= level:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True

            for i, layer in enumerate(list(self.decoder.children())):
                if i <= (level-1):
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True


class VoxelMorph(nn.Module):
    """
    A basic VoxelMorph
    """
    def __init__(self, channels=(16, 32, 64, 128, 128), image_size=(128, 160, 128), res=False, gumbel=False):
        """
        res: if True, use residual blocks; otherwise use two convolutions
        gumbel: apply Gumbel Softmax to the latent variable to squeeze into categorical vector
                compress bottleneck information
        """
        super(VoxelMorph, self).__init__()

        self.encoder = Encoder(2, channels, res=res, gumbel=gumbel)
        self.decoder = SynthDecoder(3, channels, res=res)
        self.transformer = SpatialTransformer(image_size)
        self.vectint = VecInt(image_size, 7)

    def forward(self, src, tgt):
        # Encoding
        z = self.encoder(torch.cat((src, tgt), dim=1))
        # Decoding + Skip-Connections
        flow, _, _, _, _ = self.decoder(z)
        flow = self.vectint(flow)
        reg = self.transformer(src, flow)
        return reg, flow


class JSR(nn.Module):
    def __init__(self, cdim=1, channels=(16, 32, 64, 128, 128), image_size=(128, 160, 128),
                 separate_decoders=False, skip_connect=True, res=False, skip_detach=False, gumbel=False, scale=1.0):
        """
        cdim: input image dimension
        channels: number of feature channels at each of the encoding / decoding convolutions
        image_size:
        separate_decoders: if true, no weight-sharing between MR and CBCT synthesis decoders;
                           if false, use weight-sharing
        skip_connect: if True, add skip connection from synthesis decoding branch to registration decoding branch
        """
        super(JSR, self).__init__()

        self.encoder_src = Encoder(cdim, channels, res=res, gumbel=gumbel, scale=scale)
        self.encoder_tgt = Encoder(cdim, channels, res=res, gumbel=gumbel, scale=scale)
        self.separate_decoders = separate_decoders
        self.skip_connect = skip_connect
        self.skip_detach = skip_detach
        if separate_decoders:
            self.decoder_synth_src = SynthDecoder(cdim, channels, res=res)
            self.decoder_synth_tgt = SynthDecoder(cdim, channels, res=res)
        else:
            self.decoder_synth = SynthDecoder(cdim, channels, res=res)
        self.decoder_reg = RegDecoder(3, channels, skip_connect=skip_connect, res=res)
        self.transformer = SpatialTransformer(image_size)
        self.vectint = VecInt(image_size, 7)

    def forward(self, x, y):
        z_src = self.encoder_src(x)
        z_tgt = self.encoder_tgt(y)

        # Synthesis decoding branch
        if self.separate_decoders:
            src_synth, z_src4, z_src3, z_src2, z_src1 = self.decoder_synth_src(z_src)
            tgt_synth, z_tgt4, z_tgt3, z_tgt2, z_tgt1 = self.decoder_synth_tgt(z_tgt)
        else:
            src_synth, z_src4, z_src3, z_src2, z_src1 = self.decoder_synth(z_src)
            tgt_synth, z_tgt4, z_tgt3, z_tgt2, z_tgt1 = self.decoder_synth(z_tgt)

        # Registration decoding branch
        if self.skip_connect:
            if self.skip_detach:
                flow = self.decoder_reg([torch.cat((z_src[0], z_tgt[0]), dim=1),
                                         torch.cat((z_src1, z_tgt1), dim=1).detach(),
                                         torch.cat((z_src2, z_tgt2), dim=1).detach(),
                                         torch.cat((z_src3, z_tgt3), dim=1).detach(),
                                         torch.cat((z_src4, z_tgt4), dim=1).detach()])
            else:
                flow = self.decoder_reg([torch.cat((z_src[0], z_tgt[0]), dim=1),
                                         torch.cat((z_src1, z_tgt1), dim=1),
                                         torch.cat((z_src2, z_tgt2), dim=1),
                                         torch.cat((z_src3, z_tgt3), dim=1),
                                         torch.cat((z_src4, z_tgt4), dim=1)])
        else:
            flow = self.decoder_reg(torch.cat((z_src[0], z_tgt[0]), dim=1))

        return flow, src_synth, tgt_synth

    def warp_image(self, img, flow):
        return self.transformer(img, flow)

    def exp(self, flow):
        return self.vectint(flow)

    def set_required_grad(self, mode='synth'):
        """
        mode = 'synth': encoder_src + encoder_tgt + decoder_synth to true
        mode = 'reg': all to true
        mode = 'false': all to false
        """
        if mode == 'synth':
            for param in self.encoder_src.parameters():
                param.requires_grad = True
            for param in self.encoder_tgt.parameters():
                param.requires_grad = True
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = True
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = True
            for param in self.decoder_reg.parameters():
                param.requires_grad = False
        elif mode == 'reg':
            for param in self.encoder_src.parameters():
                param.requires_grad = True
            for param in self.encoder_tgt.parameters():
                param.requires_grad = True
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = False
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = False
            for param in self.decoder_reg.parameters():
                param.requires_grad = True
        elif mode == 'joint':
            for param in self.encoder_src.parameters():
                param.requires_grad = True
            for param in self.encoder_tgt.parameters():
                param.requires_grad = True
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = True
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = True
            for param in self.decoder_reg.parameters():
                param.requires_grad = True
        elif mode == 'false':
            for param in self.encoder_src.parameters():
                param.requires_grad = False
            for param in self.encoder_tgt.parameters():
                param.requires_grad = False
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = False
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = False
            for param in self.decoder_reg.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError


class JSRCascade(nn.Module):
    def __init__(self, cdim=1, channels=(16, 32, 64, 128, 128), image_size=(128, 160, 128),
                 separate_decoders=False, res=False, version='v1'):
        """
        cdim: input image dimension
        channels: number of feature channels at each of the encoding / decoding convolutions
        image_size:
        separate_decoders: if true, no weight-sharing between MR and CBCT synthesis decoders;
                           if false, use weight-sharing
        skip_connect: if True, add skip connection from synthesis decoding branch to registration decoding branch
        """
        super(JSRCascade, self).__init__()

        self.encoder_src = Encoder(cdim, channels, res=res)
        self.encoder_tgt = Encoder(cdim, channels, res=res)
        self.separate_decoders = separate_decoders
        if separate_decoders:
            self.decoder_synth_src = SynthDecoder(cdim, channels, res=res)
            self.decoder_synth_tgt = SynthDecoder(cdim, channels, res=res)
        else:
            self.decoder_synth = SynthDecoder(cdim, channels, res=res)
        if version == 'v1':
            self.decoder_reg = RegCascadedDecoder(3, channels, res=res)
        elif version == 'v2':
            self.decoder_reg = RegCascadedDecoder_v2(3, channels, res=res)
        elif version == 'v3':
            self.decoder_reg = RegCascadedDecoder_v3(3, channels, res=res)
        elif version == 'v4':
            self.decoder_reg = RegCascadedDecoder_v4(3, channels, res=res)
        elif version == 'v5':
            self.decoder_reg = RegCascadedDecoder_v5(3, channels, res=res)
        elif version == 'v6':
            self.decoder_reg = RegCascadedDecoder_v6(3, channels, res=res)
        else:
            raise NotImplementedError("Have not implemented the specified registration decoder!")
        self.transformer = SpatialTransformer(image_size)
        self.vectint = VecInt(image_size, 7)

    def forward(self, x, y):
        z_src = self.encoder_src(x)
        z_tgt = self.encoder_tgt(y)
        # print(z_src[0].shape,z_src[1].shape,z_src[2].shape,z_src[3].shape,z_src[4].shape)
        # Synthesis decoding branch
        if self.separate_decoders:
            src_synth, z_src4, z_src3, z_src2, z_src1 = self.decoder_synth_src(z_src)
            tgt_synth, z_tgt4, z_tgt3, z_tgt2, z_tgt1 = self.decoder_synth_tgt(z_tgt)
        else:
            src_synth, z_src4, z_src3, z_src2, z_src1 = self.decoder_synth(z_src)
            tgt_synth, z_tgt4, z_tgt3, z_tgt2, z_tgt1 = self.decoder_synth(z_tgt)
        
        # Registration decoding branch [128x160, 64x80, 32x40, 16x20]
        flows = self.decoder_reg([torch.cat((z_src[0], z_tgt[0]), dim=1),
                                  [z_src1, z_tgt1],
                                  [z_src2, z_tgt2],
                                  [z_src3, z_tgt3],
                                  [z_src4, z_tgt4]])
        # print(z_src1.shape,z_src2.shape,z_src3.shape,z_src4.shape)

        # Downsample the synthetic images [128x160, 64x80, 32x40, 16x20]
        src_synths = [src_synth]
        tgt_synths = [tgt_synth]
        for i in range(3):
            src_synths.append(nn.AvgPool3d(2)(src_synths[-1]))
            tgt_synths.append(nn.AvgPool3d(2)(tgt_synths[-1]))
        # print(src_synths[0].shape,src_synths[1].shape,src_synths[2].shape,src_synths[3].shape)
        return flows, src_synths, tgt_synths
    def warp_image(self, img, flow):
        return self.transformer(img, flow)

    def exp(self, flow):
        return self.vectint(flow)

    def set_required_grad(self, mode='synth'):
        """
        mode = 'synth': encoder_src + encoder_tgt + decoder_synth to true
        mode = 'reg': all to true
        mode = 'false': all to false
        """
        if mode == 'synth':
            for param in self.encoder_src.parameters():
                param.requires_grad = True
            for param in self.encoder_tgt.parameters():
                param.requires_grad = True
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = True
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = True
            for param in self.decoder_reg.parameters():
                param.requires_grad = False
        elif mode == 'reg':
            for param in self.encoder_src.parameters():
                param.requires_grad = False
            for param in self.encoder_tgt.parameters():
                param.requires_grad = False
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = False
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = False
            for param in self.decoder_reg.parameters():
                param.requires_grad = True
        elif mode == 'joint':
            for param in self.encoder_src.parameters():
                param.requires_grad = True
            for param in self.encoder_tgt.parameters():
                param.requires_grad = True
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = True
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = True
            for param in self.decoder_reg.parameters():
                param.requires_grad = True
        elif mode == 'false':
            for param in self.encoder_src.parameters():
                param.requires_grad = False
            for param in self.encoder_tgt.parameters():
                param.requires_grad = False
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = False
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = False
            for param in self.decoder_reg.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError

class JSRCascade_v2(nn.Module):
    def __init__(self, cdim=1, channels=(16, 32, 64, 128, 128), image_size=(128, 160, 128),
                 separate_decoders=False, res=False, version='v1'):
        """
        cdim: input image dimension
        channels: number of feature channels at each of the encoding / decoding convolutions
        image_size:
        separate_decoders: if true, no weight-sharing between MR and CBCT synthesis decoders;
                           if false, use weight-sharing
        skip_connect: if True, add skip connection from synthesis decoding branch to registration decoding branch
        """
        super(JSRCascade_v2, self).__init__()

        self.encoder_src = Encoder(cdim, channels, res=res)
        self.encoder_tgt = Encoder(cdim, channels, res=res)
        self.intraCBCT = ConvBlock(cdim, cdim)
        self.intraMR = ConvBlock(cdim, cdim)
        self.separate_decoders = separate_decoders
        if separate_decoders:
            self.decoder_synth_src = SynthDecoder(cdim, channels, res=res)
            self.decoder_synth_tgt = SynthDecoder(cdim, channels, res=res)
        else:
            self.decoder_synth = SynthDecoder(cdim, channels, res=res)
        if version == 'v1':
            self.decoder_reg = RegCascadedDecoder(3, channels, res=res)
        elif version == 'v2':
            self.decoder_reg = RegCascadedDecoder_v2(3, channels, res=res)
        elif version == 'v3':
            self.decoder_reg = RegCascadedDecoder_v3(3, channels, res=res)
        elif version == 'v4':
            self.decoder_reg = RegCascadedDecoder_v4(3, channels, res=res)
        elif version == 'v5':
            self.decoder_reg = RegCascadedDecoder_v5(3, channels, res=res)
        elif version == 'v6':
            self.decoder_reg = RegCascadedDecoder_v6(3, channels, res=res)
        else:
            raise NotImplementedError("Have not implemented the specified registration decoder!")
        self.transformer = SpatialTransformer(image_size)
        self.vectint = VecInt(image_size, 7)

    def forward(self, x, y):
        z_src = self.encoder_src(x)
        z_tgt = self.encoder_tgt(y)

        # Synthesis decoding branch
        if self.separate_decoders:
            src_synth, z_src4, z_src3, z_src2, z_src1 = self.decoder_synth_src(z_src)
            tgt_synth, z_tgt4, z_tgt3, z_tgt2, z_tgt1 = self.decoder_synth_tgt(z_tgt)
        else:
            src_synth, z_src4, z_src3, z_src2, z_src1 = self.decoder_synth(z_src)
            tgt_synth, z_tgt4, z_tgt3, z_tgt2, z_tgt1 = self.decoder_synth(z_tgt)

        # Registration decoding branch [128x160, 64x80, 32x40, 16x20]
        flows = self.decoder_reg([torch.cat((z_src[0], z_tgt[0]), dim=1),
                                  [z_src1, z_tgt1],
                                  [z_src2, z_tgt2],
                                  [z_src3, z_tgt3],
                                  [z_src4, z_tgt4]])
        unc_CBCT = self.intraCBCT(src_synth)
        unc_MR = self.intraMR(tgt_synth)
        # Downsample the synthetic images [128x160, 64x80, 32x40, 16x20]
        src_synths = [src_synth]
        tgt_synths = [tgt_synth]
        for i in range(3):
            src_synths.append(nn.AvgPool3d(2)(src_synths[-1]))
            tgt_synths.append(nn.AvgPool3d(2)(tgt_synths[-1]))

        return flows, src_synths, tgt_synths, unc_CBCT, unc_MR
    def warp_image(self, img, flow):
        return self.transformer(img, flow)

    def exp(self, flow):
        return self.vectint(flow)

    def set_required_grad(self, mode='synth'):
        """
        mode = 'synth': encoder_src + encoder_tgt + decoder_synth to true
        mode = 'reg': all to true
        mode = 'false': all to false
        """
        if mode == 'synth':
            for param in self.encoder_src.parameters():
                param.requires_grad = True
            for param in self.encoder_tgt.parameters():
                param.requires_grad = True
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = True
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = True
            for param in self.decoder_reg.parameters():
                param.requires_grad = False
        elif mode == 'reg':
            for param in self.encoder_src.parameters():
                param.requires_grad = False
            for param in self.encoder_tgt.parameters():
                param.requires_grad = False
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = False
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = False
            for param in self.decoder_reg.parameters():
                param.requires_grad = True
        elif mode == 'joint':
            for param in self.encoder_src.parameters():
                param.requires_grad = True
            for param in self.encoder_tgt.parameters():
                param.requires_grad = True
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = True
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = True
            for param in self.decoder_reg.parameters():
                param.requires_grad = True
        elif mode == 'false':
            for param in self.encoder_src.parameters():
                param.requires_grad = False
            for param in self.encoder_tgt.parameters():
                param.requires_grad = False
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = False
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = False
            for param in self.decoder_reg.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError

class JSRCascade_v3(nn.Module):
    def __init__(self, cdim=1, channels=(16, 32, 64, 128, 128), image_size=(128, 160, 128),
                 separate_decoders=False, res=False, version='v1'):
        """
        cdim: input image dimension
        channels: number of feature channels at each of the encoding / decoding convolutions
        image_size:
        separate_decoders: if true, no weight-sharing between MR and CBCT synthesis decoders;
                           if false, use weight-sharing
        skip_connect: if True, add skip connection from synthesis decoding branch to registration decoding branch
        """
        super(JSRCascade_v3, self).__init__()

        self.encoder_src = Encoder(cdim, channels, res=res)
        self.encoder_tgt = Encoder(cdim, channels, res=res)
        self.separate_decoders = separate_decoders
        if separate_decoders:
            self.decoder_synth_src = SynthDecoder_v2(cdim, channels, res=res)
            self.decoder_synth_tgt = SynthDecoder_v2(cdim, channels, res=res)
        else:
            self.decoder_synth = SynthDecoder(cdim, channels, res=res)
        if version == 'v1':
            self.decoder_reg = RegCascadedDecoder(3, channels, res=res)
        elif version == 'v2':
            self.decoder_reg = RegCascadedDecoder_v2(3, channels, res=res)
        elif version == 'v3':
            self.decoder_reg = RegCascadedDecoder_v3(3, channels, res=res)
        elif version == 'v4':
            self.decoder_reg = RegCascadedDecoder_v4(3, channels, res=res)
        elif version == 'v5':
            self.decoder_reg = RegCascadedDecoder_v5(3, channels, res=res)
        elif version == 'v6':
            self.decoder_reg = RegCascadedDecoder_v6(3, channels, res=res)
        else:
            raise NotImplementedError("Have not implemented the specified registration decoder!")
        self.transformer = SpatialTransformer(image_size)
        self.vectint = VecInt(image_size, 7)

    def forward(self, x, y):
        z_src = self.encoder_src(x)
        z_tgt = self.encoder_tgt(y)

        # Synthesis decoding branch
        if self.separate_decoders:
            src_synth, z_src4, z_src3, z_src2, z_src1 = self.decoder_synth_src(z_src)
            tgt_synth, z_tgt4, z_tgt3, z_tgt2, z_tgt1 = self.decoder_synth_tgt(z_tgt)
        else:
            src_synth, z_src4, z_src3, z_src2, z_src1 = self.decoder_synth(z_src)
            tgt_synth, z_tgt4, z_tgt3, z_tgt2, z_tgt1 = self.decoder_synth(z_tgt)

        # Registration decoding branch [128x160, 64x80, 32x40, 16x20]
        flows = self.decoder_reg([torch.cat((z_src[0], z_tgt[0]), dim=1),
                                  [z_src1, z_tgt1],
                                  [z_src2, z_tgt2],
                                  [z_src3, z_tgt3],
                                  [z_src4, z_tgt4]])
        # Downsample the synthetic images [128x160, 64x80, 32x40, 16x20]
        src_synths = [src_synth]
        tgt_synths = [tgt_synth]
        for i in range(3):
            src_synths.append(nn.AvgPool3d(2)(src_synths[-1]))
            tgt_synths.append(nn.AvgPool3d(2)(tgt_synths[-1]))

        return flows, src_synths, tgt_synths
    def warp_image(self, img, flow):
        return self.transformer(img, flow)

    def exp(self, flow):
        return self.vectint(flow)

    def set_required_grad(self, mode='synth'):
        """
        mode = 'synth': encoder_src + encoder_tgt + decoder_synth to true
        mode = 'reg': all to true
        mode = 'false': all to false
        """
        if mode == 'synth':
            for param in self.encoder_src.parameters():
                param.requires_grad = True
            for param in self.encoder_tgt.parameters():
                param.requires_grad = True
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = True
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = True
            for param in self.decoder_reg.parameters():
                param.requires_grad = False
        elif mode == 'reg':
            for param in self.encoder_src.parameters():
                param.requires_grad = False
            for param in self.encoder_tgt.parameters():
                param.requires_grad = False
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = False
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = False
            for param in self.decoder_reg.parameters():
                param.requires_grad = True
        elif mode == 'joint':
            for param in self.encoder_src.parameters():
                param.requires_grad = True
            for param in self.encoder_tgt.parameters():
                param.requires_grad = True
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = True
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = True
            for param in self.decoder_reg.parameters():
                param.requires_grad = True
        elif mode == 'false':
            for param in self.encoder_src.parameters():
                param.requires_grad = False
            for param in self.encoder_tgt.parameters():
                param.requires_grad = False
            if self.separate_decoders:
                for param in self.decoder_synth_src.parameters():
                    param.requires_grad = False
                for param in self.decoder_synth_tgt.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_synth.parameters():
                    param.requires_grad = False
            for param in self.decoder_reg.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError

 
class Discriminator(nn.Module):
    def __init__(self, cdim=1, num_layers=3, num_channels=16, kernel_size=4,
                 apply_norm=True, norm='IN', skip_connect=True):
        super(Discriminator, self).__init__()

        nf = num_channels
        self.skip_connect = skip_connect
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv_0',
                              nn.Conv3d(cdim, nf, kernel_size=kernel_size, stride=2,
                                        padding=kernel_size // 2, bias=False))
        self.conv0.add_module('relu_0', nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv_1', nn.Conv3d(nf, nf * 2, kernel_size=kernel_size, stride=2,
                                                  padding=kernel_size // 2, bias=False))
        if apply_norm:
            if norm == 'IN':
                self.conv1.add_module(f'norm_1', nn.InstanceNorm3d(nf * 2))
            elif norm == 'BN':
                self.conv1.add_module(f'norm_1', nn.BatchNorm3d(nf * 2))
        self.conv1.add_module('relu_1', nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv_2', nn.Conv3d(nf * 2, nf * 4, kernel_size=kernel_size, stride=2,
                                                  padding=kernel_size // 2, bias=False))
        if apply_norm:
            if norm == 'IN':
                self.conv2.add_module(f'norm_2', nn.InstanceNorm3d(nf * 4))
            elif norm == 'BN':
                self.conv2.add_module(f'norm_2', nn.BatchNorm3d(nf * 4))
        self.conv2.add_module('relu_2', nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv_3', nn.Conv3d(nf * 4, nf * 8, kernel_size=kernel_size, stride=2,
                                                  padding=kernel_size // 2, bias=False))
        if apply_norm:
            if norm == 'IN':
                self.conv3.add_module(f'norm_3', nn.InstanceNorm3d(nf * 8))
            elif norm == 'BN':
                self.conv3.add_module(f'norm_3', nn.BatchNorm3d(nf * 8))
        self.conv3.add_module('relu_3', nn.LeakyReLU(0.2))

        self.conv_final1 = nn.Conv3d(nf * 2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv_final2 = nn.Conv3d(nf * 4, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv_final3 = nn.Conv3d(nf * 8, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        if self.skip_connect:
            return [torch.cat((self.conv_final3(x3),
                              self.conv_final2(nn.AvgPool3d(2)(x2)),
                              self.conv_final1(nn.AvgPool3d(4)(x1))), dim=1)]
        else:
            return [x3]


class MultiResDiscriminator(nn.Module):
    """
    Multiscale Patch Discriminator
    """

    def __init__(self, cdim=1, num_channels=16, norm='BN'):
        """
        cdim: input image dimension, 1 for single image, 2 for conditional GAN
        num_channels: base number of channels
        scales: number of resolutions to discriminate
        levels: for each scale, the number of strided convolutions
        """
        super(MultiResDiscriminator, self).__init__()

        self.conv = ConvBlock(cdim, num_channels, kernel=3, stride=1, padding=1, apply_act=True, apply_norm=False)

        self.stack0 = nn.Sequential()
        self.stack0.add_module(f'down_0_0', Downsample(num_channels, num_channels * 2, kernel=3, norm=norm))
        self.stack0.add_module(f'down_0_1', Downsample(num_channels * 2, num_channels * 4, kernel=3, norm=norm))
        self.stack0.add_module(f'down_0_2', Downsample(num_channels * 4, num_channels * 8, kernel=3, norm=norm))
        self.stack0.add_module(f'down_0_3', ConvBlock(num_channels * 8, 1, kernel=3, padding=2,
                                                      apply_act=False, apply_norm=False))

        self.stack1 = nn.Sequential()
        self.stack1.add_module('average_pool_1', nn.AvgPool3d(2))
        self.stack1.add_module(f'down_1_0', Downsample(num_channels, num_channels * 2, kernel=3, norm=norm))
        self.stack1.add_module(f'down_1_1', Downsample(num_channels * 2, num_channels * 4, kernel=3, norm=norm))
        self.stack1.add_module(f'down_1_2', Downsample(num_channels * 4, num_channels * 8, kernel=3, norm=norm))
        self.stack1.add_module(f'down_1_3', ConvBlock(num_channels * 8, 1, kernel=1, padding=0,
                                                      apply_act=False, apply_norm=False))

        self.stack2 = nn.Sequential()
        self.stack2.add_module('average_pool_2', nn.AvgPool3d(4))
        self.stack2.add_module(f'down_2_0', Downsample(num_channels, num_channels * 2, kernel=3, norm=norm))
        self.stack2.add_module(f'down_2_1', Downsample(num_channels * 2, num_channels * 4, kernel=3, norm=norm))
        self.stack2.add_module(f'down_2_2', Downsample(num_channels * 4, num_channels * 8, kernel=3, norm=norm))
        self.stack2.add_module(f'down_2_3', ConvBlock(num_channels * 8, 1, kernel=1, padding=0,
                                                      apply_act=False, apply_norm=False))

    def forward(self, x):
        x = self.conv(x)
        # logits = []
        # logits.append(self.stack0(x))
        # logits.append(self.stack1(x))
        # logits.append(self.stack2(x))

        return [self.stack0(x), self.stack1(x), self.stack2(x)]

if __name__ == '__main__':
    fake_mr = torch.rand([1,1,128,160,128])
    fake_cbct = torch.rand([1,1,128,160,128])
    flow, ct1,ct2 = JSRCascade(cdim=1, channels=(16, 32, 64, 128, 128), image_size=(128, 160, 128),
                   separate_decoders=True, res=True, version='v5')(fake_cbct,fake_mr)
    print(flow[0].shape,flow[-1].shape)