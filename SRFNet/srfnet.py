import torch
import torch.nn as nn


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, gn_groups=8):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(gn_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, channels, gn_groups=8):
        super().__init__()
        self.refine = ConvGNAct(channels, channels, gn_groups=gn_groups)
        self.down = ConvGNAct(channels, channels, stride=2, gn_groups=gn_groups)

    def forward(self, x):
        return self.down(self.refine(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size, gn_groups=8):
        super().__init__()
        self.depthwise = ConvGNAct(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            gn_groups=gn_groups,
        )
        self.pointwise = ConvGNAct(
            channels,
            channels,
            kernel_size=1,
            padding=0,
            gn_groups=gn_groups,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SRFBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_1=5,
        kernel_2=7,
        mlp_ratio=2.0,
        gn_groups=8,
        input_residual=True,
        input_residual_init=0.1,
    ):
        super().__init__()
        hidden_channels = max(channels, int(round(channels * mlp_ratio)))
        self.input_residual = input_residual

        self.feature_1 = ConvGNAct(channels, channels, gn_groups=gn_groups)
        self.feature_2 = ConvGNAct(channels, channels, gn_groups=gn_groups)
        self.branch_1 = DepthwiseSeparableConv(channels, kernel_1, gn_groups)
        self.branch_2 = DepthwiseSeparableConv(channels, kernel_2, gn_groups)

        self.attention = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

        self.fuse = ConvGNAct(channels, channels, gn_groups=gn_groups)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

        self.gamma = nn.Parameter(torch.tensor(float(input_residual_init)))

    def forward(self, x):
        f1 = self.feature_1(x)
        f2 = self.feature_2(f1)

        b1 = self.branch_1(f1)
        b2 = self.branch_2(f2)
        cat = torch.cat([b1, b2], dim=1)

        avg = torch.mean(cat, dim=1, keepdim=True)
        maxv = torch.amax(cat, dim=1, keepdim=True)
        att = self.attention(torch.cat([avg, maxv], dim=1))

        out = b1 * att[:, 0:1] + b2 * att[:, 1:2]
        out = self.fuse(out)
        out = self.channel_mlp(out)

        if self.input_residual:
            out = out + self.gamma * x
        return out


class DecoderBlock(nn.Module):
    def __init__(self, channels, gn_groups=8):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.GroupNorm(gn_groups, channels),
            nn.SiLU(inplace=True),
        )
        self.refine = ConvGNAct(channels, channels, gn_groups=gn_groups)

    def forward(self, x):
        return self.refine(self.up(x))


class SRFNetFDS(nn.Module):
    def __init__(
        self,
        flood_in_channels,
        dynamic_in_channels,
        static_in_channels,
        out_channels,
        flood_embed_channels=32,
        dynamic_embed_channels=32,
        static_embed_channels=32,
        encoder_blocks=2,
        num_srf_blocks=8,
        decoder_blocks=2,
        gn_groups=8,
        srf_kernel_1=5,
        srf_kernel_2=7,
        mlp_ratio=2.0,
        srf_input_residual=True,
        srf_input_residual_init=0.1,
    ):
        super().__init__()

        base_channels = flood_embed_channels + dynamic_embed_channels + static_embed_channels

        self.flood_stem = ConvGNAct(
            flood_in_channels, flood_embed_channels, gn_groups=gn_groups
        )
        self.dynamic_stem = ConvGNAct(
            dynamic_in_channels, dynamic_embed_channels, gn_groups=gn_groups
        )
        self.static_stem = ConvGNAct(
            static_in_channels, static_embed_channels, gn_groups=gn_groups
        )

        self.encoder = nn.Sequential(
            *[EncoderBlock(base_channels, gn_groups) for _ in range(encoder_blocks)]
        )

        self.srf_blocks = nn.Sequential(
            *[
                SRFBlock(
                    base_channels,
                    kernel_1=srf_kernel_1,
                    kernel_2=srf_kernel_2,
                    mlp_ratio=mlp_ratio,
                    gn_groups=gn_groups,
                    input_residual=srf_input_residual,
                    input_residual_init=srf_input_residual_init,
                )
                for _ in range(num_srf_blocks)
            ]
        )

        self.decoder = nn.Sequential(
            *[DecoderBlock(base_channels, gn_groups) for _ in range(decoder_blocks)]
        )

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, flood, dynamic, static):
        f = self.flood_stem(flood)
        d = self.dynamic_stem(dynamic)
        s = self.static_stem(static)

        x = torch.cat([f, d, s], dim=1)
        x = self.encoder(x)
        x = self.srf_blocks(x)
        x = self.decoder(x)

        return self.out_conv(x)
