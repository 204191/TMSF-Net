import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18BackboneOS16(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 2, 1, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2, 1)
        self.layer3 = self._make_layer(128, 256, 2, 2, 1)
        self.layer4 = self._make_layer(256, 512, 2, 1, 2)

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride, dilation):
        layers = [BasicBlock(in_channels, out_channels, stride, dilation)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)
        return low_level, high_level


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.act(self.bn(self.conv(self.pool(x))))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels=512, out_channels=256, dilations=(6, 12, 18), dropout=0.0):
        super().__init__()
        branches = [conv_bn_act(in_channels, out_channels, 1)]
        branches.extend(ASPPConv(in_channels, out_channels, d) for d in dilations)
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)

        layers = [
            nn.Conv2d(out_channels * len(branches), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.project = nn.Sequential(*layers)

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, high_channels, out_channels, low_level_project_channels=48, decoder_channels=256):
        super().__init__()
        self.low_project = conv_bn_act(64, low_level_project_channels, 1)
        self.decoder = nn.Sequential(
            conv_bn_act(high_channels + low_level_project_channels, decoder_channels, 3, padding=1),
            conv_bn_act(decoder_channels, decoder_channels, 3, padding=1),
        )
        self.classifier = nn.Conv2d(decoder_channels, out_channels, kernel_size=1)

    def forward(self, fused, low_level, output_size):
        low = self.low_project(low_level)
        fused = F.interpolate(fused, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([fused, low], dim=1))
        x = self.classifier(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class LSTMDeepLabV3Plus(nn.Module):
    def __init__(
        self,
        input_time_steps,
        flood_channels,
        static_channels,
        dynamic_channels,
        output_time_steps,
        aspp_out_channels=256,
        aspp_dilations=(6, 12, 18),
        lstm_hidden_size=128,
        lstm_num_layers=2,
        lstm_dropout=0.0,
        dynamic_fusion_channels=32,
        decoder_channels=256,
        low_level_project_channels=48,
        dropout=0.0,
        dynamic_dropout=0.0,
        leaky_relu_slope=0.01,
    ):
        super().__init__()

        spatial_in_channels = input_time_steps * flood_channels + static_channels

        self.backbone = ResNet18BackboneOS16(spatial_in_channels)
        self.aspp = ASPP(512, aspp_out_channels, aspp_dilations, dropout)
        self.lstm = nn.LSTM(
            input_size=dynamic_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )
        self.dynamic_proj = nn.Sequential(
            nn.Linear(lstm_hidden_size, dynamic_fusion_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(dynamic_dropout),
        )
        self.fusion_conv = conv_bn_act(
            aspp_out_channels + dynamic_fusion_channels,
            aspp_out_channels,
            1,
        )
        self.decoder = DeepLabV3PlusDecoder(
            aspp_out_channels,
            output_time_steps,
            low_level_project_channels,
            decoder_channels,
        )

    def forward(self, flood, static, dynamic):
        b, t, c, h, w = flood.shape
        flood = flood.reshape(b, t * c, h, w)
        spatial_x = torch.cat([flood, static], dim=1)

        low_level, high_level = self.backbone(spatial_x)
        spatial_feat = self.aspp(high_level)

        temporal_feat, _ = self.lstm(dynamic)
        temporal_feat = self.dynamic_proj(temporal_feat[:, -1])
        temporal_feat = temporal_feat[:, :, None, None].expand(
            -1, -1, spatial_feat.shape[-2], spatial_feat.shape[-1]
        )

        fused = self.fusion_conv(torch.cat([spatial_feat, temporal_feat], dim=1))
        return self.decoder(fused, low_level, (h, w))
