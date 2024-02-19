import argparse
import torch
import torch.nn as nn
from torchsummary import summary


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        model.weight.data.normal_(0.0, 0.03)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0.0, 0.03)


#  SE Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels * r),
            Swish(),
            nn.Linear(in_channels * r, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1)
        return x


class MBConv(nn.Module):
    expand = 6

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5
    ):
        super().__init__()
        # first MBConv is not using stochastic depth
        self.p = (
            torch.tensor(p).float()
            if (in_channels == out_channels)
            else torch.tensor(1).float()
        )

        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels * MBConv.expand,
                1,
                stride=stride,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
            nn.Conv1d(
                in_channels * MBConv.expand,
                in_channels * MBConv.expand,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
                groups=in_channels * MBConv.expand,
            ),
            nn.BatchNorm1d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
        )
        self.se = SEBlock(in_channels * MBConv.expand, se_scale)
        self.project = nn.Sequential(
            nn.Conv1d(
                in_channels * MBConv.expand,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3),
        )
        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x

        return x


class SepConv(nn.Module):
    expand = 1

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5
    ):
        super().__init__()
        # first SepConv is not using stochastic depth
        self.p = (
            torch.tensor(p).float()
            if (in_channels == out_channels)
            else torch.tensor(1).float()
        )

        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels * SepConv.expand,
                in_channels * SepConv.expand,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
                groups=in_channels * SepConv.expand,
            ),
            nn.BatchNorm1d(in_channels * SepConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
        )
        self.se = SEBlock(in_channels * SepConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv1d(
                in_channels * SepConv.expand,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3),
        )
        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x

        return x


class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        width_coef=1.0,
        depth_coef=1.0,
        scale=1.0,
        dropout=0.2,
        se_scale=4,
        stochastic_depth=False,
        p=0.5,
        strides=[1, 2, 2, 2, 1, 2, 1],
        stage1_stride=2,
    ):
        super().__init__()
        # channels = [64, 32, 24, 40, 80, 112, 192, 320, 1280]
        # repeats = [1, 2, 2, 3, 3, 4, 1]
        # kernel_size = [3, 3, 5, 3, 5, 5, 3]
        channels = [256, 256, 256, 512, 1024]
        repeats = [1, 3, 2, 1]
        kernel_size = [3, 3, 3, 3]
        depth = depth_coef
        width = width_coef

        channels = [int(x * width) for x in channels]
        repeats = [int(x * depth) for x in repeats]

        # stochastic depth
        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0

        # efficient net
        self.upsample = nn.Upsample(
            scale_factor=scale, mode="linear", align_corners=False
        )

        self.stage1 = nn.Sequential(
            nn.Conv1d(201, channels[0], 3, stride=stage1_stride, padding=1, bias=False),
            nn.BatchNorm1d(channels[0], momentum=0.99, eps=1e-3),
        )
        self.dropout1 = nn.Dropout(p=dropout)

        self.stage2 = self._make_Block(
            SepConv,
            repeats[0],
            channels[0],
            channels[1],
            kernel_size[0],
            strides[0],
            se_scale,
        )
        self.dropout2 = nn.Dropout(p=dropout)
        self.stage3 = self._make_Block(
            MBConv,
            repeats[1],
            channels[1],
            channels[2],
            kernel_size[1],
            strides[1],
            se_scale,
        )
        self.dropout3 = nn.Dropout(p=dropout)
        self.stage4 = self._make_Block(
            MBConv,
            repeats[2],
            channels[2],
            channels[3],
            kernel_size[2],
            strides[2],
            se_scale,
        )
        self.dropout4 = nn.Dropout(p=dropout)
        self.stage9 = nn.Sequential(
            nn.Conv1d(channels[3], channels[4], 1, stride=1, bias=False),
            nn.BatchNorm1d(channels[4], momentum=0.99, eps=1e-3),
            Swish(),
        )
        # self.stage5 = self._make_Block(
        #     MBConv,
        #     repeats[3],
        #     channels[3],
        #     channels[4],
        #     kernel_size[3],
        #     strides[3],
        #     se_scale,
        # )
        # self.stage6 = self._make_Block(
        #     MBConv,
        #     repeats[4],
        #     channels[4],
        #     channels[5],
        #     kernel_size[4],
        #     strides[4],
        #     se_scale,
        # )
        # self.stage7 = self._make_Block(
        #     MBConv,
        #     repeats[5],
        #     channels[5],
        #     channels[6],
        #     kernel_size[5],
        #     strides[5],
        #     se_scale,
        # )
        # self.stage8 = self._make_Block(
        #     MBConv,
        #     repeats[6],
        #     channels[6],
        #     channels[7],
        #     kernel_size[6],
        #     strides[6],
        #     se_scale,
        # )
        # self.stage9 = nn.Sequential(
        #     nn.Conv1d(channels[7], channels[8], 1, stride=1, bias=False),
        #     nn.BatchNorm1d(channels[8], momentum=0.99, eps=1e-3),
        #     Swish(),
        # )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        # self.linear = nn.Linear(channels[8], num_classes)
        self.linear = nn.Linear(channels[4], num_classes)

    def forward(self, x):
        # change dimension order
        # x = x.permute(0, 2, 1).contiguous()

        x = self.upsample(x)
        x = self.stage1(x)
        x = self.dropout1(x)
        x = self.stage2(x)
        x = self.dropout2(x)
        x = self.stage3(x)
        x = self.dropout3(x)
        x = self.stage4(x)
        x = self.dropout4(x)
        # x = self.stage5(x)
        # x = self.stage6(x)
        # x = self.stage7(x)
        # x = self.stage8(x)
        x = self.stage9(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        #        x = F.softmax(x, dim = 1)
        return x

    def bypass_for_tsne(self, x):
        x = self.upsample(x)
        x = self.stage1(x)
        x = self.dropout1(x)
        x = self.stage2(x)
        x = self.dropout2(x)
        x = self.stage3(x)
        x = self.dropout3(x)
        x = self.stage4(x)
        x = self.dropout4(x)
        x = self.stage9(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_Block(
        self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale
    ):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(in_channels, out_channels, kernel_size, stride, se_scale, self.p)
            )
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)


class InefficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer_1 = nn.Conv1d(201, 256, 3)

    def forward(self, x):
        pass


def get_model(cfg):
    return EfficientNet(
        num_classes=cfg.num_classes,
        width_coef=1.0,
        depth_coef=1.0,
        scale=1.0,
        dropout=0.2,
        se_scale=4,
        strides=[1, 1, 1, 1, 1, 1, 1],
        stage1_stride=1,
    )


if __name__ == "__main__":
    namespace = argparse.Namespace()
    namespace.num_classes = 7
    module = get_model(namespace)

    summary(module, (201, 2))
