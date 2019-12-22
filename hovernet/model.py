# PyTorch
import torch
from torch import nn
from torch.nn import functional as F
# Pretrained models
import segmentation_models_pytorch as smp

# Model pretrained on imagenet
# See: https://github.com/qubvel/segmentation_models.pytorch/
model = smp.Unet(encoder_name="resnet34",
                 encoder_depth=5,
                 encoder_weights="imagenet",
                 decoder_use_batchnorm=True,
                 decoder_channels=[256, 128, 64, 32, 16],
                 # See: https://arxiv.org/pdf/1808.08127.pdf
                 decoder_attention_type=None,
                 activation=None,
                 in_channels=3,
                 classes=1
                 )


class _BNRelu(nn.Module):
    def __init__(self, num_features):
        super(_BNRelu, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, inputs):
        return F.relu(self.bn(inputs), inplace=True)


class _ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(_ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1,
                               stride=stride, padding=1)
        self.bn_relu1 = _BNRelu(out_channels//4)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3,
                               stride=stride, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels,
                               kernel_size=1, stride=stride)
        self.bn_relu2 = _BNRelu(out_channels)

        if stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride)

    def forward(self, inputs):
        out = self.bn_relu1(self.conv1(inputs))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else out
        out = self.bn_relu1(self.conv2(out))
        out = self.bn_relu2(self.conv3(out))
        out += shortcut

        return out


class _DenseUnit(nn.Module):
    def __init__(self, in_channels):
        super(_DenseUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.bn_relu1 = _BNRelu(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=5)
        self.bn_relu2 = _BNRelu(32)

    def forward(self, inputs):
        out = self.bn_relu1(self.conv1(inputs))
        out = self.bn_relu2(self.conv2(out))

        return torch.cat([out, F.upsample(inputs)])


class _Encoder(nn.Module):
    def __init__(self, channels):
        super(_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(channels[0], channels[0],
                               kernel_size=7, padding=3)
        self.residual_block1 = nn.Sequential(
            *[_ResidualUnit(channels[0], channels[1], stride=1), ] * 3
        )
        self.residual_block2 = nn.Sequential(
            *[_ResidualUnit(channels[1], channels[2], stride=2), ] * 4
        )
        self.residual_block3 = nn.Sequential(
            *[_ResidualUnit(channels[2], channels[3], stride=4), ] * 6
        )
        self.residual_block4 = nn.Sequential(
            *[_ResidualUnit(channels[3], channels[4], stride=8), ] * 3
        )
        self.conv2 = nn.Conv2d(channels[4], channels[5], kernel_size=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.conv2(x)

        return x


class _Decoder(nn.Module):
    def __init__(self, input_shape, in_channels):
        super(_Decoder, self).__init__()
        self.upsample = nn.Upsample(size=input_shape)
        self.conv1 = nn.Conv2d(in_channels, 256,
                               kernel_size=5)
        self.dense_block1 = nn.Sequential(
            *[_DenseUnit(in_channels), ] * 8
        )
        self.conv2 = nn.Conv2d(32, 512,
                               kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(512, 128,
                               kernel_size=5)
        self.dense_block2 = nn.Sequential(
            *[_DenseUnit(in_channels), ] * 4
        )
        self.conv4 = nn.Conv2d(32, 128,
                               kernel_size=1)
        self.conv5 = nn.Conv2d(128, 256,
                               kernel_size=5)
        self.conv6 = nn.Conv2d(256, 64,
                               kernel_size=1)

    def forward(self, inputs):
        x = self.upsample(inputs)
        x = self.conv1(x)
        x = self.dense_block1(x)
        x = self.conv3(self.upsample(self.conv2(x)))
        x = self.dense_block2(x)
        x = self.conv5(self.upsample(self.conv4(x)))
        x = self.conv6(x)

        return x


class HoverNet(nn.Module):
    def __init__(self, input_shape, in_channels):
        super(HoverNet, self).__init__()
        self.encoder = _Encoder([in_channels, 256, 512, 2048, 1024])
        self.decoder_np = _Decoder(input_shape, in_channels)
        self.decoder_hv = _Decoder(input_shape, in_channels)

    def forward(self, inputs):
        x = self.encoder(inputs)
        out_np = self.decoder_np(x)
        out_hv = self.decoder_hv(x)

        return out_np, out_hv
