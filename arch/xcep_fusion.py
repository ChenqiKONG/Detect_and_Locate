"""
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

__all__ = ['xception']

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.seg1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=5, stride=2, padding=1),
        )
        self.seg2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=512, out_channels=2, kernel_size=5, stride=2, padding=1),
        )

        self.noise1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=5, stride=2, padding=1),
        )
        self.noise2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=5, stride=2, padding=1),
        )

        self.seg3_in = nn.Sequential(
            nn.Conv2d(in_channels=728, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0), )

        self.seg3_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1), )

        self.noise3_in = nn.Sequential(
            nn.Conv2d(in_channels=728, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0), )
        self.noise3_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1), )

        self.conv_cls = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0),
            nn.Conv2d(in_channels=256, out_channels=728, kernel_size=1, padding=0),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.blockcq_fusion = Block(128, 128, 2, 2, start_with_relu=True, grow_first=False)

        self.fc = nn.Sequential(
            nn.Linear(728 * 4, 100),
            nn.Linear(100, 1),
        )

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x_seg1 = self.seg1(x)
        x_noise1 = self.noise1(x)
        x = self.block1(x)
        x = self.block2(x)
        x_seg2 = self.seg2(x)
        x_noise2 = self.noise2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        feature_seg = self.seg3_in(x)
        feature_noise = self.noise3_in(x)
        x_seg3 = self.seg3_out(feature_seg)
        x_noise3 = self.noise3_out(feature_noise)
        feature = torch.cat((feature_seg, feature_noise), 1)
        feature_atten = x_seg3[:, 0, :, :].unsqueeze(1) * feature
        x = self.blockcq_fusion(feature_atten)
        x = self.conv_cls(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, x_seg1, x_seg2, x_seg3, x_noise1, x_noise2, x_noise3


def xception(pretrained=True, **kwargs):
    """
    Construct Xception.
    """
    model = Xception(**kwargs)
    if pretrained:
        pretrained_dict = torch.load('best.pth',map_location={'cuda:0': 'cuda:1'})
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        for k, v in pretrained_dict.items():
            print(k)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load successfully!')
    return model

if __name__ == "__main__":
    faces = torch.rand(size=(8, 3, 299, 299))
    MODEL = xception(pretrained=False)
    x1, x2, x3, x4, x5, x6, x7 = MODEL(faces)
    print(x1.size())
    print(x2.size())
    print(x3.size())
    print(x4.size())
    print(x5.size())
    print(x6.size())
    print(x7.size())
