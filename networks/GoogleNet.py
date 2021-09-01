import torch
import torch.nn as nn
from collections import namedtuple

from Params import Params

SE_Ratio = 16
InceptionParams = namedtuple('InceptionParams', ['in_dim', 'conv1', 'reduce3', 'conv3', 'reduce5', 'conv5', 'pool', 'out_dim'])


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(ConvBlock, self).__init__()

        layers = []

        if Params.BatchNorm:
            layers += [nn.BatchNorm2d(in_ch)]

        layers += [
            nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
            nn.ReLU(inplace = True)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InceptionBlockv1(nn.Module):
    def __init__(self, params):
        super(InceptionBlockv1, self).__init__()
        self.conv1 = ConvBlock(params.in_dim, params.conv1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Sequential(
            ConvBlock(params.in_dim, params.reduce3, kernel_size=1, stride=1, padding=0),
            ConvBlock(params.reduce3, params.conv3, kernel_size=3, stride=1, padding=1),
        )
        self.conv5 = nn.Sequential(
            ConvBlock(params.in_dim, params.reduce5, kernel_size=1, stride=1, padding=0),
            ConvBlock(params.reduce5, params.conv5, kernel_size=5, stride=1, padding=2)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(params.in_dim, params.pool, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat(
            (
                self.conv1(x),
                self.conv3(x),
                self.conv5(x),
                self.pool(x)
            ),
            dim = 1
        )


class SE_InceptionBlockv1(InceptionBlockv1):
    def __init__(self, params):
        super(SE_InceptionBlockv1, self).__init__(params)
        self.se_squeeze = nn.AdaptiveAvgPool2d(1)
        self.se_excitation = nn.Sequential(
            nn.Linear(params.out_dim, params.out_dim // SE_Ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(params.out_dim // SE_Ratio, params.out_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        inc_output = super(SE_InceptionBlockv1, self).forward(x)

        n, c, _, _ = inc_output.size()
        attention = self.se_squeeze(inc_output).view(n, c)
        attention = self.se_excitation(attention).view(n, c, 1, 1)
        return inc_output * attention.expand_as(inc_output)


class GoogleNet(nn.Module):
    def __init__(self, in_channels, n_classes, img_size):
        super(GoogleNet, self).__init__()
        self.head = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        if Params.isTrue('UseSqueezeExcitation'):
            InceptionBlock = SE_InceptionBlockv1
        else:
            InceptionBlock = InceptionBlockv1

        # Layers up to aux classifier 1
        inc3a = InceptionBlock( InceptionParams(192, 64, 96, 128, 16, 32, 32, 256) )
        inc3b = InceptionBlock( InceptionParams(256, 128, 128, 192, 32, 96, 64, 480) )
        max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        inc4a = InceptionBlock( InceptionParams(480, 192, 96, 208, 16, 48, 64, 512) )
        self.step1 = nn.Sequential(
            inc3a,
            inc3b,
            max1,
            inc4a
        )

        # Layers up to aux classifier 2
        inc4b = InceptionBlock( InceptionParams(512, 160, 112, 224, 24, 64, 64, 512) )
        inc4c = InceptionBlock( InceptionParams(512, 128, 128, 256, 24, 64, 64, 512) )
        inc4d = InceptionBlock( InceptionParams(512, 112, 144, 288, 32, 64, 64, 528) )
        self.step2 = nn.Sequential(
            inc4b,
            inc4c,
            inc4d
        )

        # Layers up to main classifier
        inc4e = InceptionBlock( InceptionParams(528, 256, 160, 320, 32, 128, 128, 832) )
        max2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        inc5a = InceptionBlock( InceptionParams(832, 256, 160, 320, 32, 128, 128, 832) )
        inc5b = InceptionBlock( InceptionParams(832, 384, 192, 384, 48, 128, 128, 1024) )
        self.step3 = nn.Sequential(
            inc4e,
            max2,
            inc5a,
            inc5b
        )

        self.aux_classifier1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
            ConvBlock(512, 128, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(16 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, n_classes)
        )
        self.aux_classifier2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
            ConvBlock(528, 128, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(16 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, n_classes)
        )

        # Network is designed for 224 x 224 images. Here we introduce some flexibility
        # TODO: Aux Classifiers still assume 224x224 (although up to 256x256 should still work)
        test_img = torch.zeros( (1, 3, img_size, img_size) )
        with torch.no_grad():
            test_img = self.head(test_img)
            test_img = self.step1(test_img)
            test_img = self.step2(test_img)
            test_img = self.step3(test_img)
        out_img_size = test_img.shape[2]

        self.main_classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=out_img_size, stride=1, padding=0),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(1024, n_classes)
        )

        self.optimizer = Params.create_optimizer(self.parameters())


    def forward(self, x):
        result = dict()

        x = self.head(x)

        x = self.step1(x)
        result['aux1'] = self.aux_classifier1(x)

        x = self.step2(x)
        result['aux2'] = self.aux_classifier2(x)

        x = self.step3(x)
        result['final'] = self.main_classifier(x)

        return result
