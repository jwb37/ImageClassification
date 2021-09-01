import sys
import torch
import torch.nn as nn
from collections import namedtuple
from sklearn.metrics import top_k_accuracy_score

from Params import Params

SE_Ratio = 16
ClassificationLoss = nn.CrossEntropyLoss()
AuxClassifierWeight = 0.3
InceptionParams = namedtuple('InceptionParams', ['in_dim', 'conv1', 'reduce3', 'conv3', 'reduce5', 'conv5', 'pool', 'out_dim'])


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(ConvBlock, self).__init__()

        if Params.BatchNorm:
            layers = [nn.BatchNorm2d(in_ch)
        else:
            layers = []

        layers += [
            nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
            nn.ReLU(inplace = True)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InceptionBlockv2_1(nn.Module):
    def __init__(self, params):
        super(InceptionBlockv2_1, self).__init__()
        self.conv1 = ConvBlock(params.in_dim, params.conv1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Sequential(
            ConvBlock(params.in_dim, params.reduce3, kernel_size=1, stride=1, padding=0),
            ConvBlock(params.reduce3, params.conv3, kernel_size=3, stride=1, padding=1),
        )
        self.conv5 = nn.Sequential(
            ConvBlock(params.in_dim, params.reduce5, kernel_size=1, stride=1, padding=0),
            ConvBlock(params.reduce5, params.conv5, kernel_size=3, stride=1, padding=1),
            ConvBlock(params.reduce5, params.conv5, kernel_size=3, stride=1, padding=1),
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

def paired_conv_1d( k, channel_sizes1, channel_sizes2 ):
    p = (n-1) // 2
    return [
        ConvBlock( *channel_sizes1, kernel_size=(1, k), stride=1, padding=(0, 0, p, p) ),
        ConvBlock( *channel_sizes2, kernel_size=(k, 1), stride=1, padding=(p, p, 0, 0) )
    ]

# TODO : look up correct filter sizes
class InceptionBlockv2_2(nn.Module):
    def __init__(self, n, params):
        super(InceptionBlockv2_2, self).__init__()
        self.conv1 = ConvBlock(params.in_dim, params.conv1, kernel_size=1, stride=1, padding=0)

        p = (n-1) // 2
        self.conv_single = nn.Sequential(
            ConvBlock(params.in_dim, params.reduce3, kernel_size=1, stride=1, padding=0),
            *paired_conv_1d( n, (params.reduce3, params.conv3), (params.reduce3, params.conv3))
        )
        self.conv_double = nn.Sequential(
            ConvBlock(params.in_dim, params.reduce5, kernel_size=1, stride=1, padding=0),
            *paired_conv_1d( n, (params.reduce5, params.conv5), (params.reduce3, params.conv3)),
            *paired_conv_1d( n, (params.reduce5, params.conv5), (params.reduce3, params.conv3))
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(params.in_dim, params.pool, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat(
            (
                self.conv1(x),
                self.conv_single(x),
                self.conv_double(x),
                self.pool(x)
            ),
            dim = 1
        )

    class InceptionBlockv2_3(nn.Module):
    def __init__(self, in_dim):
        super(InceptionBlockv2_2, self).__init__()
        self.conv1 = ConvBlock(in_dim, 320, kernel_size=1, stride=1, padding=0)

        self.conv_single_head = ConvBlock(in_dim, 384, kernel_size=1, stride=1, padding=0)
        self.conv_single_1, self.conv_single_2 = paired_conv_1d( 3, (384, 384), (384, 384) )

        self.conv_double_head = nn.Sequential(
            ConvBlock(params.in_dim, 448, kernel_size=1, stride=1, padding=0),
            ConvBlock(448, 384, kernel_size=3, stride=1, padding=1),
        )
        self.conv_double_1, self.conv_double_2 = paired_conv_1d( 3, (384, 384), (384, 384) )

        self.pool = nn.Sequential(
            # Max Pool changed to Avg Pool according to TorchVision model
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_dim, 192, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        single_input = self.conv_single_head(x)
        double_input = self.conv_double_head(x)
        return torch.cat(
            (
                self.conv1(x),
                self.conv_single_1(single_input),
                self.conv_single_2(single_input),
                self.conv_double_1(double_input),
                self.conv_double_2(double_input),
                self.pool(x)
            ),
            dim = 1
        )

class InceptionBlock_v2_dim_reduce1(nn.Module):
    def __init__(self, in_dim = 288):
        super(InceptionBlock_v2_dim_reduce1, self).__init__()

        self.single = ConvBlock(in_dim, 384, kernel_size = 3, stride=2, padding=0)
        self.double = nn.Sequential(
            ConvBlock(in_dim, 64, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(64, 96, kernel_size = 3, stride = 1, padding = 1),
            ConvBlock(96, 96, kernel_size = 3, stride = 2, padding = 0)
        )
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        return torch.cat(
            (
                self.single,
                self.double,
                self.pool
            ),
            dim = 1
        )

class InceptionBlock_v2_dim_reduce2(nn.Module):
    def __init__(self, in_dim = 768):
        super(InceptionBlock_v2_dim_reduce1, self).__init__()

        self.single = nn.Sequential(
            ConvBlock(in_dim, 192, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(192, 320, kernel_size = 3, stride = 2, padding = 0)
        )
        self.double = nn.Sequential(
            ConvBlock(in_dim, 192, kernel_size = 1, stride = 1, padding = 0),
            *paired_conv_1d( 7, (192, 192), (192, 192) ),
            ConvBlock(192, 192, kernel_size = 3, stride = 2, padding = 0)
        )
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        return torch.cat(
            (
                self.single,
                self.double,
                self.pool
            ),
            dim = 1
        )


class SE_Layer(nn.module):
    def __init__(self, params):
        super(SE_Layer, self).__init__(in_dim)
        self.se_squeeze = nn.AdaptiveAvgPool2d(1)
        self.se_excitation = nn.Sequential(
            nn.Linear(params.in_dim, params.in_dim // SE_Ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(params.in_dim // SE_Ratio, params.in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        attention = self.se_squeeze(x).view(n, c)
        attention = self.se_excitation(attention).view(n, c, 1, 1)
        return x * attention.expand_as(x)


class Inceptionv2(nn.Module):
    def __init__(self, in_channels, n_classes, img_size):
        super(Inceptionv2, self).__init__()
        self.head = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        InceptionBlock = {
            'v1': InceptionBlockv1,
            'SE_v1': SE_InceptionBlockv1
        }[Params.InceptionType]

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

        self.n_classes = n_classes
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


    def calculate_losses(self, result, targets, use_aux=True):
        loss_record = dict()  
        loss_record['final'] = ClassificationLoss( result['final'], targets )

        if use_aux:
            loss_record['aux1'] = ClassificationLoss( result['aux1'], targets )
            loss_record['aux2'] = ClassificationLoss( result['aux2'], targets )
            loss_record['final'] += AuxClassifierWeight * (loss_record['aux1'] + loss_record['aux2'])

        return loss_record


    def training_step(self, images, targets):
        self.optimizer.zero_grad()

        result = self.forward(images)
        loss_record = self.calculate_losses(result, targets)
        loss_record['final'].backward()

        self.optimizer.step()
        return loss_record


    def test(self, images, targets, use_aux=False):
        result = self.forward(images)
        loss_record = self.calculate_losses(result, targets, use_aux)

        # Accuracy counts
        model_prediction = result['final'].cpu().numpy()
        targets_np = targets.cpu().numpy()
        accuracy = {
            'top1': top_k_accuracy_score(targets_np, model_prediction, k=1, normalize=False, labels=range(self.n_classes)),
            'top5': top_k_accuracy_score(targets_np, model_prediction, k=5, normalize=False, labels=range(self.n_classes))
        }

        return [loss_record, accuracy]


    def save(self, filename):
        save_state = dict()
        save_state['model'] = self.state_dict()
        save_state['optimizer'] = self.optimizer.state_dict()
        torch.save( save_state, filename )


    def load(self, filename):
        save_state = torch.load( filename )
        self.load_state_dict(save_state['model'])
        self.optimizer.load_state_dict(save_state['optimizer'])
