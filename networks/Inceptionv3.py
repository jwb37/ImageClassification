from .SE_Layer import SE_Layer
from Params import Params

import torch
import torch.nn as nn
from torchvision.models.inception import BasicConv2d, Inception3, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux


def se_wrap_block(BlockClass):
    class WrappedInceptionBlock(nn.Module):
        def __init__(self, in_channels, *args, **kwargs):
            super().__init__()
            self.inception = BlockClass(in_channels, *args, **kwargs)

            # We need to know the number of channels in the output tensor.
            # This feels a bit sketchy, but we'll just put a sample tensor through
            # the inception block and examine the shape of the tensor that comes out
            test_tensor = torch.zeros( (1, in_channels, 299, 299) )
            with torch.no_grad():
                out_tensor = self.inception(test_tensor)
            out_channels = out_tensor.shape[1]

            self.se = SE_Layer(out_channels)

        def forward(self, x):
            x = self.inception(x)
            return self.se(x)

    return WrappedInceptionBlock


class Conv2d_NoBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.net(x)


if Params.isTrue('BatchNorm'):
    WrappedBlocks = [BasicConv2d]
else:
    WrappedBlocks = [Conv2d_NoBatchNorm]
WrappedBlocks += [se_wrap_block(BlockClass) for BlockClass in [InceptionA, InceptionB, InceptionC, InceptionD, InceptionE]]
WrappedBlocks += [InceptionAux]


def Inceptionv3_Net(n_classes, **kwargs):
    if Params.isTrue('UseSqueezeExcitation'):
        return Inception3(n_classes, inception_blocks=WrappedBlocks, **kwargs)
    else:
        return Inception3(n_classes, **kwargs)
