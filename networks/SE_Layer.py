import torch.nn as nn

SE_Ratio = 16

class SE_Layer(nn.Module):
    def __init__(self, in_dim):
        super(SE_Layer, self).__init__()

        self.se_squeeze = nn.AdaptiveAvgPool2d(1)
        self.se_excitation = nn.Sequential(
            nn.Linear(in_dim, in_dim // SE_Ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // SE_Ratio, in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        attention = self.se_squeeze(x).view(n, c)
        attention = self.se_excitation(attention).view(n, c, 1, 1)
        return x * attention.expand_as(x)
