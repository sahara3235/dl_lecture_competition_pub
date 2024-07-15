import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool1 = nn.AvgPool1d(kernel_size=5, stride=3)
        self.conv0 = BasicConv1d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv1d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.pool2 = nn.AdaptiveAvgPool1d((1, 1))
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = self.pool1(x)  # (N, 768, 17, 17)
        x = self.conv0(x)  # (N, 768, 5, 5)
        x = self.conv1(x)  # (N, 128, 5, 5)
        x = self.pool2(x)  # (N, 768, 1, 1)
        x = torch.flatten(x, 1)  # (N, 768)
        x = self.fc(x)  # (N, 1000)

        return x
    
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch5x5 = nn.Sequential(
            BasicConv1d(in_channels, 48, kernel_size=1),
            BasicConv1d(48, 64, kernel_size=5, padding=2),
        )
        self.branch3x3db1 = nn.Sequential(
            BasicConv1d(in_channels, 64, kernel_size=1),
            BasicConv1d(64, 96, kernel_size=3, padding=1),
            BasicConv1d(96, 96, kernel_size=3, padding=1),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            BasicConv1d(in_channels, pool_features, kernel_size=1),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3db1 = self.branch3x3db1(x)
        branch_pool = self.branch_pool(x)

        out = torch.cat([branch1x1, branch5x5, branch3x3db1, branch_pool], 1)

        return out
    
class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = BasicConv1d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3db1 = nn.Sequential(
            BasicConv1d(in_channels, 64, kernel_size=1),
            BasicConv1d(64, 96, kernel_size=3, padding=1),
            BasicConv1d(96, 96, kernel_size=3, stride=2),
        )
        self.branch_pool = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3db1 = self.branch3x3db1(x)
        branch_pool = self.branch_pool(x)

        out = torch.cat([branch3x3, branch3x3db1, branch_pool], 1)

        return out
    
class InceptionC(nn.Module):
    def __init__(self, in_channels, ch7x7):
        super().__init__()
        # fmt: off
        self.branch1x1 = BasicConv1d(in_channels, 192, kernel_size=1)
        self.branch7x7 = nn.Sequential(
            BasicConv1d(in_channels, ch7x7, kernel_size=1),
            BasicConv1d(ch7x7, ch7x7, kernel_size= 7, padding= 3),
        )
        self.branch7x7dbl = nn.Sequential(
            BasicConv1d(in_channels, ch7x7, kernel_size=1),
            BasicConv1d(ch7x7, ch7x7, kernel_size=7, padding= 3),
            BasicConv1d(ch7x7, ch7x7, kernel_size=7, padding= 3),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            BasicConv1d(in_channels, 192, kernel_size=1),
        )
        # fmt: on

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7dbl = self.branch7x7dbl(x)
        branch_pool = self.branch_pool(x)

        out = torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)

        return out
    
class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv1d(in_channels, 192, kernel_size=1),
            BasicConv1d(192, 320, kernel_size=3, stride=2),
        )
        self.branch7x7x3 = nn.Sequential(
            BasicConv1d(in_channels, 192, kernel_size=1),
            BasicConv1d(192, 192, kernel_size= 7, padding= 3),
            BasicConv1d(192, 192, kernel_size=3, stride=2),
        )
        self.branch_pool = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7x3 = self.branch7x7x3(x)
        branch_pool = self.branch_pool(x)

        out = torch.cat([branch3x3, branch7x7x3, branch_pool], 1)

        return out
    
class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = BasicConv1d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv1d(in_channels, 384, kernel_size=1)
        self.branch3x3_2 = BasicConv1d(384, 384, kernel_size= 3, padding= 1)
        self.branch3x3dbl_1 = BasicConv1d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(384, 384, kernel_size= 3, padding=1)
        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            BasicConv1d(in_channels, 192, kernel_size=1),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3),
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl),
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = self.branch_pool(x)

        out = torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)

        return out
    
class Inception3(nn.Module):
    def __init__(
        self,
        num_classes,
        seq_len: int,
        in_channel,
        aux_logits=True,
        dropout=0.2,
    ):
        super().__init__()
        self.aux_logits = aux_logits
        self.conv_1a_3x3 = BasicConv1d(in_channel, 32, kernel_size=3, stride=2)
        self.conv_2a_3x3 = BasicConv1d(32, 32, kernel_size=3)
        self.conv_2b_3x3 = BasicConv1d(32, 64, kernel_size=3, padding=1)
        self.maxpool_3a_3x3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv_3b_1x1 = BasicConv1d(64, 80, kernel_size=1)
        self.conv_4a_3x3 = BasicConv1d(80, 192, kernel_size=3)
        self.maxpool_3a_3x3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.miaxed_5b = InceptionA(192, pool_features=32)
        self.miaxed_5c = InceptionA(256, pool_features=64)
        self.miaxed_5d = InceptionA(288, pool_features=64)
        self.miaxed_6a = InceptionB(288)
        self.miaxed_6b = InceptionC(768, ch7x7=128)
        self.miaxed_6c = InceptionC(768, ch7x7=160)
        self.miaxed_6d = InceptionC(768, ch7x7=160)
        self.miaxed_6e = InceptionC(768, ch7x7=192)
        self.aux = InceptionAux(768, num_classes) if aux_logits else None
        self.miaxed_7a = InceptionD(768)
        self.miaxed_7b = InceptionE(1280)
        self.miaxed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool1d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if hasattr(m, "stddev"):
                # Auxiliary Classifier の2つの層
                nn.init.trunc_normal_(m.weight, std="stddev")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="conv1d")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_1a_3x3(x)  # (N, 32, 149, 149)
        x = self.conv_2a_3x3(x)  # (N, 32, 147, 147)
        x = self.conv_2b_3x3(x)  # (N, 64, 147, 147)
        x = self.maxpool_3a_3x3(x)  # (N, 64, 73, 73)

        x = self.conv_3b_1x1(x)  # (N, 80, 73, 73)
        x = self.conv_4a_3x3(x)  # (N, 192, 71, 71)
        x = self.maxpool_3a_3x3(x)  # (N, 192, 35, 35)

        x = self.miaxed_5b(x)  # (N, 256, 35, 35)
        x = self.miaxed_5c(x)  # (N, 288, 35, 35)
        x = self.miaxed_5d(x)  # (N, 288, 35, 35)

        x = self.miaxed_6a(x)  # (N, 768, 17, 17)
        x = self.miaxed_6b(x)  # (N, 768, 17, 17)
        x = self.miaxed_6c(x)  # (N, 768, 17, 17)
        x = self.miaxed_6d(x)  # (N, 768, 17, 17)
        x = self.miaxed_6e(x)  # (N, 768, 17, 17)

        aux = self.aux(x) if self.aux_logits and self.training else None

        x = self.miaxed_7a(x)  # (N, 1280, 8, 8)
        x = self.miaxed_7b(x)  # (N, 2048, 8, 8)
        x = self.miaxed_7c(x)  # (N, 2048, 8, 8)
        x = self.avgpool(x)
        x = self.dropout(x)  # (N, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (N, 2048)
        x = self.fc(x)  # (N, 1000)

        if self.training and self.aux_logits:
            return x, aux
        else:
            return aux


def inception_v3():
    return Inception3()
