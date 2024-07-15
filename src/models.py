import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from FPAE import FPN_Gray
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)



class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return self.dropout(X)
    

class RNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        rnn_layers: int = 3,
        hid_dim: int = 128,
    ) -> None:
        super().__init__()
        self.rnn=nn.RNN(in_channels,hid_dim,rnn_layers,batch_first=True)
        self.line=nn.Linear(hid_dim,num_classes)


    def forward(self, X):
        x = x.permute(2,0, 1)
        y_rnn, h = self.rnn(x, None) 
        y = self.line(y_rnn[:, -1, :])

        return y
    

class Inception(nn.Module):
    
    def __init__(self, in_channels, ch1x1, ch3x3, ch5x5, pool_proj):
        super(Inception, self).__init__()
        
        # 1 conv
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 1 conv -> 3 conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch1x1, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 1 conv -> 5 conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch1x1, ch5x5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3 pool -> 1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)
    
class BasicConvClassifier2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks1 = nn.Sequential(
            ConvBlock(in_channels, hid_dim,kernel_size=1),
            ConvBlock(hid_dim, hid_dim,kernel_size=3),
        )

        self.blocks2 = nn.Sequential(
            ConvBlock(in_channels, hid_dim,kernel_size=3),
            ConvBlock(hid_dim, hid_dim,kernel_size=3),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes/2),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        x1 = self.blocks1(X)
        x1=self.head(x1)
        x2 = self.blocks2(X)
        x2=self.head(x2)

        return torch.cat((x1,x2),dim=0)