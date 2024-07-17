import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 40
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim,kernel_size1=3,kernel_size2=3),
            ConvBlock(hid_dim, hid_dim,kernel_size1=3,kernel_size2=3),
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
        kernel_size1: int = 3,
        kernel_size2: int = 3,
        p_drop: float = 0.4,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size1, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size2, padding="same")
        #self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.elu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.elu(self.batchnorm1(X))

        #X = self.conv2(X)
        #X = F.glu(X, dim=-2)

        return self.dropout(X)
    

class RNN(nn.Module):
    def __init__(
        self,trial,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        rnn_layers: int = 3,
        hid_dim: int = 128,
        p_drop: float =0.2
    ) -> None:
        super().__init__()
        self.rnn=nn.RNN(in_channels,hid_dim,rnn_layers,batch_first=True)
        self.line=nn.Linear(hid_dim,num_classes)
        self.dropout = nn.Dropout(p_drop)


    def forward(self, x):
        #x = self.dropout(x)
        x = x.permute( 0,2, 1)
        y_rnn, h = self.rnn(x, None) 
        y = self.line(y_rnn[:, -1, :])

        return y
    

    
class BasicConvClassifier2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 32
    ) -> None:
        super().__init__()

        self.blocks1 = nn.Sequential(
            ConvBlock(in_channels, hid_dim,kernel_size1=1),
            #ConvBlock(hid_dim, hid_dim,kernel_size=1),
        )

        self.blocks2 = nn.Sequential(
            ConvBlock(in_channels, hid_dim,kernel_size1=1),
            ConvBlock(hid_dim, hid_dim,kernel_size1=3),
        )

        self.blocks3 = nn.Sequential(
            ConvBlock(in_channels, hid_dim,kernel_size1=1),
            ConvBlock(hid_dim, hid_dim,kernel_size1=5),
        )

        self.blocks4 = nn.Sequential(
            nn.AvgPool1d(stride=1,kernel_size=3,padding=1),
            ConvBlock(in_channels, hid_dim,kernel_size1=1),
            #ConvBlock(hid_dim, hid_dim,kernel_size=3),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(4*hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        x1 = self.blocks1(X)
        x2 = self.blocks2(X)
        x3 = self.blocks3(X)
        x4 = self.blocks4(X)
        X=torch.cat([x1,x2,x3,x4],dim=1)

        return self.head(X)
    
class LSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        rnn_layers: int = 2,
        hid_dim: int = 5,
        p_drop: float =0.3
    ) -> None:
        super().__init__()
        self.lstm=nn.LSTM(in_channels,hid_dim,rnn_layers,batch_first=True)
        self.line=nn.Linear(hid_dim,num_classes)
        self.dropout = nn.Dropout(p_drop)


    def forward(self, x):
        x = self.dropout(x)
        x = x.permute( 0,2, 1)
        y_rnn, h = self.lstm(x, None) 
        y = self.line(y_rnn[:, -1, :])

        return y
    