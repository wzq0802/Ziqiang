import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
class Conv_Block2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class DownSample2d(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(DownSample2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 2, 2, 0),
        )

    def forward(self, x):
        return self.layer(x)
class UpSample2d(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(UpSample2d, self).__init__()
        self.layer = nn.ConvTranspose2d(inchannel,outchannel,2,2,0)

    def forward(self, x,feature_map):
        out = self.layer(x)
        return torch.cat((out,feature_map),dim=1)

class SegNet(nn.Module):
    def __init__(self, n=8):
        super(SegNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, n * 1, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(n * 1, n * 1, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),
            nn.Conv2d(n * 1, n * 2, 2, 2, 0),

            nn.Conv2d(n * 2, n * 2, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),
            nn.Conv2d(n * 2, n * 4, 2, 2, 0),

            nn.Conv2d(n * 4, n * 4, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),
            nn.Conv2d(n * 4, n * 8, 2, 2, 0),

            nn.Conv2d(n * 8, n * 8, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),
            nn.Conv2d(n * 8, n * 16, 2, 2, 0),

            nn.Conv2d(n * 16, n * 16, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),
            nn.Conv2d(n * 16, n * 16, 2, 2, 0),

            nn.ConvTranspose2d(n * 16, n * 16, 2, 2, 0),
            nn.Conv2d(n * 16, n * 16, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),

            nn.ConvTranspose2d(n * 16, n * 8, 2, 2, 0),
            nn.Conv2d(n * 8, n * 8, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),

            nn.ConvTranspose2d(n * 8, n * 4, 2, 2, 0),
            nn.Conv2d(n * 4, n * 4, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),

            nn.ConvTranspose2d(n * 4, n * 2, 2, 2, 0),
            nn.Conv2d(n * 2, n * 2, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),

            nn.ConvTranspose2d(n * 2, n, 2, 2, 0),
            nn.Conv2d(n, n, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(),

            nn.Conv2d(n, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

class UNet(nn.Module):
    def __init__(self,n=8):
        super(UNet, self).__init__()
        self.c1 = Conv_Block2(1, n)
        self.d1 = DownSample2d(n,n*2)
        self.c2 = Conv_Block2(n*2, n*2)
        self.d2 = DownSample2d(n*2,n*4)
        self.c3 = Conv_Block2(n*4, n*4)
        self.d3 = DownSample2d(n*4,n*8)
        self.c4 = Conv_Block2(n*8, n*8)
        self.d4 = DownSample2d(n*8,n*16)
        self.c5 = Conv_Block2(n*16, n*16)
        self.u1 = UpSample2d(n*16,n*8)
        self.c6 = Conv_Block2(n*16,n*8)
        self.u2 = UpSample2d(n*8,n*4)
        self.c7 = Conv_Block2(n*8,n*4)
        self.u3 = UpSample2d(n*4,n*2)
        self.c8 = Conv_Block2(n*4,n*2)
        self.u4 = UpSample2d(n*2,n)
        self.c9 = Conv_Block2(n*2,n)
        self.out = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)


class edge(nn.Module):
    def __init__(self):
        super(edge, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=1)

        self.channel_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1,dilation=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.residual_connection = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1,dilation=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1,dilation=1),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):

        x1 = self.conv1(x)

        attention = self.channel_attention(x1)
        x_attention = x1 * attention

        x_residual = self.residual_connection(x_attention)
        x_residual = F.relu(x_attention+x_residual)

        x_avg_pool = F.avg_pool2d(x_residual, kernel_size=1, stride=1)
        x_max_pool = F.max_pool2d(x_residual, kernel_size=1, stride=1)

        edge_features = x_avg_pool + x_max_pool
        return edge_features


class DualSegNet(nn.Module):
    def __init__(self, n=8,edge_model_path='edge.pth'):
        super(DualSegNet, self).__init__()
        self.edge_model = edge()
        self.edge_model.load_state_dict(torch.load(edge_model_path))
        self.edge_model.eval()
        self.conv1 = nn.Conv2d(2, n*1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv2 = Conv_Block2(n * 1, n * 1)
        self.conv3 = Conv_Block2(n * 2, n * 2)
        self.conv4 = Conv_Block2(n * 4, n * 4)
        self.conv5 = Conv_Block2(n * 8, n * 8)
        self.conv6 = Conv_Block2(n * 16, n * 16)
        self.conv7 = nn.Conv2d(n, 1, kernel_size=3, padding=1)
        self.c11 = Conv_Block2(1, n // 2)
        self.c12 = Conv_Block2(n // 2, n)
        self.c13 = Conv_Block2(n, n * 2)
        self.c14 = Conv_Block2(n * 2, n * 4)
        self.c15 = Conv_Block2(n * 4, n * 8)
        self.conv9 = Conv_Block2(n * 16, n * 8)
        self.conv10 = Conv_Block2(n * 8, n * 4)
        self.conv11 = Conv_Block2(n * 4, n * 2)
        self.conv12 = Conv_Block2(n * 2, n * 1)
        self.conv13 = Conv_Block2(n * 1, n // 2)

        self.down1 = nn.Conv2d(n*1, n*2, 2, 2, 0)
        self.down2 = nn.Conv2d(n * 2, n * 4, 2, 2, 0)
        self.down3 = nn.Conv2d(n * 4, n * 8, 2, 2, 0)
        self.down4 = nn.Conv2d(n * 8, n * 16, 2, 2, 0)
        self.down5 = nn.Conv2d(n * 16, n * 16, 2, 2, 0)
        self.d = DownSample2d(n // 2, n // 2)
        self.d5 = DownSample2d(n, n)
        self.d6 = DownSample2d(n * 2, n * 2)
        self.d7 = DownSample2d(n * 4, n * 4)
        self.d8 = DownSample2d(n * 8, n * 8)

        self.up1 = nn.ConvTranspose2d(n*16, n*16, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(n * 16, n * 8, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(n * 8, n * 4, 2, 2, 0)
        self.up4 = nn.ConvTranspose2d(n * 4, n * 2, 2, 2, 0)
        self.up5 = nn.ConvTranspose2d(n * 2, n * 1, 2, 2, 0)

    def forward(self, x):
        edge_256_ = self.edge_model(x)
        edge_256 = self.c11(edge_256_)

        edge_128 = self.d(edge_256)
        edge_128 = self.c12(edge_128)

        edge_64 = self.d5(edge_128)
        edge_64 = self.c13(edge_64)

        edge_32 = self.d6(edge_64)
        edge_32 = self.c14(edge_32)

        edge_16 = self.d7(edge_32)
        edge_16 = self.c15(edge_16)

        edge_8 = self.d8(edge_16)
        edge_8 = self.conv5(edge_8)

        x1 = self.relu(self.conv1(torch.cat((x, edge_256_), dim=1)))
        x2 = self.down1(self.conv2(x1))
        x3 = self.down2(self.conv3(x2))
        x4 = self.down3(self.conv4(x3))
        x5 = self.down4(self.conv5(x4))
        x6 = self.down5(self.conv6(x5))
        x7 = self.conv6(self.up1(x6))
        x8 = self.conv5(self.up2(x7))
        x9 = self.conv4(self.up3(x8))
        x10 = self.conv3(self.up4(x9))
        x11 = self.conv2(self.up5(x10))

        x6 = self.conv9(x6)
        x7 = self.conv6(self.up1(torch.cat((x6, edge_8), dim=1)))
        x7 = self.conv9(x7)
        x8 = self.conv5(self.up2(torch.cat((x7, edge_16), dim=1)))
        x8 = self.conv10(x8)
        x9 = self.conv4(self.up3(torch.cat((x8, edge_32), dim=1)))
        x9 = self.conv11(x9)
        x10 = self.conv3(self.up4(torch.cat((x9, edge_64), dim=1)))
        x10 = self.conv12(x10)
        x11 = self.conv2(self.up5(torch.cat((x10, edge_128), dim=1)))
        x11 = self.conv13(x11)
        x12 = self.sigmoid(self.conv7(torch.cat((x11, edge_256), dim=1)))

        return x12

class Fusion(nn.Module):
    def __init__(self, in_channels):
        super(Fusion, self).__init__()
        self.conv1x1_sigmoid = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv1x1_relu = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, y):
        conv1x1_sigmoid_result = self.conv1x1_sigmoid(x)
        multiply_result = torch.mul(conv1x1_sigmoid_result, y)
        fusion_result = x + multiply_result
        fusion_result = self.conv1x1_relu(fusion_result)
        fusion_result = torch.relu(fusion_result)
        return fusion_result

class DualUNet(nn.Module):
    def __init__(self,n=8,edge_model_path='edge.pth'):
        super(DualUNet, self).__init__()
        self.c1 = Conv_Block2(2, n)
        self.d1 = DownSample2d(n,n*2)
        self.c2 = Conv_Block2(n*2, n*2)
        self.d2 = DownSample2d(n*2,n*4)
        self.c3 = Conv_Block2(n*4, n*4)
        self.d3 = DownSample2d(n*4,n*8)
        self.c4 = Conv_Block2(n*8, n*8)
        self.d4 = DownSample2d(n*8,n*16)
        self.c5 = Conv_Block2(n*16, n*16)
        self.u1 = UpSample2d(n*16,n*8)
        self.c6 = Conv_Block2(n*16,n*8)
        self.u2 = UpSample2d(n*8,n*4)
        self.c7 = Conv_Block2(n*8,n*4)
        self.u3 = UpSample2d(n*4,n*2)
        self.c8 = Conv_Block2(n*4,n*2)
        self.u4 = UpSample2d(n*2,n)
        self.c9 = Conv_Block2(n*2,n)
        self.c10 = Conv_Block2(n, n//2)
        self.out = nn.Conv2d(n, 1, 3, 1, 1)
        self.edge_model = edge()
        self.edge_model.load_state_dict(torch.load(edge_model_path))
        self.edge_model.eval()
        self.c11 = Conv_Block2(1, n )
        self.c12 = Conv_Block2(n,n*2)
        self.c13 = Conv_Block2(n*2, n*4)
        self.c14 = Conv_Block2(n*4, n*8)

        self.fusion_module1 = Fusion(in_channels=n)
        self.fusion_module2 = Fusion(in_channels=n*2)
        self.fusion_module3 = Fusion(in_channels=n*4)
        self.fusion_module4 = Fusion(in_channels=n*8)


        self.d=DownSample2d(n,n)
        self.d5 = DownSample2d(n *2, n*2 )
        self.d6 = DownSample2d(n *4, n *4)
        self.d7= DownSample2d(n *8 , n *8)
    def forward(self, x):
        edge_256=self.edge_model(x)
        R1 = self.c1(torch.cat((x, edge_256), dim=1))
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        edge_256 = self.c11(edge_256)
        edge_128 = self.d(edge_256)
        edge_128 = self.c12(edge_128)
        edge_64 = self.d5(edge_128)
        edge_64 = self.c13(edge_64)
        edge_32 = self.d6(edge_64)
        edge_32 = self.c14(edge_32)

        x_256 = self.fusion_module1(R1, edge_256)
        x_128 = self.fusion_module2(R2, edge_128)
        x_64 = self.fusion_module3(R3, edge_64)
        x_32 = self.fusion_module4(R4, edge_32)

        O1 = self.c6(self.u1(R5, x_32))
        O2 = self.c7(self.u2(O1, x_64))
        O3 = self.c8(self.u3(O2, x_128))
        O4 = self.c9(self.u4(O3, x_256))
        return self.out(O4)

class ResidualBlock2d(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2d, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = torch.nn.functional.relu(self.conv1(x))
        y = self.conv2(y)
        return torch.nn.functional.relu(y+x)

class ResSegNet(nn.Module):
    def __init__(self,n=8):
        super(ResSegNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, n, 3,1,1),
            nn.BatchNorm2d(n),
            nn.ReLU(),

            ResidualBlock2d(n),
            nn.Conv2d(n, n*2,2, 2, 0),

            ResidualBlock2d(n*2),
            nn.Conv2d(n*2, n*4,2, 2, 0),

            ResidualBlock2d(n*4),
            nn.Conv2d(n*4, n*8,2, 2, 0),

            ResidualBlock2d(n*8),
            nn.Conv2d(n*8, n*16, 2, 2, 0),

            ResidualBlock2d(n*16),
            nn.Conv2d(n*16, n*16, 2, 2, 0),

            nn.ConvTranspose2d(n*16, n*16, 2, 2, 0),
            ResidualBlock2d(n*16),

            nn.ConvTranspose2d(n*16,n*8,2,2,0),
            ResidualBlock2d(n*8),

            nn.ConvTranspose2d(n*8, n*4, 2, 2, 0),
            ResidualBlock2d(n*4),

            nn.ConvTranspose2d(n*4, n*2, 2, 2, 0),
            ResidualBlock2d(n*2),

            nn.ConvTranspose2d(n*2, n, 2, 2, 0),
            ResidualBlock2d(n),

            nn.Conv2d(n, 1,kernel_size=3, padding=1),
            nn.Sigmoid()


        )
    def forward(self,x):
        x = self.model(x)

        return x

class DualResSegNet(nn.Module):
    def __init__(self, n=8,edge_model_path='edge.pth'):
        super(DualResSegNet, self).__init__()
        self.edge_model = edge()
        self.edge_model.load_state_dict(torch.load(edge_model_path))
        self.edge_model.eval()

        self.conv1 = nn.Conv2d(2, n*1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(n*1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.res2 = ResidualBlock2d(n * 1)
        self.res3 = ResidualBlock2d(n * 2)
        self.res4 = ResidualBlock2d(n * 4)
        self.res5 = ResidualBlock2d(n * 8)
        self.res6 = ResidualBlock2d(n * 16)
        self.conv7 = nn.Conv2d(n, 1, kernel_size=3, padding=1)
        self.c11 = Conv_Block2(1, n // 2)
        self.c12 = Conv_Block2(n // 2, n)
        self.c13 = Conv_Block2(n, n * 2)
        self.c14 = Conv_Block2(n * 2, n * 4)
        self.c15 = Conv_Block2(n * 4, n * 8)
        self.conv9 = Conv_Block2(n * 16, n * 8)
        self.conv10 = Conv_Block2(n * 8, n * 4)
        self.conv11 = Conv_Block2(n * 4, n * 2)
        self.conv12 = Conv_Block2(n * 2, n * 1)
        self.conv13 = Conv_Block2(n * 1, n // 2)

        self.down1 = nn.Conv2d(n*1, n*2, 2, 2, 0)
        self.down2 = nn.Conv2d(n * 2, n * 4, 2, 2, 0)
        self.down3 = nn.Conv2d(n * 4, n * 8, 2, 2, 0)
        self.down4 = nn.Conv2d(n * 8, n * 16, 2, 2, 0)
        self.down5 = nn.Conv2d(n * 16, n * 16, 2, 2, 0)
        self.d = DownSample2d(n // 2, n // 2)
        self.d5 = DownSample2d(n, n)
        self.d6 = DownSample2d(n * 2, n * 2)
        self.d7 = DownSample2d(n * 4, n * 4)
        self.d8 = DownSample2d(n * 8, n * 8)

        self.up1 = nn.ConvTranspose2d(n*16, n*16, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(n * 16, n * 8, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(n * 8, n * 4, 2, 2, 0)
        self.up4 = nn.ConvTranspose2d(n * 4, n * 2, 2, 2, 0)
        self.up5 = nn.ConvTranspose2d(n * 2, n * 1, 2, 2, 0)

    def forward(self, x):
        edge_256_ = self.edge_model(x)
        edge_256 = self.c11(edge_256_)

        edge_128 = self.d(edge_256)
        edge_128 = self.c12(edge_128)

        edge_64 = self.d5(edge_128)
        edge_64 = self.c13(edge_64)

        edge_32 = self.d6(edge_64)
        edge_32 = self.c14(edge_32)

        edge_16 = self.d7(edge_32)
        edge_16 = self.c15(edge_16)

        edge_8 = self.d8(edge_16)
        edge_8 = self.res5(edge_8)


        x1 = self.relu(self.bn(self.conv1(torch.cat((x, edge_256_), dim=1))))
        x2 = self.down1(self.res2(x1))
        x3 = self.down2(self.res3(x2))
        x4 = self.down3(self.res4(x3))
        x5 = self.down4(self.res5(x4))
        x6 = self.down5(self.res6(x5))
        x7 = self.res6(self.up1(x6))
        x8 = self.res5(self.up2(x7))
        x9 = self.res4(self.up3(x8))
        x10 = self.res3(self.up4(x9))
        x11 = self.res2(self.up5(x10))

        x6 = self.conv9(x6)
        x7 = self.res6(self.up1(torch.cat((x6, edge_8), dim=1)))
        x7 = self.conv9(x7)
        x8 = self.res5(self.up2(torch.cat((x7, edge_16), dim=1)))
        x8 = self.conv10(x8)
        x9 = self.res4(self.up3(torch.cat((x8, edge_32), dim=1)))
        x9 = self.conv11(x9)
        x10 = self.res3(self.up4(torch.cat((x9, edge_64), dim=1)))
        x10 = self.conv12(x10)
        x11 = self.res2(self.up5(torch.cat((x10, edge_128), dim=1)))
        x11 = self.conv13(x11)
        x12 = self.sigmoid(self.conv7(torch.cat((x11, edge_256), dim=1)))

        return x12

class RB2d(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(RB2d, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = torch.nn.functional.relu(self.conv1(x))
        y = self.conv2(y)

        return torch.nn.functional.relu(y + x)


class ResUNet(nn.Module):
    def __init__(self,n=8):
        super(ResUNet, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1,n,3,1,1),RB2d(n, n))
        self.d1 = DownSample2d(n, n*2)
        self.c2 = nn.Sequential(RB2d(n*2, n*2))
        self.d2 = DownSample2d(n*2, n*4)
        self.c3 = nn.Sequential(RB2d(n*4, n*4))
        self.d3 = DownSample2d(n*4, n*8)
        self.c4 = nn.Sequential(RB2d(n*8, n*8))
        self.d4 = DownSample2d(n*8, n*16)
        self.c5 = nn.Sequential(RB2d(n*16, n*16))
        self.u1 = UpSample2d(n*16, n*8)
        self.c6 = nn.Sequential(RB2d(n*16, n*16),nn.Conv2d(n*16,n*8,1,1,0))
        self.u2 = UpSample2d(n*8, n*4)
        self.c7 = nn.Sequential(RB2d(n*8, n*8),nn.Conv2d(n*8,n*4,1,1,0))
        self.u3 = UpSample2d(n*4, n*2)
        self.c8 = nn.Sequential(RB2d(n*4, n*4),nn.Conv2d(n*4,n*2,1,1,0))
        self.u4 = UpSample2d(n*2, n)
        self.c9 = nn.Sequential(RB2d(n*2, n*2),nn.Conv2d(n*2,n,1,1,0))
        self.out = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)

class DualResUNet(nn.Module):
    def __init__(self,n=8,edge_model_path='edge.pth'):
        super(DualResUNet, self).__init__()
        self.edge_model = edge()
        self.edge_model.load_state_dict(torch.load(edge_model_path))
        self.edge_model.eval()

        self.c1 = nn.Sequential(nn.Conv2d(2, n, 3, 1, 1), RB2d(n, n))
        self.d1 = DownSample2d(n, n * 2)
        self.c2 = nn.Sequential(RB2d(n * 2, n * 2))
        self.d2 = DownSample2d(n * 2, n * 4)
        self.c3 = nn.Sequential(RB2d(n * 4, n * 4))
        self.d3 = DownSample2d(n * 4, n * 8)
        self.c4 = nn.Sequential(RB2d(n * 8, n * 8))
        self.d4 = DownSample2d(n * 8, n * 16)
        self.c5 = nn.Sequential(RB2d(n * 16, n * 16))
        self.u1 = UpSample2d(n * 16, n * 8)
        self.c6 = nn.Sequential(RB2d(n * 16, n * 16), nn.Conv2d(n * 16, n * 8, 1, 1, 0))
        self.u2 = UpSample2d(n * 8, n * 4)
        self.c7 = nn.Sequential(RB2d(n * 8, n * 8), nn.Conv2d(n * 8, n * 4, 1, 1, 0))
        self.u3 = UpSample2d(n * 4, n * 2)
        self.c8 = nn.Sequential(RB2d(n * 4, n * 4), nn.Conv2d(n * 4, n * 2, 1, 1, 0))
        self.u4 = UpSample2d(n * 2, n)
        self.c9 = nn.Sequential(RB2d(n * 2, n * 2), nn.Conv2d(n * 2, n, 1, 1, 0))
        self.c10 = Conv_Block2(n, n//2)
        self.out = nn.Conv2d(n, 1, 3, 1, 1)

        self.fusion_module1 = Fusion(in_channels=n)
        self.fusion_module2 = Fusion(in_channels=n * 2)
        self.fusion_module3 = Fusion(in_channels=n * 4)
        self.fusion_module4 = Fusion(in_channels=n * 8)


        self.c11 = Conv_Block2(1, n )
        self.c12 = Conv_Block2(n,n*2)
        self.c13 = Conv_Block2(n*2, n*4)
        self.c14 = Conv_Block2(n*4, n*8)

        self.d=DownSample2d(n,n)
        self.d5 = DownSample2d(n*2 , n*2 )
        self.d6 = DownSample2d(n *4, n *4)
        self.d7= DownSample2d(n *8 , n *8)
    def forward(self, x):
        edge_256=self.edge_model(x)
        R1 = self.c1(torch.cat((x, edge_256), dim=1))
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))


        edge_256 = self.c11(edge_256)
        edge_128 = self.d(edge_256)
        edge_128 = self.c12(edge_128)
        edge_64 = self.d5(edge_128)
        edge_64 = self.c13(edge_64)
        edge_32 = self.d6(edge_64)
        edge_32 = self.c14(edge_32)

        x_256 = self.fusion_module1(R1, edge_256)
        x_128 = self.fusion_module2(R2, edge_128)
        x_64 = self.fusion_module3(R3, edge_64)
        x_32 = self.fusion_module4(R4, edge_32)

        O1 = self.c6(self.u1(R5, x_32))
        O2 = self.c7(self.u2(O1, x_64))
        O3 = self.c8(self.u3(O2, x_128))
        O4 = self.c9(self.u4(O3, x_256))
        return self.out(O4)


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()

        # 加载预训练的 MobileNetV2 模型
        model = mobilenet_v2(pretrained=pretrained)

        # 修改第一层卷积以支持单通道输入
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.features = model.features[:-1]
        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        else:
            raise ValueError('downsample_factor is {}, but must be one of 8 and 16'.format(downsample_factor))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv2dNormActivation') != -1:
            if m[0].stride == (2, 2):
                m[0].stride = (1, 1)
                if m[0].kernel_size == (3, 3):
                    m[0].dilation = (dilate // 2, dilate // 2)
                    m[0].padding = (dilate // 2, dilate // 2)
            else:
                if m[0].kernel_size == (3, 3):
                    m[0].dilation = (dilate, dilate)
                    m[0].padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rate=1, bn_momentum=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0, dilation=rate),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=6 * rate, dilation=6 * rate),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=12 * rate, dilation=12 * rate),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=18 * rate, dilation=18 * rate),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.branch5_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.branch5_bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.size()

        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, dim=2, keepdim=True)
        global_feature = torch.mean(global_feature, dim=3, keepdim=True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, size=(h, w), mode='bilinear', align_corners=True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLabV3plus(nn.Module):
    def __init__(self, backbone='mobilenet', pretrained=True, downsample_factor=16):
        super(DeepLabV3plus, self).__init__()
        if backbone == 'mobilenet':
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - {}, Use mobilenet'.format(backbone))

        self.aspp = ASPP(
            in_channels=in_channels,
            out_channels=256,
            rate=16 // downsample_factor
        )

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.cls_conv = nn.Conv2d(256, 1, kernel_size=(1, 1))  # 修改输出通道数为1

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        low_level_feature, x = self.backbone(x)
        x = self.aspp(x)
        low_level_feature = self.shortcut_conv(low_level_feature)

        x = F.interpolate(x, size=(low_level_feature.size(2), low_level_feature.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_feature), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
