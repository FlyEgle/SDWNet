import torch
import torch.nn as nn
from model.NTIRE2020_Deblur_top.AFN.SENet import SELayer


class AFB_0(nn.Module):
    def __init__(self, Channels, kSize=3):
        super(AFB_0, self).__init__()
        Ch = Channels
        self.conv1 = nn.Conv2d(Ch, Ch, kSize, padding=(kSize-1)//2, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(Ch, Ch, kSize, padding=(kSize-1)//2, stride=1)

    def forward(self, x):
        return x + self.relu(self.conv2(self.relu(self.conv1(x))))


class AFB_L1(nn.Module):
    def __init__(self, growRate0, kSize=3):
        super(AFB_L1, self).__init__()
        G0 = self.G0 = growRate0

        self.conv1 = AFB_0(G0)
        self.conv2 = AFB_0(G0)
        self.conv3 = AFB_0(G0)
        self.LFF = nn.Sequential(
            SELayer(G0+(self.G0)*2),
            nn.Conv2d(G0+(self.G0)*2, G0, 1, padding=0, stride=1),
        )

    def forward(self, x):
        res = []
        ox = x

        x = self.conv1(x)
        res.append(x)
        x = self.conv2(x)
        res.append(x)
        x = self.conv3(x)
        res.append(x)

        return self.LFF(torch.cat(res, 1)) + ox


class AFB_L2(nn.Module):
    def __init__(self, growRate0, kSize=3):
        super(AFB_L2, self).__init__()
        G0 = self.G0 = growRate0

        self.conv1 = AFB_L1(growRate0=G0)
        self.conv2 = AFB_L1(growRate0=G0)
        self.conv3 = AFB_L1(growRate0=G0)
        self.conv4 = AFB_L1(growRate0=G0)

        self.LFF = nn.Sequential(
            SELayer(G0+(self.G0)*3),
            nn.Conv2d(G0+(self.G0)*3, G0, 1, padding=0, stride=1),
        )

    def forward(self, x):
        res = []

        ox = x
        x = self.conv1(x)
        res.append(x)
        x = self.conv2(x)
        res.append(x)
        x = self.conv3(x)
        res.append(x)
        x = self.conv4(x)
        res.append(x)

        return self.LFF(torch.cat(res, 1)) + ox


class AFB_L3(nn.Module):
    def __init__(self, growRate0, kSize=3):
        super(AFB_L3, self).__init__()
        G0 = self.G0 = growRate0

        self.conv1 = AFB_L2(growRate0=G0)
        self.conv2 = AFB_L2(growRate0=G0)
        self.conv3 = AFB_L2(growRate0=G0)
        self.conv4 = AFB_L2(growRate0=G0)

        self.LFF = nn.Sequential(
            SELayer(G0+(self.G0)*3),
            nn.Conv2d(G0+(self.G0)*3, G0, 1, padding=0, stride=1),
        )

    def forward(self, x):
        res = []

        ox = x
        x = self.conv1(x)
        res.append(x)
        x = self.conv2(x)
        res.append(x)
        x = self.conv3(x)
        res.append(x)
        x = self.conv4(x)
        res.append(x)

        return self.LFF(torch.cat(res, 1)) + ox


class RDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, args=None):
        super(RDN, self).__init__()
        r = 2
        G0 = 128
        kSize = 3

        G = 64

        self.encoding1 = nn.Conv2d(
            in_channels, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.encoding2 = nn.Conv2d(
            G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.n_r = 3
        self.AFBs = nn.ModuleList()
        for i in range(self.n_r):
            self.AFBs.append(
                AFB_L3(growRate0=G0)
            )

        self.GFF = nn.Sequential(*[
            SELayer(self.n_r * G0),
            nn.Conv2d(self.n_r * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        if r == 2:
            self.decoding = nn.Sequential(*[
                nn.Conv2d(G0, G, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G, out_channels, kSize, padding=(kSize-1)//2, stride=1)
            ])
        

    def forward(self, x):
        ori = x
        x = ori

        f__1 = self.encoding1(x)
        x = self.encoding2(f__1)
        f__2 = x

        AFBs_out = []
        for i in range(self.n_r):
            x = self.AFBs[i](x)
            AFBs_out.append(x)

        x = self.GFF(torch.cat(AFBs_out, 1))
        x += f__2

        x = self.decoding(x)

        return x

class AFN(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.s2conv = nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1)
        self.rdn = RDN(64,3)
    
    def forward(self,input):
        x = input
        x = self.s2conv(x)
        x = self.rdn(x)
        x = x + input
        return x

if __name__ == "__main__":
    device = "cuda"
    net = RDN(in_channels=3,out_channels=3)
    image = torch.rand(1, 3, 320, 320)
    image = image.to(device)
    net = net.to(device)
    # print(net)
    with torch.no_grad():
        output = net(image).to(device)
        print(output.shape)
