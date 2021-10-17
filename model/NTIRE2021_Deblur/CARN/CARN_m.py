import torch
import torch.nn as nn
import model.NTIRE2021_Deblur.CARN.ops as ops


class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.EResidualBlock(64, 64, group=group)
        self.c1 = ops.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, group = 4):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group

        self.entry = nn.Conv2d(self.in_channels, 64, 3, 1, 1)

        self.b1 = Block(64, 64, group=self.group)
        self.b2 = Block(64, 64, group=self.group)
        self.b3 = Block(64, 64, group=self.group)
        self.c1 = ops.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64 * 4, 64, 1, 1, 0)

        self.exit = nn.Conv2d(64, self.out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.exit(o3)

        return out


if __name__ == "__main__":
    device = "cuda"
    net = Net(3,3)
    image = torch.rand(16, 3, 224, 224)
    image = image.to(device)
    net = net.to(device)
    # print(net)
    with torch.no_grad():
        output = net(image).to(device)
        print(output.shape)