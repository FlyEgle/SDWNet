import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def cropimage(x):
    H, W = x.size(2), x.size(3)
    x_l_top = x[:, :, 0:int(H/2), 0:int(W/2)]
    x_r_top = x[:, :, 0:int(H/2), int(W/2):W]
    x_l_bot = x[:, :, int(H/2):H, 0:int(W/2)]
    x_r_bot = x[:, :, int(H/2):H, int(W/2):W]
    return torch.cat((x_l_top, x_r_top, x_l_bot, x_r_bot), dim=0)

def reconstruct(x):
     out_batch = int(x.size(0)/4)
     x_l_top = x[0:out_batch, :, :, :]
     x_r_top = x[out_batch:out_batch * 2, :, :, :]
     x_l_bot = x[out_batch * 2:out_batch * 3, :, :, :]
     x_r_bot = x[out_batch * 3:out_batch * 4, :, :, :]
     x_top = torch.cat((x_l_top, x_r_top), dim=3)
     x_bot = torch.cat((x_l_bot, x_r_bot), dim=3)
     x_rec = torch.cat((x_top, x_bot), dim=2)
     return x_rec


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch / r ** 2), int(in_channel), r * in_height, r * in_width
    x1 = x[0:out_batch, :, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class crop(nn.Module):
    def __init__(self):
        super(crop, self).__init__()

    def forward(self, x):
        return cropimage(x)

class crop_re(nn.Module):
    def __init__(self):
        super(crop_re, self).__init__()

    def forward(self, x):
        return reconstruct(x)

class SRCNN_Wide(nn.Module):
    def __init__(self, num_channels=3, output_channel=3):
        super(SRCNN_Wide, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, output_channel, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(output_channel, 64, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(64, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.DWT = DWT()
        self.IDWT = IWT()
        self.crop = crop()
        self.rec_crop = crop_re()

    def forward(self, x):
        x = self.DWT(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.IDWT(x)

        crop = self.crop(x)
        crop = self.relu(self.conv1(crop))
        crop = self.relu(self.conv2(crop))
        crop = self.conv3(crop)

        y = self.rec_crop(crop)

        return x + y


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.DWT = DWT()
        self.IDWT = IWT()
        self.crop = crop()
        self.rec_crop = crop_re()

    def forward(self, x):
        x = self.DWT(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.IDWT(x)

        crop = self.crop(x)
        crop = self.relu(self.conv1(crop))
        crop = self.relu(self.conv2(crop))
        crop = self.conv3(crop)

        y = self.rec_crop(crop)

        return x + y



if __name__ == "__main__":
    device = "cuda"
    net = SRCNN(num_channels=3)
    image = torch.rand(1, 3, 256, 256)
    image = image.to(device)
    net = net.to(device)
    with torch.no_grad():
        output = net(image).to(device)
        print(output.shape)