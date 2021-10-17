"""
use the wavelt after the upsample layer to merge the result
"""
import torch
import torch.nn as nn
from model.NTIRE2021_Deblur.uniA_ELU.layerlib_stage1 import Conv2D_ReflectPad, AtrousBlockPad2
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_EfficientAttention import EfficientAttention
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_ContextBlock import ContextBlock


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
    out_batch, out_channel, out_height, out_width = int(in_batch / r ** 2), int(
        in_channel), r * in_height, r * in_width
    x1 = x[0:out_batch, :, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

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


class SRCNN(nn.Module):
    def __init__(self, num_channels, out_channels):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.DWT = DWT()
        self.IDWT = IWT()

    def forward(self, x):
        x = self.DWT(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.IDWT(x)

        return x


class AtrousNet_SRCNN_tail(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                num_blocks=10,
                max_global_stride=8,
                pad_to_fit_global_stride=True,
                d_mult=32,
                activation=nn.ELU(alpha=1.0, inplace=True),
                atrousDim=[[1, 2, 4, 8],[1, 3, 5, 7]],
                efficientattention=False,
                gcattention=False,
                ):
        super(AtrousNet_SRCNN_tail, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.max_global_stride = max_global_stride
        self.pad_to_fit_global_stride = pad_to_fit_global_stride
        self.d_mult = d_mult
        self.activation = activation
        self.efficientattention = efficientattention
        self.gcattention = gcattention

        self.downsampling_layers = []
        self.downsampling_layers.append(Conv2D_ReflectPad(in_channels=self.in_channels,
                                                          out_channels=self.d_mult,
                                                          kernel_size=7,
                                                          stride=1,
                                                          use_bias=True,
                                                          use_scale=True,
                                                          padding="same",
                                                          act=self.activation))
        self.downsampling_layers.append(Conv2D_ReflectPad(in_channels=self.d_mult,
                                                          out_channels=self.d_mult * 2,
                                                          kernel_size=3,
                                                          stride=2,
                                                          use_scale=True,
                                                          use_bias=True,
                                                          padding=1,
                                                          act=self.activation))
        self.downsampling_layers = nn.Sequential(*self.downsampling_layers)
        self.SRCNN = SRCNN(self.d_mult*7, self.out_channels)
        self.blocks = []
        for x in range(num_blocks):
            if x == 0:
                self.blocks.append(AtrousBlockPad2(in_channels=self.d_mult * 2,
                                                   out_channels=self.d_mult * 4,
                                                   kernel_size=3,
                                                   stride=1,
                                                   use_bias=True,
                                                   use_scale=True,
                                                   activation=self.activation,
                                                   atrousBlock=atrousDim[0],
                                                   needs_projection=True))
            elif x != num_blocks-1:
                self.blocks.append(AtrousBlockPad2(in_channels=self.d_mult * 4,
                                                   out_channels=self.d_mult * 4,
                                                   kernel_size=3,
                                                   stride=1,
                                                   use_bias=True,
                                                   use_scale=True,
                                                   activation=self.activation,
                                                   atrousBlock=atrousDim[0]))
            else:
                self.blocks.append(AtrousBlockPad2(in_channels=self.d_mult * 4,
                                                   out_channels=self.d_mult * 4,
                                                   kernel_size=3,
                                                   stride=1,
                                                   use_bias=True,
                                                   use_scale=True,
                                                   activation=self.activation,
                                                   atrousBlock=atrousDim[1]))

        self.blocks = nn.Sequential(*self.blocks)

        self.upsampling_layers = []
        self.upsampling_layers.append(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True))
        self.upsampling_layers.append(nn.ELU(alpha=1.0, inplace=True))
        self.upsampling_layers = nn.Sequential(*self.upsampling_layers)

        self.output_layer = []
        # modify output later from 3 * d_multi to 7 * multi
        self.output_layer.append(Conv2D_ReflectPad(in_channels=self.d_mult * 7,
                                                   out_channels=self.d_mult,
                                                   kernel_size=3,
                                                   stride=1,
                                                   use_bias=True,
                                                   use_scale=True,
                                                   padding='same',
                                                   act=self.activation))
        if self.efficientattention:
            self.output_layer.append(EfficientAttention(in_channels=self.d_mult, key_channels=self.d_mult, head_count=4, value_channels=self.d_mult))
        elif self.gcattention:
            self.output_layer.append(ContextBlock(inplanes=self.d_mult, ratio=0.25))
        self.output_layer.append(Conv2D_ReflectPad(in_channels=self.d_mult,
                                                   out_channels=self.out_channels,
                                                   kernel_size=3,
                                                   stride=1,
                                                   use_bias=True,
                                                   use_scale=True,
                                                   padding='same',
                                                   act=None))
        self.output_layer = nn.Sequential(*self.output_layer)

    def forward(self, input_data):
        downs = []
        net = input_data
        for x in range(len(self.downsampling_layers)):
            net = self.downsampling_layers[x](net)
            downs.append(net)

        for x in range(len(self.blocks)):
            net = self.blocks[x](net)

        for x in range(len(self.upsampling_layers)):
            idx = len(downs) - x - 1
            net = torch.cat((net, downs[idx]), 1)
            net = self.upsampling_layers[x](net)

        SRCNN_net = self.SRCNN(net)
        for x in range(len(self.output_layer)):
            net = self.output_layer[x](net)

        return input_data + net + SRCNN_net


if __name__ == "__main__":
    device = "cuda"
    net = AtrousNet_SRCNN_tail(in_channels=3,
                    out_channels=3)
    image = torch.rand(1, 3, 256, 256)
    image = image.to(device)
    net = net.to(device)
    with torch.no_grad():
        output = net(image).to(device)
        # print(output.shape)