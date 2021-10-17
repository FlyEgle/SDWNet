import torch
import torch.nn as nn
from model.NTIRE2021_Deblur.uniA_ELU.layerlib_stage1 import Conv2D_ReflectPad, AtrousBlockPad2


def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin

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
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
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

class AtrousNet_wavlet_remix(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=10, max_global_stride=8, pad_to_fit_global_stride=True,
                 d_mult=16,
                 activation=nn.ELU(alpha=1.0, inplace=True),
                 atrousDim=[[1, 2, 4, 8],[1, 3, 5, 7]]):
        super(AtrousNet_wavlet_remix, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.max_global_stride = max_global_stride
        self.pad_to_fit_global_stride = pad_to_fit_global_stride
        self.d_mult = d_mult
        self.activation = activation

        self.DWT = DWT()
        self.IDWT = IWT()
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
        self.wavelet_conv = Conv2D_ReflectPad(in_channels=self.d_mult,
                                                          out_channels=self.d_mult * 2,
                                                          kernel_size=3,
                                                          stride=1,
                                                          use_scale=True,
                                                          use_bias=True,
                                                          padding=1,
                                                          act=self.activation)
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
        # self.upsampling_layers.append(nn.ConvTranspose2d(in_channels=self.d_mult * 6,  # Error: should be 4
        #                                                  out_channels=self.d_mult * 2,
        #                                                  kernel_size=3,
        #                                                  stride=2,
        #                                                  padding=1,
        #                                                  output_padding=1,
        #                                                  bias=True))
        self.upsampling_layers.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        self.upsampling_layers.append(nn.ELU(alpha=1.0, inplace=True))
        self.upsampling_layers = nn.Sequential(*self.upsampling_layers)

        self.output_layer = []
        # modify output later from 3 * d_multi to 7 * multi
        self.output_layer.append(Conv2D_ReflectPad(in_channels=self.d_mult * 13,
                                                   out_channels=self.d_mult,
                                                   kernel_size=3,
                                                   stride=1,
                                                   use_bias=True,
                                                   use_scale=True,
                                                   padding='same',
                                                   act=self.activation))
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

        wavelet_down = self.DWT(downs[0])
        wavelet_down = self.wavelet_conv(wavelet_down)
        wavelet_net = wavelet_down

        for x in range(len(self.blocks)):
            net = self.blocks[x](net)

        for x in range(len(self.blocks)):
            wavelet_net = self.blocks[x](wavelet_net)

        wavelet_net = torch.cat((wavelet_down, wavelet_net), 1)

        for x in range(len(self.upsampling_layers)):
            idx = len(downs) - x - 1
            net = torch.cat((net, downs[idx]), 1)
            net = self.upsampling_layers[x](net)

        wavelet_net = self.upsampling_layers[-1](self.IDWT(wavelet_net))
        net = torch.cat((wavelet_net, net), 1)

        for x in range(len(self.output_layer)):
            net = self.output_layer[x](net)

        return input_data + net

if __name__ == "__main__":
    device = "cuda"
    net = AtrousNet_wavlet_remix(in_channels=3,
                    out_channels=3)
    image = torch.rand(1, 3, 256, 256)
    image = image.to(device)
    net = net.to(device)
    print(net)
    with torch.no_grad():
        output = net(image).to(device)
        print(output.shape)