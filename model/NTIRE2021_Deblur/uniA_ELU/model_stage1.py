import torch
import torch.nn as nn
from model.NTIRE2021_Deblur.uniA_ELU.layerlib_stage1 import Conv2D_ReflectPad, AtrousBlockPad2


class AtrousNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=10, max_global_stride=8, pad_to_fit_global_stride=True,
                 d_mult=16,
                 activation=nn.ELU(alpha=1.0, inplace=True),
                 atrousDim=[1, 2, 4, 8]):
        super(AtrousNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.max_global_stride = max_global_stride
        self.pad_to_fit_global_stride = pad_to_fit_global_stride
        self.d_mult = d_mult
        self.activation = activation

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
                                                   atrousBlock=atrousDim,
                                                   needs_projection=True))
            else:
                self.blocks.append(AtrousBlockPad2(in_channels=self.d_mult * 4,
                                                   out_channels=self.d_mult * 4,
                                                   kernel_size=3,
                                                   stride=1,
                                                   use_bias=True,
                                                   use_scale=True,
                                                   activation=self.activation,
                                                   atrousBlock=atrousDim))

        self.blocks = nn.Sequential(*self.blocks)

        self.upsampling_layers = []
        self.upsampling_layers.append(nn.ConvTranspose2d(in_channels=self.d_mult * 6,  # Error: should be 4
                                                         out_channels=self.d_mult * 2,
                                                         kernel_size=3,
                                                         stride=2,
                                                         padding=1,
                                                         output_padding=1,
                                                         bias=True))
        self.upsampling_layers.append(nn.ELU(alpha=1.0, inplace=True))
        self.upsampling_layers = nn.Sequential(*self.upsampling_layers)

        self.output_layer = []
        self.output_layer.append(Conv2D_ReflectPad(in_channels=self.d_mult * 3,
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

        for x in range(len(self.blocks)):
            net = self.blocks[x](net)

        for x in range(len(self.upsampling_layers)):
            idx = len(downs) - x - 1
            net = torch.cat((net, downs[idx]), 1)
            net = self.upsampling_layers[x](net)

        for x in range(len(self.output_layer)):
            net = self.output_layer[x](net)

        return input_data + net


if __name__ == "__main__":
    device = "cuda"
    net = AtrousNet(in_channels=3,
                    out_channels=3)
    image = torch.rand(1, 3, 768, 768)
    image = image.to(device)
    net = net.to(device)
    # print(net)
    with torch.no_grad():
        output = net(image).to(device)
        print(output.shape)
