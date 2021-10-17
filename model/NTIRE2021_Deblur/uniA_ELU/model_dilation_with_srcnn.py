"""AtrousNet with SRCNN
"""
import torch
import torch.nn as nn
from model.NTIRE2021_Deblur.uniA_ELU.wavelet_SRCNN_remix import SRCNN_Wide
from model.NTIRE2021_Deblur.uniA_ELU.layerlib_stage1 import Conv2D_ReflectPad, AtrousBlockPad2


class AtrousNet_billinear_Wide_dilated_srcnn(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=20, max_global_stride=8, pad_to_fit_global_stride=True,
                 d_mult=32,
                 activation=nn.ELU(alpha=1.0, inplace=True),
                 atrousDim=[[1, 2, 4, 8],[1, 3, 5, 7]]):
        super(AtrousNet_billinear_Wide_dilated_srcnn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.max_global_stride = max_global_stride
        self.pad_to_fit_global_stride = pad_to_fit_global_stride
        self.d_mult = d_mult
        self.activation = activation
        self.srcnn = SRCNN_Wide(self.d_mult * 6, self.d_mult * 7)

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
        self.upsampling_layers.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        # self.upsampling_layers.append(nn.ELU(alpha=1.0, inplace=True))
        self.upsampling_layers = nn.Sequential(*self.upsampling_layers)

        self.output_layer = []
        # modify output later from 3 * d_multi to 7 * multi
        self.output_layer.append(Conv2D_ReflectPad(in_channels=self.d_mult * 6,
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

        idx = len(downs) - 1
        net = torch.cat((net, downs[idx]), 1)

        # use the srcnn
        srcnn_output = self.srcnn(net)
        net_add = net + srcnn_output

        net = self.upsampling_layers[0](net_add)

        for x in range(len(self.output_layer)):
            net = self.output_layer[x](net)

        output = input_data  + net

        return output


# build the srcnn output for vgg loss
class AtrousNet_billinear_Wide_dilated_srcnn_output(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=20, max_global_stride=8, pad_to_fit_global_stride=True,
                 d_mult=32,
                 activation=nn.ELU(alpha=1.0, inplace=True),
                 atrousDim=[[1, 2, 4, 8],[1, 3, 5, 7]],
                 srcnn_add=True,
                 srcnn_smooth=False):
        super(AtrousNet_billinear_Wide_dilated_srcnn_output, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.max_global_stride = max_global_stride
        self.pad_to_fit_global_stride = pad_to_fit_global_stride
        self.d_mult = d_mult
        self.activation = activation
        self.srcnn = SRCNN_Wide(self.d_mult * 6, self.d_mult * 7)
        self.srcnn_add = srcnn_add
        self.srcnn_smooth = srcnn_smooth

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
        self.upsampling_layers.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        # self.upsampling_layers.append(nn.ELU(alpha=1.0, inplace=True))
        self.upsampling_layers = nn.Sequential(*self.upsampling_layers)

        self.output_layer = []
        # modify output later from 3 * d_multi to 7 * multi
        self.output_layer.append(Conv2D_ReflectPad(in_channels=self.d_mult * 6,
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

        idx = len(downs) - 1
        net = torch.cat((net, downs[idx]), 1)
        net = self.upsampling_layers[0](net)

        srcnn_output = self.srcnn(net)

        if self.srcnn_add:
            net += srcnn_output

        for x in range(len(self.output_layer)):
            net = self.output_layer[x](net)

        # srcnn output smmoth
        if self.srcnn_smooth:
            for x in range(len(self.output_layer)):
                srcnn_output = self.output_layer[x](srcnn_output)
            srcnn_output += input_data

        output = input_data + net

        return output, srcnn_output


if __name__ == "__main__":
    device = "cuda"
    net = AtrousNet_billinear_Wide_dilated_srcnn_output(in_channels=3,
                    out_channels=3)
    image = torch.rand(1, 3, 320, 320)
    image = image.to(device)
    net = net.to(device)
    print(net)
    with torch.no_grad():
        output = net(image)
        print(output[1].shape)
