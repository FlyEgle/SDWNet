import torch
from torch import nn
from torch.nn import functional as f
from model.NTIRE2021_Deblur.uniA_ELU.layerlib_stage1 import Conv2D_ReflectPad, AtrousBlockPad2
from model.NTIRE2021_Deblur.uniA_ELU.wavelet_SRCNN_remix import SRCNN_Wide


class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = f.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class AtrousNet_billinear_EfficientAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=10, max_global_stride=8, pad_to_fit_global_stride=True,
                 d_mult=32,
                 activation=nn.ELU(alpha=1.0, inplace=True),
                 atrousDim=[[1, 2, 4, 8],[1, 3, 5, 7]]):
        super(AtrousNet_billinear_EfficientAttention, self).__init__()
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
        self.output_layer.append(EfficientAttention(in_channels=self.d_mult, key_channels=self.d_mult, head_count=4, value_channels=self.d_mult))
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


# attention + dilation + wavelet
class AtrousNet_billinear_Wide_dilated_wavelet_attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=20, max_global_stride=8, pad_to_fit_global_stride=True,
                 d_mult=32,
                 activation=nn.ELU(alpha=1.0, inplace=True),
                 atrousDim=[[1, 2, 4, 8],[1, 3, 5, 7]]):
        super(AtrousNet_billinear_Wide_dilated_wavelet_attention, self).__init__()
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
        self.output_layer.append(EfficientAttention(in_channels=self.d_mult, key_channels=self.d_mult, head_count=4, value_channels=self.d_mult))
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




if __name__ == "__main__":
    device = "cuda"
    net = AtrousNet_billinear_Wide_dilated_wavelet_attention(in_channels=3,
                    out_channels=3)
    image = torch.rand(1, 3, 256, 256)
    image = image.to(device)
    net = net.to(device)
    with torch.no_grad():
        output = net(image).to(device)
        # print(output.shape)