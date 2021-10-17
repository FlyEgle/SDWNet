import torch
from torch import nn
from torch.nn import functional as f
from model.NTIRE2021_Deblur.uniA_ELU.layerlib_stage1 import Conv2D_ReflectPad, AtrousBlockPad2
from model.NTIRE2021_Deblur.uniA_ELU.wavelet_SRCNN_remix import SRCNN_Wide
from torch import nn

def kaiming_init(module,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                # nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                # nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class AtrousNet_billinear_Wide_dilated_gcnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=20, max_global_stride=8, pad_to_fit_global_stride=True,
                 d_mult=32,
                 activation=nn.ELU(alpha=1.0, inplace=True),
                 atrousDim=[[1, 2, 4, 8],[1, 3, 5, 7]]):
        super(AtrousNet_billinear_Wide_dilated_gcnet, self).__init__()
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


class AtrousNet_billinear_ContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=10, max_global_stride=8, pad_to_fit_global_stride=True,
                 d_mult=32,
                 activation=nn.ELU(alpha=1.0, inplace=True),
                 atrousDim=[[1, 2, 4, 8],[1, 3, 5, 7]]):
        super(AtrousNet_billinear_ContextBlock, self).__init__()
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

        for x in range(len(self.output_layer)):
            net = self.output_layer[x](net)

        return input_data + net



if __name__ == "__main__":
    device = "cuda"
    net = AtrousNet_billinear_ContextBlock(in_channels=3,
                    out_channels=3)
    image = torch.rand(1, 3, 256, 256)
    image = image.to(device)
    net = net.to(device)
    with torch.no_grad():
        output = net(image).to(device)
        # print(output.shape)