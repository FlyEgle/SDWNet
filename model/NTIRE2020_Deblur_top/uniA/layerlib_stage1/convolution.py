import torch
import torch.nn as nn


# Here we removed padding and use pytorch's default
# from .padding import Padding

class ConvAftermath(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True, use_scale=True, norm=None, act=None):
        super(ConvAftermath, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.norm = norm
        self.act = act
        self.b = None
        self.s = None

    def forward(self, input):
        net = input
        if self.use_bias and self.b is not None:
            net = net + self.use_bias
        if self.use_scale and self.s is not None:
            net = net * self.s
        if self.norm is not None:
            net = self.norm(net)
        if self.act is not None:
            net = self.act(net)
        return net


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bias=True, use_scale=True, norm=None,
                 act=None,
                 padding='same'):
        super(Conv2D, self).__init__()
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rate = 1
        self.pad_flag = False
        if padding == 'same':
            self.padding = self.kernel_size // 2 if self.stride == 1 else 0
            self.pad_flag = True
        else:
            self.padding = padding

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              bias=False)
        self.conv_aftermath = ConvAftermath(in_channels=self.out_channels, out_channels=self.out_channels,
                                            use_bias=self.use_bias,
                                            use_scale=self.use_scale, norm=self.norm, act=self.act)

    def forward(self, input):
        if self.pad_flag and self.stride > 1:
            pad_total = 1 * (self.kernel_size - 1) - 1
            pad_begin = pad_total // 2
            pad_end = pad_total - pad_begin
            input = torch.nn.functional.pad(input, pad=(pad_begin, pad_end, pad_begin, pad_end))

        net = self.conv(input)
        return self.conv_aftermath(net)


class Conv2D_ReflectPad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bias=True, use_scale=True, norm=None,
                 act=None,
                 padding='same', padding_algorithm="reflect"):
        super(Conv2D_ReflectPad, self).__init__()
        self.padding = padding
        self.padding_algorithm = padding_algorithm
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rate = 1
        if padding == 'same':
            self.padding = self.kernel_size // 2 if self.strides == 1 else 0
            self.pad_flag = True
        else:
            self.padding = padding

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.strides,
                              bias=False,
                              padding=self.padding,
                              padding_mode=self.padding_algorithm)
        self.conv_aftermath = ConvAftermath(in_channels=self.out_channels,
                                            out_channels=self.out_channels,
                                            use_bias=self.use_bias,
                                            use_scale=self.use_scale,
                                            norm=self.norm,
                                            act=self.act)

    def forward(self, input):
        x = self.conv(input)
        y = self.conv_aftermath(x)
        return y


class AtrousConv2D(nn.Module):
    #TODO: Perform same padding
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, use_bias=True, use_scale=True,
                 norm=None,
                 act=None):
        super(AtrousConv2D, self).__init__()
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rate = 1
        self.pad_flag = False
        self.dilation = dilation
        self.padding = dilation

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              dilation = self.dilation,
                              stride=self.stride,
                              padding=self.padding,
                              bias=False)
        self.conv_aftermath = ConvAftermath(in_channels=self.out_channels, out_channels=self.out_channels,
                                            use_bias=self.use_bias,
                                            use_scale=self.use_scale, norm=self.norm, act=self.act)

    def forward(self, input):
        net = self.conv(input)
        return self.conv_aftermath(net)


class AtrousConv2D_ReflectPad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, use_bias=True, use_scale=True,
                 norm=None,
                 act=None):
        super(AtrousConv2D_ReflectPad, self).__init__()
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_flag = False
        self.dilation = dilation
        self.padding = dilation

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              dilation=self.dilation,
                              padding=self.padding,
                              padding_mode="reflect",
                              bias=False)
        self.conv_aftermath = ConvAftermath(in_channels=self.out_channels, out_channels=self.out_channels,
                                            use_bias=self.use_bias,
                                            use_scale=self.use_scale, norm=self.norm, act=self.act)

    def forward(self, input):
        net = self.conv(input)
        net = self.conv_aftermath(net)
        return net


if __name__ == "__main__":
    model = AtrousConv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=2,
                                    act=nn.ReLU(True))
    image = torch.rand(16, 32, 64, 64)
    output = model.forward(image)
    print(output.shape)

    # image = torch.rand(1, 256, 512, 512)
    # output = ac.forward(image)
    # print(output.shape)
