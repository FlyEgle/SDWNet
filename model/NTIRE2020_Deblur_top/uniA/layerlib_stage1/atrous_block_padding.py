import torch
import torch.nn as nn
from .convolution import *


class AtrousBlockPad2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_bias, use_scale, activation,
                 needs_projection = False,atrousBlock=[1, 2, 4, 8]):
        super(AtrousBlockPad2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.activation = activation
        self.atrousBlocks = atrousBlock
        # self.needs_projection = self.stride > 1
        self.dims_match = self.in_channels != self.out_channels
        # self.needs_projection = self.needs_projection or self.dims_match
        self.needs_projection = needs_projection

        if self.needs_projection:
            self.projection = Conv2D_ReflectPad(in_channels=self.in_channels,
                                                out_channels=self.out_channels,
                                                kernel_size=1,
                                                stride=self.stride,
                                                use_bias=self.use_bias,
                                                act=self.activation)
        self.atrous_layers = []

        for i in range(4):
            self.atrous_layers.append(AtrousConv2D_ReflectPad(in_channels=self.out_channels,
                                                              out_channels=int(self.out_channels / 2),
                                                              kernel_size=self.kernel_size,
                                                              stride=self.stride,
                                                              dilation=atrousBlock[i],
                                                              use_bias=self.use_bias,
                                                              use_scale=self.use_scale,
                                                              act=self.activation))
        self.atrous_layers = nn.Sequential(*self.atrous_layers)

        self.conv1 = Conv2D_ReflectPad(in_channels=self.out_channels * 2,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       use_bias=self.use_bias,
                                       use_scale=self.use_scale,
                                       act=self.activation
                                       )

    def forward(self, input):
        if self.needs_projection:
            input = self.projection(input)

        x1 = self.atrous_layers[0](input)
        x2 = self.atrous_layers[1](input)
        x3 = self.atrous_layers[2](input)
        x4 = self.atrous_layers[3](input)

        x = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.conv1(x)

        return input + x5


if __name__ == "__main__":
    model = AtrousBlockPad2(in_channels=32, out_channels=32, kernel_size=3, stride=1,use_bias = True, use_scale = True,
                            activation=nn.ReLU(True))
    if torch.cuda.is_available():
        image = torch.rand(16, 32, 64, 64)
        output = model.forward(image)
        print(output.shape)
