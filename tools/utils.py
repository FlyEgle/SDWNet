import torch
import torch.nn as nn

import time
import numpy as np
import threading

'''
Mainly borrowed from: 
[1] https://github.com/kingsj0405/ciplab-NTIRE-2020
'''
### COMMON FUNCTIONS ###
def _rgb2ycbcr(img, maxVal=255):
    #    r = img[:,:,0]
    #    g = img[:,:,1]
    #    b = img[:,:,2]

    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    #    ycbcr = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 1:
        O = O / 255.0

    #    ycbcr[:,:,0] = ((T[0,0] * r) + (T[0,1] * g) + (T[0,2] * b) + O[0])
    #    ycbcr[:,:,1] = ((T[1,0] * r) + (T[1,1] * g) + (T[1,2] * b) + O[1])
    #    ycbcr[:,:,2] = ((T[2,0] * r) + (T[2,1] * g) + (T[2,2] * b) + O[2])

    t = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    #    print(np.all((ycbcr - ycbcr_) < 1/255.0/2.0))

    return ycbcr


def _load_img_array(path, color_mode='RGB', channel_mean=None, modcrop=[0, 0, 0, 0]):
    '''Load an image using PIL and convert it into specified color space,
    and return it as an numpy array.
    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    The code is modified from Keras.preprocessing.image.load_img, img_to_array.
    '''
    ## Load image
    from PIL import Image
    img = Image.open(path)
    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')

    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:, :, 0:1]

    ## To 0-1
    x *= 1.0 / 255.0

    if channel_mean:
        x[:, :, 0] -= channel_mean[0]
        x[:, :, 1] -= channel_mean[1]
        x[:, :, 2] -= channel_mean[2]

    if modcrop[0] * modcrop[1] * modcrop[2] * modcrop[3]:
        x = x[modcrop[0]:-modcrop[1], modcrop[2]:-modcrop[3], :]

    return x


def PSNR(y_true, y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(255. / rmse)


# MATLAB imresize function
# Key difference from other resize funtions is antialiasing when downsampling
# This function only for downsampling
def DownSample2DMatlab(tensor, scale, method='cubic', antialiasing=True, cuda=True):
    '''
    This gives same result as MATLAB downsampling
    tensor: 4D tensor [Batch, Channel, Height, Width],
            height and width must be divided by the denominator of scale factor
    scale: Even integer denominator scale factor only (e.g. 1/2,1/4,1/8,...)
           Or list [1/2, 1/4] : [V scale, H scale]
    method: 'cubic' as default, currently cubic supported
    antialiasing: True as default
    '''

    # For cubic interpolation,
    # Cubic Convolution Interpolation for Digital Image Processing, ASSP, 1981
    def cubic(x):
        absx = np.abs(x)
        absx2 = np.multiply(absx, absx)
        absx3 = np.multiply(absx2, absx)

        f = np.multiply((1.5 * absx3 - 2.5 * absx2 + 1), np.less_equal(absx, 1)) + \
            np.multiply((-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2), \
                        np.logical_and(np.less(1, absx), np.less_equal(absx, 2)))

        return f

    # Generate resize kernel (resize weight computation)
    def contributions(scale, kernel, kernel_width, antialiasing):
        if scale < 1 and antialiasing:
            kernel_width = kernel_width / scale

        x = np.ones((1, 1))

        u = x / scale + 0.5 * (1 - 1 / scale)

        left = np.floor(u - kernel_width / 2)

        P = int(np.ceil(kernel_width) + 2)

        indices = np.tile(left, (1, P)) + np.expand_dims(np.arange(0, P), 0)

        if scale < 1 and antialiasing:
            weights = scale * kernel(scale * (np.tile(u, (1, P)) - indices))
        else:
            weights = kernel(np.tile(u, (1, P)) - indices)

        weights = weights / np.expand_dims(np.sum(weights, 1), 1)

        save = np.where(np.any(weights, 0))
        weights = weights[:, save[0]]

        return weights

    # Resize along a specified dimension
    def resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights):  # , indices):
        if scale_v < 1 and antialiasing:
            kernel_width_v = kernel_width / scale_v
        else:
            kernel_width_v = kernel_width
        if scale_h < 1 and antialiasing:
            kernel_width_h = kernel_width / scale_h
        else:
            kernel_width_h = kernel_width

        # Generate filter
        f_height = np.transpose(weights[0][0:1, :])
        f_width = weights[1][0:1, :]
        f = np.dot(f_height, f_width)
        f = f[np.newaxis, np.newaxis, :, :]
        F = torch.from_numpy(f.astype('float32'))

        # Reflect padding
        i_scale_v = int(1 / scale_v)
        i_scale_h = int(1 / scale_h)
        pad_top = int((kernel_width_v - i_scale_v) / 2)
        if i_scale_v == 1:
            pad_top = 0
        pad_bottom = int((kernel_width_h - i_scale_h) / 2)
        if i_scale_h == 1:
            pad_bottom = 0
        pad_array = ([pad_bottom, pad_bottom, pad_top, pad_top])
        kernel_width_v = int(kernel_width_v)
        kernel_width_h = int(kernel_width_h)

        #
        tensor_shape = tensor.size()
        num_channel = tensor_shape[1]
        FT = nn.Conv2d(1, 1, (kernel_width_v, kernel_width_h), (i_scale_v, i_scale_h), bias=False)
        FT.weight.data = F
        if cuda:
            FT.cuda()
        FT.requires_grad = False

        # actually, we want 'symmetric' padding, not 'reflect'
        outs = []
        for c in range(num_channel):
            padded = nn.functional.pad(tensor[:, c:c + 1, :, :], pad_array, 'reflect')
            outs.append(FT(padded))
        out = torch.cat(outs, 1)

        return out

    if method == 'cubic':
        kernel = cubic

    kernel_width = 4

    if type(scale) is list:
        scale_v = float(scale[0])
        scale_h = float(scale[1])

        weights = []
        for i in range(2):
            W = contributions(float(scale[i]), kernel, kernel_width, antialiasing)
            weights.append(W)
    else:
        scale = float(scale)

        scale_v = scale
        scale_h = scale

        weights = []
        for i in range(2):
            W = contributions(scale, kernel, kernel_width, antialiasing)
            weights.append(W)

    # np.save('bic_x4_downsample_h.npy', weights[0])

    tensor = resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights)

    return tensor


def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)

    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear

    if reduce:
        return torch.mean(losses)
    else:
        return losses


def im2tensor(image, imtype=np.uint8, cent=1., factor=255. / 2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

