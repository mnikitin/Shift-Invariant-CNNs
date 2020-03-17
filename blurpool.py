import mxnet as mx
from mxnet import nd

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class Downsample(nn.HybridBlock):
    def __init__(self, filt_size=3, stride=2, channels=None, pad_off=0,
                 context=mx.cpu(), **kwargs):
        super(Downsample, self).__init__(**kwargs)
        self.filt_size = filt_size
        assert self.filt_size in [1, 3, 5, 7]
        self.pad_off = pad_off
        self.pad_size = (filt_size - 1) // 2 + pad_off
        self.stride = stride
        self.channels = channels

        if self.filt_size == 1:
            filt = nd.array([1.0,])
        elif self.filt_size == 3:
            filt = nd.array([1.0, 2.0, 1.0])
        elif self.filt_size == 5:    
            filt = nd.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 7:    
            filt = nd.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        kernel = filt[:, None] * filt[None, :]
        kernel = kernel / nd.sum(kernel)
        kernel = kernel[None, None, :, :].repeat(channels, axis=0)

        with self.name_scope():
            self.pad = nn.ReflectionPad2D(self.pad_size)
            self.blur_conv = nn.Conv2D(channels=channels, kernel_size=self.filt_size,
                                       strides=self.stride, use_bias=False,
                                       groups=channels, in_channels=channels)
            self.blur_conv.initialize(ctx=context)
            self.blur_conv.weight.set_data(kernel)
            self.blur_conv.weight.grad_req = 'null'

    def hybrid_forward(self, F, x):
        if self.filt_size == 1:
            if self.pad_off > 0:
                x = self.pad(x)
            return x[:, :, ::self.stride, ::self.stride]
        else:
            x = self.pad(x)
            x = self.blur_conv(x)
            return x
