##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import numpy  as np
from   numpy.random import RandomState
import torch
from   torch.autograd import Variable
import torch.nn.functional  as F
import torch.nn as nn
from   torch.nn.parameter  import Parameter
from   torch.nn import Module
from   Models.quaternion_ops import *
import math
import sys

class QuaternionTransposeConv(Module):
    r"""Applies a Quaternion Transposed Convolution (or Deconvolution) to the incoming data.
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride,
    #              dilatation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
    #              weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
    #              quaternion_format=False): I have changed operation='convolution2d' to operation only
    def __init__(self, in_channels, out_channels, kernel_size, stride, operation,
                 dilatation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='quaternion', seed=None, rotation=False,
                 quaternion_format=False):

        super(QuaternionTransposeConv, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.output_padding    = output_padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             = {'quaternion': quaternion_init,
                                  'unitary'   : unitary_init,
                                  'random'    : random_init}[self.weight_init]


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.out_channels, self.in_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):

        if self.rotation:
            return quaternion_transpose_conv_rotation(input, self.r_weight, self.i_weight,
                self.j_weight, self.k_weight, self.bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilatation, self.quaternion_format)
        else:
            return quaternion_transpose_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.output_padding,
                self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', dilation='       + str(self.dilation) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', operation='      + str(self.operation) + ')'



class QuaternionConv(Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride,operation,
    #              dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
    #              weight_init='quaternion', seed=None, rotation=False, quaternion_format=True, scale=False):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution1d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):


        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'




class QuaternionConv2d(Module):
    
    r"""Applies a Quaternion Convolution to the incoming data.
    """
    # call_count        = 0
    # def __init__(self, in_channels, out_channels, kernel_size, stride,operation,
    #              dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
    #              weight_init='quaternion', seed=None, rotation=False, quaternion_format=True, scale=False):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv2d, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale
        

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):


        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'



class QuaternionConv2dNoBias(Module):
    
    r"""Applies a Quaternion Convolution to the incoming data.
    """
    # call_count        = 0
    # def __init__(self, in_channels, out_channels, kernel_size, stride,operation,
    #              dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
    #              weight_init='quaternion', seed=None, rotation=False, quaternion_format=True, scale=False):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=False, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv2dNoBias, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale
        

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):


        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'



class QuaternionConv2dASM(Module):
    
    r"""Applies a Quaternion Convolution to the incoming data with ASM matrix
    """
    # call_count        = 0
    # def __init__(self, in_channels, out_channels, kernel_size, stride,operation,
    #              dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
    #              weight_init='quaternion', seed=None, rotation=False, quaternion_format=True, scale=False):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion_asm', seed=None, operation='convolution2d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv2dASM, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init,
                                     'quaternion_asm': quaternion_init_ASM}[self.weight_init]
        self.scale             = scale
        

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):


        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'





class QuaternionConv2dUni(Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride,operation,
    #              dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
    #              weight_init='quaternion', seed=None, rotation=False, quaternion_format=True, scale=False):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='unitary', seed=None, operation='convolution2d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv2dUni, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):


        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'



# class QuaternionMaxPool(Module):
#     def __init__(self) -> None:
#         super().__init__()












class QuaternionLinearAutograd(Module):
    r"""Applies a quaternion linear transformation to the incoming data. A custom
    Autograd function is call to drastically reduce the VRAM consumption. Nonetheless, computing
    time is also slower compared to QuaternionLinear().
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=False, quaternion_format=True, scale=False):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features       = in_features//4
        self.out_features      = out_features//4
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.scale    = scale

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel  = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init 
        self.seed = seed if seed is not None else np.random.randint(0,1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init, 'random': random_init}[self.weight_init]
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.rotation:
            return quaternion_linear_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias, self.quaternion_format, self.scale_param)
        else:
            return quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', rotation='       + str(self.rotation) \
            + ', seed=' + str(self.seed) + ')'

class QuaternionLinear(Module):
    r"""Applies a quaternion linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='he', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinear, self).__init__()
        self.in_features  = in_features//4
        self.out_features = out_features//4
        self.r_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias     = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init    = weight_init
        self.seed           = seed if seed is not None else np.random.randint(0,1234)
        self.rng            = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input  = input.view(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'




def moving_average_update(statistic, curr_value, momentum):
    
    new_value = (1 - momentum) * statistic + momentum * curr_value
    
    return  new_value.data

class QuaternionBatchNorm2d(Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, gamma_init=1., beta_param=True, momentum=0.1):
        super(QuaternionBatchNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.eps = torch.tensor(1e-5)
        
        self.register_buffer('moving_var', torch.ones(1) )
        self.register_buffer('moving_mean', torch.zeros(4))
        self.momentum = momentum

    def reset_parameters(self):
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        # print(self.training)
        if self.training:
            quat_components = torch.chunk(input, 4, dim=1)
    
            r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
            
            mu_r = torch.mean(r)
            mu_i = torch.mean(i)
            mu_j = torch.mean(j)
            mu_k = torch.mean(k)
            mu = torch.stack([mu_r,mu_i, mu_j, mu_k], dim=0)
            # print('mu shape', mu.shape)
            
            delta_r, delta_i, delta_j, delta_k = r - mu_r, i - mu_i, j - mu_j, k - mu_k
    
            quat_variance = torch.mean(delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2)
            var = quat_variance
            denominator = torch.sqrt(quat_variance + self.eps)
            
            # Normalize
            r_normalized = delta_r / denominator
            i_normalized = delta_i / denominator
            j_normalized = delta_j / denominator
            k_normalized = delta_k / denominator
    
            beta_components = torch.chunk(self.beta, 4, dim=1)
    
            # Multiply gamma (stretch scale) and add beta (shift scale)
            new_r = (self.gamma * r_normalized) + beta_components[0]
            new_i = (self.gamma * i_normalized) + beta_components[1]
            new_j = (self.gamma * j_normalized) + beta_components[2]
            new_k = (self.gamma * k_normalized) + beta_components[3]
    
            new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)
        

            # with torch.no_grad():
            self.moving_mean.copy_(moving_average_update(self.moving_mean.data, mu.data, self.momentum))
            self.moving_var.copy_(moving_average_update(self.moving_var.data, var.data, self.momentum))
                
            # print(var, self.moving_var)

            
            return new_input
        
        else:
            with torch.no_grad():
                # print(input.shape, self.moving_mean.shape)
                r,i,j,k = torch.chunk(input, 4, dim=1)
                quaternions = [r,i,j,k]
                output = []
                denominator = torch.sqrt(self.moving_var + self.eps)
                beta_components = torch.chunk(self.beta, 4, dim=1)
                # print(torch.tensor(quaternions).shape)
                # print(quaternions[0].shape, self.moving_mean.shape, self.moving_var.shape, torch.squeeze(self.beta).shape)
                for q in range(4):
                    new_quat = self.gamma * ( (quaternions[q] - self.moving_mean[q]) / denominator ) + beta_components[q]
                    output.append(new_quat)
                output = torch.cat(output, dim=1)

                return  output 

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma.shape) \
               + ', beta=' + str(self.beta.shape) \
               + ', eps=' + str(self.eps.shape) + ')'
               
       