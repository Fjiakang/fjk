from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from typing import List
from typing import Optional




class BaseModule(nn.Module):
    """
    Implements the basic module.
    All other modules inherit from this one
    """
    def load_w(self, checkpoint_path):
        # type: (str) -> None
        """
        Loads a checkpoint into the state_dict.

        :param checkpoint_path: the checkpoint file to be loaded.
        """
        self.load_state_dict(torch.load(checkpoint_path))

    def __repr__(self):
        # type: () -> str
        """
        String representation
        """
        good_old = super(BaseModule, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)

        return good_old + '\n' + addition
      
    def __call__(self, *args, **kwargs):
         return super(BaseModule, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        # type: () -> int
        """
        Number of parameters of the model.
        """
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, 'mask'):
                n_parameters += torch.sum(p.mask).item()
            else:
                n_parameters += reduce(mul, p.shape)
        return int(n_parameters)






def residual_op(x, functions, bns, activation_fn):
    # type: (torch.Tensor, List[Module, Module, Module], List[Module, Module, Module], Module) -> torch.Tensor
    """
    Implements a global residual operation.

    :param x: the input tensor.
    :param functions: a list of functions (nn.Modules).
    :param bns: a list of optional batch-norm layers.
    :param activation_fn: the activation to be applied.
    :return: the output of the residual operation.
    """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # A-branch
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # B-branch
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)

    # Residual connection
    out = ha + hb
    return activation_fn(out)








class Encoder(BaseModule):
    """
    MNIST model encoder.
    """
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()
        
        self.input_shape = input_shape
        self.code_length = code_length

        c, h, w = input_shape
       
        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=3, channel_out=32, activation_fn=activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn),
        )
        self.deepest_shape = (64, h // 4, w // 4)

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=code_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """

        h = x
        h = self.conv(h)
        h = h.view(len(h), -1)
        o = self.fc(h)

        return o


class Decoder(BaseModule):
    """
    MNIST model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of MNIST samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=activation_fn),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h

        return o




class BaseBlock(BaseModule):
    """ Base class for all blocks. """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(BaseBlock, self).__init__()

        assert not (use_bn and use_bias), 'Using bias=True with batch_normalization is forbidden.'

        self._channel_in = channel_in
        self._channel_out = channel_out
        self._activation_fn = activation_fn
        self._use_bn = use_bn
        self._bias = use_bias

    def get_bn(self):
        # type: () -> Optional[Module]
        """
        Returns batch norm layers, if needed.
        :return: batch norm layers or None
        """
        return nn.BatchNorm2d(num_features=self._channel_out) if self._use_bn else None

    def forward(self, x):
        """
        Abstract forward function. Not implemented.
        """
        raise NotImplementedError


class DownsampleBlock(BaseBlock):
    """ Implements a Downsampling block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(DownsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=2, bias=use_bias)
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                padding=0, stride=2, bias=use_bias)

        # Batch Normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )


class UpsampleBlock(BaseBlock):
    """ Implements a Upsampling block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(UpsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5,
                                         padding=2, stride=2, output_padding=1, bias=use_bias)
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=1,
                                         padding=0, stride=2, output_padding=1, bias=use_bias)

        # Batch Normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )


class ResidualBlock(BaseBlock):
    """ Implements a Residual block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(ResidualBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)
        self.conv2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)

        # Batch Normalization layers
        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1, self.conv2, None],
            bns=[self.bn1, self.bn2, None],
            activation_fn=self._activation_fn
        )


class MaskedFullyConnection(BaseModule, nn.Linear):
    """
    Implements a Masked Fully Connection layer (MFC, Eq. 6).
    This is the autoregressive layer employed for the estimation of
    densities of image feature vectors.
    """
    def __init__(self, mask_type, in_channels, out_channels, *args, **kwargs):
        """
        Class constructor.

        :param mask_type: type of autoregressive layer, either `A` or `B`.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        """
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(MaskedFullyConnection, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone())

        # Build mask
        self.mask.fill_(0)
        for f in range(0 if mask_type == 'B' else 1, self.out_features // self.out_channels):
            start_row = f*self.out_channels
            end_row = (f+1)*self.out_channels
            start_col = 0
            end_col = f*self.in_channels if mask_type == 'A' else (f+1)*self.in_channels
            if start_col != end_col:
                self.mask[start_row:end_row, start_col:end_col] = 1

        self.weight.mask = self.mask

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input tensor.
        :return: the output of a MFC manipulation.
        """

        # Reshape
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(len(x), -1)

        # Mask weights and call fully connection
        self.weight.data *= self.mask
        o = super(MaskedFullyConnection, self).forward(x)

        # Reshape again
        o = o.view(len(o), -1, self.out_channels)
        o = torch.transpose(o, 1, 2).contiguous()

        return o

    def __repr__(self):
        # type: () -> str
        """
        String representation.
        """
        return self.__class__.__name__ + '(' \
               + 'mask_type=' + str(self.mask_type) \
               + ', in_features=' + str(self.in_features // self.in_channels) \
               + ', out_features=' + str(self.out_features // self.out_channels)\
               + ', in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', n_params=' + str(self.n_parameters) + ')'


class Estimator1D(BaseModule):
    """
    Implements an estimator for 1-dimensional vectors.
    1-dimensional vectors arise from the encoding of images.
    This module is employed in MNIST and CIFAR10 LSA models.
    Takes as input a latent vector and outputs cpds for each variable.
    """
    def __init__(self, code_length, fm_list, cpd_channels):
        # type: (int, List[int], int) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param fm_list: list of channels for each MFC layer.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(Estimator1D, self).__init__()

        self.code_length = code_length
        self.fm_list = fm_list
        self.cpd_channels = cpd_channels

        activation_fn = nn.LeakyReLU()

        # Add autoregressive layers
        layers_list = []
        mask_type = 'A'
        fm_in = 1
        for l in range(0, len(fm_list)):

            fm_out = fm_list[l]
            layers_list.append(
                MaskedFullyConnection(mask_type=mask_type,
                                      in_features=fm_in * code_length,
                                      out_features=fm_out * code_length,
                                      in_channels=fm_in, out_channels=fm_out)
            )
            layers_list.append(activation_fn)

            mask_type = 'B'
            fm_in = fm_list[l]

        # Add final layer providing cpd params
        layers_list.append(
            MaskedFullyConnection(mask_type=mask_type,
                                  in_features=fm_in * code_length,
                                  out_features=cpd_channels * code_length,
                                  in_channels=fm_in,
                                  out_channels=cpd_channels))

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of CPD estimates.
        """
        h = torch.unsqueeze(x, dim=1)  # add singleton channel dim
        h = self.layers(h)
        o = h

        return o
class LS_A(BaseModule):
    """
    LSA model for MNIST one-class classification.
    """
    def __init__(self,  input_shape, code_length, cpd_channels):

        """
        Class constructor.

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(LS_A, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.cpd_channels = cpd_channels

        # Build encoder
        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = Decoder(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape
        )

        # Build estimator
        self.estimator = Estimator1D(
            code_length=code_length,
            fm_list=[32, 32, 32, 32],
            cpd_channels=cpd_channels
        )

    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors and CPD estimates.
        """
        h = x
    
        # Produce representations
        z = self.encoder(h)

        # Estimate CPDs with autoregression
        z_dist = self.estimator(z)

        # Reconstruct x
        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z, z_dist


