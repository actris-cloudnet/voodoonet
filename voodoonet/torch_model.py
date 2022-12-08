"""
This module contains functions for generating deep learning models with Tensorflow and Keras.
"""
from collections import OrderedDict

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from voodoonet.utils import VoodooOptions


class VoodooNet(nn.Module):
    def __init__(self, input_shape: torch.Size, options: VoodooOptions):
        """
        Defining a PyTorch model.

        Args:
            input_shape: Shape of the input tensor
            options

        """
        super().__init__()
        self.input_shape = input_shape
        self.options = options
        self.flatten = nn.Flatten()
        self.activation_fun = nn.Softmax
        self.convolution_network = self._define_cnn()
        self.dense_network = self._define_dense(dropout=0.0)

    def predict(self, x_test: Tensor, batch_size: int = 4096) -> Tensor:
        self.to(self.options.device)
        self.eval()
        pred = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(x_test), batch_size), ncols=100, unit=" batches"):
                batch_x = x_test[i : i + batch_size].to(self.options.device)
                pred.append(self(batch_x))
        return torch.cat(pred, 0)

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.convolution_network(tensor)
        tensor = self.flatten(tensor)
        tensor = self.dense_network(tensor)
        return tensor

    def _define_cnn(self) -> nn.Sequential:
        in_shape = self.input_shape[1]
        iterator = enumerate(
            zip(
                self.options.num_filters,
                self.options.kernel_sizes,
                self.options.stride_sizes,
                self.options.pad_sizes,
            )
        )
        conv_2d = OrderedDict()
        for i, (ifltrs, ikrn, istride, ipad) in iterator:
            conv_2d.update({f"conv2d_{i}": Conv2DUnit(in_shape, ifltrs, ikrn, istride, ipad)})
            in_shape = ifltrs
        return nn.Sequential(conv_2d)  # type: ignore

    def _define_dense(self, dropout: float) -> nn.Sequential:
        in_shape = self._flatten_conv()
        dense = OrderedDict()
        i = 0
        for i, inodes in enumerate(self.options.dense_layers):
            dense_unit = DenseUnit(in_shape, inodes, dropout)
            dense.update({f"dense_{i}": dense_unit})
            in_shape = inodes
        output_unit = OutputUnit(in_shape, self.options.output_shape, self.activation_fun)
        dense.update({f"dense_{i + 2}": output_unit})  # type: ignore
        return nn.Sequential(dense)  # type: ignore

    def _flatten_conv(self) -> int:
        tensor = torch.rand(((1,) + tuple(self.input_shape[1:])))
        tensor = self.convolution_network(tensor)
        tensor = self.flatten(tensor)
        return tensor.shape[1]


class Conv2DUnit(nn.Module):
    def __init__(
        self,
        in_shape: int,
        n_features: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_shape,
            n_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="circular",
        )
        self.bn = nn.BatchNorm2d(num_features=n_features)
        self.relu = nn.ELU()

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.conv(tensor)
        tensor = self.bn(tensor)
        tensor = self.relu(tensor)
        return tensor


class DenseUnit(nn.Module):
    def __init__(self, in_shape: int, n_nodes: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(in_shape, n_nodes)
        self.bn = nn.BatchNorm1d(num_features=n_nodes)
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.dense(tensor)
        tensor = self.bn(tensor)
        tensor = self.relu(tensor)
        tensor = self.dropout(tensor)
        return tensor


class OutputUnit(nn.Module):
    def __init__(self, in_shape: int, n_nodes: int, activation_fun: type):
        super().__init__()
        self.dense = nn.Linear(in_shape, n_nodes)
        self.activation = activation_fun(dim=1)

    def forward(self, tensor: Tensor) -> Tensor:
        output = self.dense(tensor)
        output = self.activation(output)
        return output
