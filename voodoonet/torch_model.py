"""
This module contains functions for generating deep learning models with Tensorflow and Keras.
"""
from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

import wandb
from voodoonet.utils import (
    VoodooOptions,
    VoodooTrainingOptions,
    calc_cm,
    metrics_to_dict,
    validation_metrics,
)


class VoodooNet(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size,
        options: VoodooOptions,
        training_options: VoodooTrainingOptions,
    ):
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

        # training paramters
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam
        self.lr = training_options.learning_rate
        self.lr_decay = training_options.learning_rate_decay
        self.lr_decay_step = training_options.learning_rate_decay_steps
        self.lr_scheduler = StepLR

        # Capture a dictionary of hyperparameters with config
        if self.options.use_wandb is True:
            self.wandb = wandb.init(project="voodoonet", name="v2", entity="krljhnsn")
            assert self.wandb is not None
            self.wandb.config.update(self.options.dict(), allow_val_change=True)

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

    def fwd_pass(self, X: Tensor, y: Tensor, train: bool = False) -> tuple[Tensor, Tensor]:
        if train:
            self.zero_grad()

        outputs = self(X)
        cm = calc_cm(outputs[:, 0], y[:, 0])
        loss = self.loss(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()  # type: ignore

        return cm, loss

    def optimize(
        self,
        X: Tensor,
        y: Tensor,
        X_test: Tensor,
        y_test: Tensor,
        batch_size: int = 256,
        epochs: int = 2,
        logging_frequency: int = 20,
    ) -> None:

        self.to(self.options.device)
        self.train()

        # what with weights and biases
        if self.options.use_wandb is True:
            assert self.wandb is not None
            self.wandb.watch(self, self.loss, log="all", log_freq=100)

        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)  # type: ignore
        self.lr_scheduler = self.lr_scheduler(
            self.optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay  # type: ignore
        )

        for epoch in range(epochs):
            iterator = tqdm(
                range(0, len(X), batch_size),
                ncols=100,
                unit=f" batches - epoch:{epoch + 1}/{epochs}",
            )

            # initialize batch confusion matrix entries
            batch_cm = Tensor([0, 0, 0, 0])
            batch_loss = Tensor([0])

            for i in iterator:
                batch_X = X[i : i + batch_size].to(self.options.device)
                batch_y = y[i : i + batch_size].to(self.options.device)
                if len(batch_y) < batch_size:
                    continue

                b_cm, b_loss = self.fwd_pass(batch_X, batch_y, train=True)

                batch_cm += b_cm
                batch_loss += b_loss

                if (i > 0) and i % logging_frequency == 0:
                    val_metrics, val_loss = self.validation(X_test, y_test)

                    batch_metrics = validation_metrics(batch_cm)
                    batch_loss = batch_loss / (i // batch_size)

                    if self.options.use_wandb is True:
                        assert self.wandb is not None
                        self.wandb.log(
                            {
                                "batch_metrics": metrics_to_dict(batch_metrics),
                                "batch_loss": batch_loss,
                                "val_metrics": metrics_to_dict(val_metrics),
                                "val_loss": val_loss,
                            }
                        )

            # advance lr schedular after epoch
            if self.options.use_wandb is True:
                assert self.wandb is not None
                self.wandb.log({"learning_rate": self.optimizer.param_groups[0]["lr"]})
            self.lr_scheduler.step()  # type: ignore

    def validation(self, X: Tensor, y: Tensor, batch_size: int = 256) -> tuple:
        iterator = tqdm(range(0, len(X), batch_size), ncols=100, unit=" batches - validation")

        # initialize batch confusion matrix entries
        val_cm = Tensor([0, 0, 0, 0])
        val_loss = Tensor([0])

        j = 0
        for j in iterator:
            test_batch_X = X[j : j + batch_size].to(self.options.device)
            test_batch_y = y[j : j + batch_size].to(self.options.device)

            with torch.inference_mode():
                v_cm, v_loss = self.fwd_pass(test_batch_X, test_batch_y)
                val_cm += v_cm
                val_loss += v_loss

        val_metrics = validation_metrics(val_cm)
        val_loss = val_loss / (j // batch_size)

        return val_metrics, val_loss

    def print_nparams(self) -> None:
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        pytorch_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"Total non-trainable parameters: {pytorch_total_params - pytorch_trainable_params:,d}"
        )
        print(f"    Total trainable parameters: {pytorch_trainable_params:_d}")
        print(f"             Total  parameters: {pytorch_total_params:_d}")

    def save(self, path: str, aux: dict) -> None:
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),  # type: ignore
            "auxiliary": aux,
        }

        torch.save(checkpoint, path)

        if self.options.use_wandb is True:
            assert self.wandb is not None
            self.wandb.save(path.replace(".pt", ".onnx"))


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
