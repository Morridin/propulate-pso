#!/usr/bin/env python3
"""
This is the python script with which I conducted my NAS experiments.

It is derived from the torch example in the tutorials folder.
"""
import logging
import random
import sys
import time
from typing import Tuple, Dict, Union

import numpy as np
import torch
from lightning.pytorch import loggers
from mpi4py import MPI
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

from propulate import set_logger_config, Propulator
from propulate.propagators import Conditional, pso, Propagator

num_generations = int(sys.argv[2])
pop_size = 2 * MPI.COMM_WORLD.size
GPUS_PER_NODE = int(sys.argv[3])
log_path = "tbm/" if len(sys.argv) < 6 else sys.argv[5]

limits = {
    "conv_layers": (2.0, 10.0),
    "lr": (0.01, 0.0001),
    "epochs": (2.0, float(sys.argv[4])),
}


class Net(LightningModule):
    def __init__(self, convlayers: int, activation, lr: float, loss_fn):
        super(Net, self).__init__()

        self.lr = lr
        self.loss_fn = loss_fn
        self.best_accuracy = 0.0
        layers = []
        layers += [
            nn.Sequential(nn.Conv2d(1, 10, kernel_size=3, padding=1), activation()),
        ]
        layers += [
            nn.Sequential(nn.Conv2d(10, 10, kernel_size=3, padding=1), activation())
            for _ in range(convlayers - 1)
        ]

        self.fc = nn.Linear(7840, 10)
        self.conv_layers = nn.Sequential(*layers)

        self.val_acc = Accuracy("multiclass", num_classes=10)
        self.train_acc = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
           data sample

        Returns
        -------
        torch.Tensor
            The model's predictions for input data sample
        """
        b, c, w, h = x.size()
        x = self.conv_layers(x)
        x = x.view(b, 10 * 28 * 28)
        x = self.fc(x)
        return x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Calculate loss for training step in Lightning train loop.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor]
               input batch
        batch_idx: int
                   batch index

        Returns
        -------
        torch.Tensor
            training loss for input batch
        """
        x, y = batch
        pred = self(x)
        loss_val = self.loss_fn(pred, y)
        self.log("train loss", loss_val)
        train_acc_val = self.train_acc(torch.nn.functional.softmax(pred, dim=-1), y)
        self.log("train_ acc", train_acc_val)
        return loss_val

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Calculate loss for validation step in Lightning validation loop during training.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor]
               current batch
        batch_idx: int
                   batch index

        Returns
        -------
        torch.Tensor
            validation loss for input batch
        """
        x, y = batch
        pred = self(x)
        loss_val = self.loss_fn(pred, y)
        val_acc_val = self.val_acc(torch.nn.functional.softmax(pred, dim=-1), y)
        self.log("val_loss", loss_val)
        self.log("val_acc", val_acc_val)
        return loss_val

    def configure_optimizers(self) -> torch.optim.SGD:
        """
        Configure optimizer.

        Returns
        -------
        torch.optim.sgd.SGD
            stochastic gradient descent optimizer
        """
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def on_validation_epoch_end(self):
        """
        Calculate and store the model's validation accuracy after each epoch.
        """
        val_acc_val: torch.Tensor = self.val_acc.compute()
        self.val_acc.reset()
        if val_acc_val.item() > self.best_accuracy:
            self.best_accuracy = val_acc_val.item()


def get_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST train and validation dataloaders.

    Parameters
    ----------
    batch_size: int
                batch size

    Returns
    -------
    DataLoader
        training dataloader
    DataLoader
        validation dataloader
    """
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    dl_root = f"{log_path}/data/rank{MPI.COMM_WORLD.rank:0>2}"
    train_loader = DataLoader(
        dataset=CIFAR10(
            download=True,
            root=dl_root,
            transform=data_transform,
        ),  # Use CIFAR-10 training dataset.
        batch_size=batch_size,  # Batch size
        shuffle=True,  # Shuffle data.
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        dataset=CIFAR10(
            root=dl_root, transform=data_transform, train=False
        ),  # Use CIFAR-10 testing dataset.
        shuffle=False,  # Do not shuffle data.
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
    )
    return train_loader, val_loader


def ind_loss(params: Dict[str, Union[int, float, str]]) -> float:
    """
    Loss function for evolutionary optimization with Propulate. Minimize the model's negative validation accuracy.

    Parameters
    ----------
    params: dict[str, int | float | str]]

    Returns
    -------
    float
        The trained model's negative validation accuracy
    """
    # Extract hyperparameter combination to test from input dictionary.
    conv_layers = int(np.round(params["conv_layers"]))  # Number of convolutional layers
    epochs = int(np.round(params["epochs"]))
    lr = params["lr"]  # Learning rate

    extra_loss: float = 0  # additional penalty loss, if PSO is bad-behaved.
    if conv_layers < 2:
        extra_loss += float(10 - 5 * conv_layers)
        conv_layers = 2
    if epochs < 2:
        extra_loss += float(10 - 5 * epochs)
        epochs = 2

    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }  # Define activation function mapping.
    activation = activations["leaky_relu"]  # Get activation function.
    loss_fn = (
        torch.nn.CrossEntropyLoss()
    )  # Use cross-entropy loss for multi-class classification.

    model = Net(
        conv_layers, activation, lr, loss_fn
    )  # Set up neural network with specified hyperparameters.
    model.best_accuracy = 0.0  # Initialize the model's best validation accuracy.

    train_loader, val_loader = get_data_loaders(
        batch_size=8
    )  # Get training and validation data loaders.

    tb_logger = loggers.TensorBoardLogger(
        save_dir=log_path + "lightning_logs"
    )  # Get tensor board logger.

    # Under the hood, the Lightning Trainer handles the training loop details.
    trainer = Trainer(
        max_epochs=epochs,  # Stop training once this number of epochs is reached.
        accelerator="gpu",  # Pass accelerator type.
        devices=[MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE],  # Devices to train on
        enable_progress_bar=True,  # Disable progress bar.
        logger=tb_logger,  # Logger
    )
    print(
        f"[R{MPI.COMM_WORLD.rank:0>2}]: Starting training with configuration: \n"
        f"    Epochs:               {epochs:>3}\n"
        f"    Convolutional layers: {conv_layers:>3}\n"
        f"    Learning rate:        {lr:>8.4f}"
    )
    trainer.fit(  # Run full model training optimization routine.
        model=model,  # Model to train
        train_dataloaders=train_loader,  # Dataloader for training samples
        val_dataloaders=val_loader,  # Dataloader for validation samples
    )
    # Return negative best validation accuracy as an individual's loss.
    print(
        f"#-----------------------------------------#\n"
        f"| [R{MPI.COMM_WORLD.rank:0>2}] Current time: {time.time_ns()} |\n"
        f"#-----------------------------------------#"
    )
    return -model.best_accuracy + extra_loss


if __name__ == "__main__":
    rng = random.Random(MPI.COMM_WORLD.rank)

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.DEBUG,  # logging level DEBUG > INFO > WARNING > ERROR > CRITICAL
        log_file=f"{log_path}islands.log",  # logging path
    )

    propagator: Propagator = [
        pso.Basic(0.729, 1.49445, 1.49445, MPI.COMM_WORLD.rank, limits, rng),
        pso.VelocityClamping(
            0.729, 1.49445, 1.49445, MPI.COMM_WORLD.rank, limits, rng, 0.6
        ),
        pso.Constriction(2.05, 2.05, MPI.COMM_WORLD.rank, limits, rng),
        pso.Canonical(2.05, 2.05, MPI.COMM_WORLD.rank, limits, rng),
    ][int(sys.argv[1])]

    if MPI.COMM_WORLD.rank == 0:
        print("#-----------------------------------#")
        print(f"| Current time: {time.time_ns()} |")
        print("#-----------------------------------#")

    propagator = Conditional(
        pop_size, propagator, pso.InitUniform(limits, rng=rng, rank=MPI.COMM_WORLD.rank)
    )
    propulator = Propulator(
        ind_loss,
        propagator,
        rng=rng,
        generations=num_generations,
        checkpoint_path=log_path + "checkpoints",
    )
    propulator.propulate(debug=0, logging_interval=1)

    if MPI.COMM_WORLD.rank == 0:
        print("#-----------------------------------#")
        print(f"| Current time: {time.time_ns()} |")
        print("#-----------------------------------#")
