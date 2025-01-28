import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, basecontainer
from torch import nn

T = TypeVar("T")


def load_config(
    config_path: Union[str, Path],
    config_name: str,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load a Hydra configuration given a path and file name
    Parameters
    ----------
    config_path: str
        Path to the Hydra config folder.
    config_name: str
        Name of the Hydra config file.
    overrides: Optional[List[str]]
        Overrides for values defined in the config.
        See https://hydra.cc/docs/advanced/override_grammar/basic/ for syntax
    """
    # Hydra expects the config_path to be relative to the _calling file_, i.e. this file.
    # Since we're taking config_path as a command line argument we want to treat it as
    # relative to the current working directory. We do the conversion here.
    desired_path = Path.cwd() / config_path
    print(desired_path)
    relative_path = os.path.relpath(desired_path, Path(__file__).parent)
    print(relative_path)
    with hydra.initialize(version_base=None, config_path=relative_path):
        config = hydra.compose(config_name=config_name, overrides=overrides)
    return config


def _maybe_instantiate(config: Union[T, DictConfig]) -> T:
    if isinstance(config, basecontainer.BaseContainer):
        return instantiate(config)
    else:
        return config


def get_params_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Split the model parameters into groups with/without weight decay.
    Bias and normalization parameters should not have weight decay, all others should.
    """
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases or norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [
        {"params": regularized, "weight_decay": weight_decay},
        {"params": not_regularized, "weight_decay": 0.0},
    ]


def get_lr_scheduler(
    dataloader: pl.LightningDataModule,
    num_nodes: int = 1,
    batch_size: int = 128,
    lr: float = 0.0005,
    min_lr: float = 1e-6,
    scheduling_epochs: int = 100,
    warmup_epochs: int = 10,
    warmup_initial_lr: float = 0.0,
    scaling_rule: str = "deit",
) -> np.ndarray:
    """Create an array containing the LR schedule.

    Parameters
    ----------
    dataloader: pl.LightningDataModule
        The dataloader being used for training.
    batch_size: int
        batch size
    lr: float
        Initial learning rate
    min_lr: float
        Final learning rate.
    scheduling_epochs: int
        The number of epochs to schedule for. Must be >= the number of training epochs.
    warmup_epochs: int
        Number of epochs for linear warmup.
    warmup_initial_lr: float
        Initial learning rate for linear warmup.
    scaling_rule: str
        Scaling rule for the learning rate. Must be "linear" or "sqrt".
    """

    world_size = torch.cuda.device_count() * num_nodes

    if scaling_rule == "linear":
        base_value = lr * (batch_size * world_size) / 256.0
    elif scaling_rule == "deit":
        base_value = lr * (batch_size * world_size) / 512.0
    elif scaling_rule == "sqrt":
        base_value = lr * math.sqrt(batch_size * world_size / 1024.0)
    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")

    try:  # when pytorch lightning datamodule
        len_train_dataloader = len(dataloader.dataset_train) // dataloader.batch_size

        print("len(train_dataloader), world_size", len_train_dataloader, world_size)

        lr_schedule = cosine_scheduler(
            base_value,
            min_lr,
            scheduling_epochs,
            len_train_dataloader // world_size,
            warmup_epochs=warmup_epochs,
            warmup_initial_value=warmup_initial_lr,
        )
    except Exception:
        lr_schedule = cosine_scheduler(
            base_value,
            min_lr,
            scheduling_epochs,
            len(dataloader) // world_size,
            warmup_epochs=warmup_epochs,
            warmup_initial_value=warmup_initial_lr,
        )

    return lr_schedule


def cosine_scheduler(
    base_value: float,
    final_value: float,
    scheduling_epochs: int,
    iter_per_epoch: int,
    warmup_epochs: int = 0,
    warmup_initial_value: float = 0.0,
) -> np.ndarray:
    """Cosine rate schedule with optional linear warmup.
    Used for learning rate, weight decay, and temperature scheduling.
    Parameters
    ----------
    base_value: float
        Initial value.
    final_value: float
        Final value.
    scheduling_epochs: int
        Total epochs to be used to compute the scheduler.
    niter_per_ep: int
        Total number of steps per epoch. Should generally be num_batches // batch_size.
    warmup_epochs: int
        Total number of epochs for linear warmup.
    warmup_initial_value: float
        Rate at the start of the warmup.

    Returns
    -------
        Numpy array of length epochs * niter_per_ep containing the rate
        at each training step.
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * iter_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(warmup_initial_value, base_value, warmup_iters)

    iters = np.arange(scheduling_epochs * iter_per_epoch - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == scheduling_epochs * iter_per_epoch
    return schedule
