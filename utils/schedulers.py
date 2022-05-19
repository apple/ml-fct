#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Callable, Optional

import torch
import numpy as np

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "get_policy"]


def get_policy(optimizer: torch.optim.Optimizer,
               algorithm: str,
               **kwargs) -> Callable:
    """Get learning policy given its configurations.

    :param optimizer: A torch optimizer.
    :param algorithm: Name of the learning rate scheduling algorithm.
    :return: A callable to adjust learning rate for each epoch.
    """
    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
    }
    return out_dict[algorithm](optimizer, **kwargs)


def assign_learning_rate(optimizer: torch.optim.Optimizer,
                         new_lr: float) -> None:
    """Update lr parameter of an optimizer.

    :param optimizer: A torch optimizer.
    :param new_lr: updated value of learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer: torch.optim.Optimizer,
                warmup_length: int,
                lr: float,
                **kwargs) -> Callable:
    """Get lr adjustment callable with constant schedule.

    :param optimizer: A torch optimizer.
    :param warmup_length: Number of epochs for initial warmup.
    :param lr: Nominal learning rate value.
    :return: A callable to adjust learning rate per epoch.
    """

    def _lr_adjuster(epoch: int, iteration: Optional[int]) -> float:
        """Get updated learning rate.

        :param epoch: Epoch number.
        :param iteration: Iteration number.
        :return: Updated learning rate value.
        """
        if epoch < warmup_length:
            new_lr = _warmup_lr(lr, warmup_length, epoch)
        else:
            new_lr = lr

        assign_learning_rate(optimizer, new_lr)

        return new_lr

    return _lr_adjuster


def cosine_lr(optimizer: torch.optim.Optimizer,
              warmup_length: int,
              epochs: int,
              lr: float,
              **kwargs) -> Callable:
    """Get lr adjustment callable with cosine schedule.

    :param optimizer: A torch optimizer.
    :param warmup_length: Number of epochs for initial warmup.
    :param epochs: Epoch number.
    :param lr: Nominal learning rate value.
    :return: A callable to adjust learning rate per epoch.
    """

    def _lr_adjuster(epoch: int, iteration: Optional[int]) -> float:
        """Get updated learning rate.

        :param epoch: Epoch number.
        :param iteration: Iteration number.
        :return: Updated learning rate value.
        """
        if epoch < warmup_length:
            new_lr = _warmup_lr(lr, warmup_length, epoch)
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            new_lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lr

        assign_learning_rate(optimizer, new_lr)

        return new_lr

    return _lr_adjuster


def multistep_lr(optimizer: torch.optim.Optimizer,
                 lr_gamma: float,
                 lr_adjust: int,
                 lr: float,
                 **kwargs) -> Callable:
    """Get lr adjustment callable with multi-step schedule.

    :param optimizer: A torch optimizer.
    :param lr_gamma: Learning rate decay factor.
    :param lr_adjust: Number of epochs to apply decay.
    :param lr: Nominal Learning rate.
    :return: A callable to adjust learning rate per epoch.
    """

    def _lr_adjuster(epoch: int, iteration: Optional[int]) -> float:
        """Get updated learning rate.

        :param epoch: Epoch number.
        :param iteration: Iteration number.
        :return: Updated learning rate value.
        """
        new_lr = lr * (lr_gamma ** (epoch // lr_adjust))

        assign_learning_rate(optimizer, new_lr)

        return new_lr

    return _lr_adjuster


def _warmup_lr(base_lr: float,
               warmup_length: int,
               epoch: int) -> float:
    """Get updated lr after applying initial warmup.

    :param base_lr: Nominal learning rate.
    :param warmup_length: Number of epochs for initial warmup.
    :param epoch: Epoch number.
    :return: Warmup-updated learning rate.
    """
    return base_lr * (epoch + 1) / warmup_length
