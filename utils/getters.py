#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Dict, Optional

import torch
import torch.nn as nn

import models


def get_model(arch_params: Dict, **kwargs) -> nn.Module:
    """Get a model given its configurations.

    :param arch_params: A dictionary containing all model parameters.
    :return: A torch model.
    """
    print("=> Creating model '{}'".format(arch_params.get('arch')))
    model = models.__dict__[arch_params.get('arch')](**arch_params)
    return model


def get_optimizer(model: nn.Module,
                  algorithm: str,
                  lr: float,
                  weight_decay: float,
                  momentum: Optional[float] = None,
                  no_bn_decay: bool = False,
                  nesterov: bool = False,
                  **kwargs) -> torch.optim.Optimizer:
    """Get an optimizer given its configurations.

    :param model: A torch model (with parameters to be trained).
    :param algorithm: String defining what optimization algorithm to use.
    :param lr: Learning rate.
    :param weight_decay: Weight decay coefficient.
    :param momentum: Momentum value.
    :param no_bn_decay: Whether to avoid weight decay for Batch Norm params.
    :param nesterov: Whether to use Nesterov update.
    :return: A torch optimizer objet.
    """
    if algorithm == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if
                     ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if
                       ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if no_bn_decay else weight_decay,
                },
                {"params": rest_params, "weight_decay": weight_decay},
            ],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif algorithm == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    return optimizer
