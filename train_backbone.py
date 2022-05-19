#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Dict
from argparse import ArgumentParser

import yaml
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn

from trainers import BackboneTrainer
from dataset import SubImageFolder
from utils.net_utils import LabelSmoothing, backbone_to_torchscript
from utils.schedulers import get_policy
from utils.getters import get_model, get_optimizer


def main(config: Dict) -> None:
    """Run training.

    :param config: A dictionary with all configurations to run training.
    :return:
    """
    model = get_model(config.get('arch_params'))

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    trainer = BackboneTrainer()
    optimizer = get_optimizer(model, **config.get('optimizer_params'))
    data = SubImageFolder(**config.get('dataset_params'))
    lr_policy = get_policy(optimizer, **config.get('lr_policy_params'))

    if config.get('label_smoothing') is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothing(smoothing=config.get('label_smoothing'))

    # Training loop
    for epoch in range(config.get('epochs')):
        lr_policy(epoch, iteration=None)

        train_acc1, train_acc5, train_loss = trainer.train(
            train_loader=data.train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print(
            "Train: epoch = {}, Loss = {}, Top 1 = {}, Top 5 = {}".format(
                epoch, train_loss, train_acc1, train_acc5
            ))

        test_acc1, test_acc5, test_loss = trainer.validate(
            val_loader=data.val_loader,
            model=model,
            criterion=criterion,
            device=device,
        )

        print(
            "Test: epoch = {}, Loss = {}, Top 1 = {}, Top 5 = {}".format(
                epoch, test_loss, test_acc1, test_acc5
            ))

    backbone_to_torchscript(model, config.get('output_model_path'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)
    main(read_config)
