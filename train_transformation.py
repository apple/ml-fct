#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Dict

import yaml
from argparse import ArgumentParser

import torch
import torch.nn as nn

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from trainers import TransformationTrainer
from dataset import SubImageFolder
from utils.net_utils import transformation_to_torchscripts
from utils.schedulers import get_policy
from utils.getters import get_model, get_optimizer


def main(config: Dict) -> None:
    """Run training.

    :param config: A dictionary with all configurations to run training.
    :return:
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    model = get_model(config.get('arch_params'))
    old_model = torch.jit.load(config.get('old_model_path'))
    new_model = torch.jit.load(config.get('new_model_path'))

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        old_model = torch.nn.DataParallel(old_model)
        new_model = torch.nn.DataParallel(new_model)

    model.to(device)
    old_model.to(device)
    new_model.to(device)

    if config.get('side_info_model_path') is not None:
        side_info_model = torch.jit.load(config.get('side_info_model_path'))
        if torch.cuda.is_available():
            side_info_model = torch.nn.DataParallel(side_info_model)
        side_info_model.to(device)
    else:
        side_info_model = old_model

    trainer = TransformationTrainer(old_model, new_model, side_info_model)

    optimizer = get_optimizer(model, **config.get('optimizer_params'))
    data = SubImageFolder(**config.get('dataset_params'))
    lr_policy = get_policy(optimizer, **config.get('lr_policy_params'))

    criterion = nn.MSELoss()

    for epoch in range(config.get('epochs')):
        lr_policy(epoch, iteration=None)

        if config.get('switch_mode_to_eval'):
            switch_mode_to_eval = epoch >= config.get('epochs') / 2
        else:
            switch_mode_to_eval = False

        train_loss = trainer.train(
            train_loader=data.train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            switch_mode_to_eval=switch_mode_to_eval,
        )

        print("Train: epoch = {}, Average Loss = {}".format(epoch, train_loss))

        # evaluate on validation set
        test_loss = trainer.validate(
            val_loader=data.val_loader,
            model=model,
            criterion=criterion,
            device=device,
        )

        print("Test: epoch = {}, Average Loss = {}".format(
            epoch, test_loss
        ))

    transformation_to_torchscripts(old_model, side_info_model, model,
                                   config.get('output_transformation_path'),
                                   config.get(
                                       'output_transformed_old_model_path'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)
    main(read_config)
