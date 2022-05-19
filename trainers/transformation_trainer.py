#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Union, Callable

import tqdm
import torch
import torch.nn as nn

from utils.logging_utils import AverageMeter


class TransformationTrainer:
    """Class to train and evaluate transformation models."""
    def __init__(self,
                 old_model: Union[nn.Module, torch.jit.ScriptModule],
                 new_model: Union[nn.Module, torch.jit.ScriptModule],
                 side_info_model: Union[nn.Module, torch.jit.ScriptModule],
                 **kwargs) -> None:
        """Construct a TransformationTrainer module.

        :param old_model: A model that returns old embedding given x.
        :param new_model: A model that returns new embedding given x.
        :param side_info_model: A model that returns side-info given x.
        """

        self.old_model = old_model
        self.old_model.eval()
        self.new_model = new_model
        self.new_model.eval()
        self.side_info_model = side_info_model
        self.side_info_model.eval()

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              model: nn.Module,
              criterion: Callable,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              switch_mode_to_eval: bool) -> float:
        """Run one epoch of training.

        :param train_loader: Data loader to train the model.
        :param model: Model to be trained.
        :param criterion: Loss criterion module.
        :param optimizer: A torch optimizer object.
        :param device: Device the model is on.
        :param switch_mode_to_eval: If true model is train on eval mode!
        :return: Average loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")

        if switch_mode_to_eval:
            model.eval()
        else:
            model.train()

        for i, (images, _) in tqdm.tqdm(
                enumerate(train_loader), ascii=True, total=len(train_loader)
        ):
            images = images.to(device, non_blocking=True)

            with torch.no_grad():
                old_feature = self.old_model(images)
                new_feature = self.new_model(images)
                side_info = self.side_info_model(images)

            recycled_feature = model(old_feature, side_info)
            loss = criterion(new_feature, recycled_feature)

            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses.avg

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 model: nn.Module,
                 criterion: Callable,
                 device: torch.device) -> float:
        """Run validation.

        :param val_loader: Data loader to evaluate the model.
        :param model: Model to be evaluated.
        :param criterion: Loss criterion module.
        :param device: Device the model is on.
        :return: Average loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        model.eval()

        for i, (images, _) in tqdm.tqdm(
                enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            images = images.to(device, non_blocking=True)

            with torch.no_grad():
                old_feature = self.old_model(images)
                new_feature = self.new_model(images)
                side_info = self.side_info_model(images)

                recycled_feature = model(old_feature, side_info)
                loss = criterion(new_feature, recycled_feature)

            losses.update(loss.item(), images.size(0))

        return losses.avg
