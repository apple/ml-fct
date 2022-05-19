#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Tuple, Callable

import tqdm
import torch
import torch.nn as nn

from utils.logging_utils import AverageMeter
from utils.eval_utils import accuracy


class BackboneTrainer():
    """Class to train and evaluate backbones."""

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              model: nn.Module,
              criterion: Callable,
              optimizer: torch.optim.Optimizer,
              device: torch.device) -> Tuple[float, float, float]:
        """Run one epoch of training.

        :param train_loader: Data loader to train the model.
        :param model: Model to be trained.
        :param criterion: Loss criterion module.
        :param optimizer: A torch optimizer object.
        :param device: Device the model is on.
        :return: average of top-1, top-5, and loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        model.train()

        for i, (images, target) in tqdm.tqdm(
                enumerate(train_loader), ascii=True, total=len(train_loader)
        ):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output, _ = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return top1.avg, top5.avg, losses.avg

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 model: nn.Module,
                 criterion: Callable,
                 device: torch.device) -> Tuple[float, float, float]:
        """Run validation.
        
        :param val_loader: Data loader to evaluate the model.
        :param model: Model to be evaluated.
        :param criterion: Loss criterion module.
        :param device: Device the model is on.
        :return: average of top-1, top-5, and loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        model.eval()

        with torch.no_grad():
            for i, (images, target) in tqdm.tqdm(
                    enumerate(val_loader), ascii=True, total=len(val_loader)
            ):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output, _ = model(images)

                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

        return top1.avg, top5.avg, losses.avg
