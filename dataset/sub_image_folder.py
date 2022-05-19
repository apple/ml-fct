#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os

import torch
from torchvision import datasets

from .data_transforms import get_data_transforms


class SubImageFolder:
    """Class to support training on subset of classes."""

    def __init__(self, name: str, data_root: str, num_workers: int,
                 batch_size: int,
                 num_classes=None) -> None:
        """Construct a SubImageFolder module.

        :param name: Name of the dataset (e.g., cifar100, imagenet).
        :param data_root: Path to a directory with training and validation
            subdirs of the dataset.
        :param num_workers: Number of workers for data loader.
        :param batch_size: Size of the batch per GPU.
        :param num_classes: Number of classes to use for training. This should
            be smaller ot equal than the total number of classes in the
            dataset. Not that for evaluation we use all classes.
        """
        super(SubImageFolder, self).__init__()

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": num_workers,
                  "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "training")
        valdir = os.path.join(data_root, "validation")

        train_transforms, val_transforms = get_data_transforms(name)

        self.train_dataset = datasets.ImageFolder(
            traindir,
            train_transforms,
        )

        # Filtering out some classes
        if num_classes is not None:
            self.train_dataset.imgs = [
                (path, cls_num)
                for path, cls_num in self.train_dataset.imgs
                if cls_num < num_classes
            ]

        self.train_dataset.samples = self.train_dataset.imgs

        self.train_sampler = None

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
            **kwargs
        )

        # Note: for evaluation we use all classes.
        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                val_transforms,
            ),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
