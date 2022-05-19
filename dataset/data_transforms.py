#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Tuple
from torchvision import transforms


def imagenet_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transformations for Imagenet."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transforms, val_transforms


def cifar100_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transformations for Cifar100.

    Note that these are not optimal transformations (including normalization),
    yet provided for quick experimentation similar to Imagenet
    (and its corresponding side-information model).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transforms, val_transforms


data_transforms_map = {
    'cifar100': cifar100_transforms,
    'imagenet': imagenet_transforms
}


def get_data_transforms(dataset_name: str) -> Tuple[transforms.Compose,
                                                    transforms.Compose]:
    """Get training and validation transforms of a dataset.

    :param dataset_name: Name of the dataset (e.g., cifar100, imagenet)
    :return: Tuple of training and validation transformations.
    """
    return data_transforms_map.get(dataset_name)()
