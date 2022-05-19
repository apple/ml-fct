#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Optional

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convenience convolution module."""

    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 normalizer: Optional[nn.Module] = nn.BatchNorm2d,
                 activation: Optional[nn.Module] = nn.ReLU) -> None:
        """Construct a ConvBlock module.

        :param channels_in: Number of input channels.
        :param channels_out: Number of output channels.
        :param kernel_size: Size of the kernel.
        :param stride: Size of the convolution stride.
        :param normalizer: Optional normalization to use.
        :param activation: Optional activation module to use.
        """
        super().__init__()

        self.conv = nn.Conv2d(channels_in, channels_out,
                              kernel_size=kernel_size, stride=stride,
                              bias=normalizer is None,
                              padding=kernel_size // 2)
        if normalizer is not None:
            self.normalizer = normalizer(channels_out)
        else:
            self.normalizer = None
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        x = self.conv(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MLP_BN_SIDE_PROJECTION(nn.Module):
    """FCT transformation module."""

    def __init__(self,
                 old_embedding_dim: int,
                 new_embedding_dim: int,
                 side_info_dim: int,
                 inner_dim: int = 2048,
                 **kwargs) -> None:
        """Construct MLP_BN_SIDE_PROJECTION module.

        :param old_embedding_dim: Size of the old embeddings.
        :param new_embedding_dim: Size of the new embeddings.
        :param side_info_dim: Size of the side-information.
        :param inner_dim: Dimension of transformation MLP inner layer.
        """
        super().__init__()

        self.inner_dim = inner_dim
        self.p1 = nn.Sequential(
            ConvBlock(old_embedding_dim, 2 * old_embedding_dim),
            ConvBlock(2 * old_embedding_dim, 2 * new_embedding_dim),
        )

        self.p2 = nn.Sequential(
            ConvBlock(side_info_dim, 2 * side_info_dim),
            ConvBlock(2 * side_info_dim, 2 * new_embedding_dim),
        )

        self.mixer = nn.Sequential(
            ConvBlock(4 * new_embedding_dim, self.inner_dim),
            ConvBlock(self.inner_dim, self.inner_dim),
            ConvBlock(self.inner_dim, new_embedding_dim, normalizer=None,
                      activation=None)
        )

    def forward(self,
                old_feature: torch.Tensor,
                side_info: torch.Tensor) -> torch.Tensor:
        """Apply forward pass.

        :param old_feature: Old embedding.
        :param side_info: Side-information.
        :return: Recycled old embedding compatible with new embeddings.
        """
        x1 = self.p1(old_feature)
        x2 = self.p2(side_info)
        return self.mixer(torch.cat([x1, x2], dim=1))
