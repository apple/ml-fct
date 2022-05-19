#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Union

import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing: float = 0.0):
        """Construct LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply forward pass.

        :param x: Logits tensor.
        :param target: Ground truth target classes.
        :return: Loss tensor.
        """
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FeatureExtractor(nn.Module):
    """A wrapper class to return only features (no logits)."""

    def __init__(self,
                 model: Union[nn.Module, torch.jit.ScriptModule]) -> None:
        """Construct FeatureExtractor module.

        :param model: A model that outputs both logits and features.
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass.

        :param x: Input data.
        :return: Feature tensor computed for x.
        """
        _, feature = self.model(x)
        return feature


class TransformedOldModel(nn.Module):
    """A wrapper class to return transformed features."""

    def __init__(self,
                 old_model: Union[nn.Module, torch.jit.ScriptModule],
                 side_model: Union[nn.Module, torch.jit.ScriptModule],
                 transformation: Union[
                     nn.Module, torch.jit.ScriptModule]) -> None:
        """Construct TransformedOldModel module.

        :param old_model: Old model.
        :param side_model: Side information model.
        :param transformation: Transformation model.
        """
        super().__init__()
        self.old_model = old_model
        self.transformation = transformation
        self.side_info_model = side_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass.

        :param x: Input data
        :return: Transformed old feature.
        """
        old_feature = self.old_model(x)
        side_info = self.side_info_model(x)
        recycled_feature = self.transformation(old_feature, side_info)
        return recycled_feature


def prepare_model_for_export(
        model: Union[nn.Module, torch.jit.ScriptModule]
) -> Union[nn.Module, torch.jit.ScriptModule]:
    """Prepare a model to be exported as torchscript."""
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    model.cpu()
    return model


def backbone_to_torchscript(model: Union[nn.Module, torch.jit.ScriptModule],
                            output_model_path: str) -> None:
    """Convert a backbone model to torchscript.

    :param model: A backbone model to be converted to torch script.
    :param output_model_path: Path to save torch script.
    """
    model = prepare_model_for_export(model)
    f = FeatureExtractor(model)
    model_script = torch.jit.script(f)
    torch.jit.save(model_script, output_model_path)


def transformation_to_torchscripts(
        old_model: Union[nn.Module, torch.jit.ScriptModule],
        side_model: Union[nn.Module, torch.jit.ScriptModule],
        transformation: Union[nn.Module, torch.jit.ScriptModule],
        output_transformation_path: str,
        output_transformed_old_model_path: str) -> None:
    """Convert a transformation model to torchscript.

    :param old_model: Old model.
    :param side_model: Side information model.
    :param transformation: Transformation model.
    :param output_transformation_path: Path to store transformation torch
        script.
    :param output_transformed_old_model_path: Path to store combined old and
        transformation models' torch script.
    """
    transformation = prepare_model_for_export(transformation)
    old_model = prepare_model_for_export(old_model)
    side_model = prepare_model_for_export(side_model)

    model_script = torch.jit.script(transformation)
    torch.jit.save(model_script, output_transformation_path)

    f = TransformedOldModel(old_model, side_model, transformation)
    model_script = torch.jit.script(f)
    torch.jit.save(model_script, output_transformed_old_model_path)
