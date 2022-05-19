#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Dict
from argparse import ArgumentParser

import yaml
import torch

from dataset import SubImageFolder
from utils.eval_utils import cmc_evaluate


def main(config: Dict) -> None:
    """Run evaluation.

    :param config: A dictionary with all configurations to run evaluation.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    # Load models:
    gallery_model = torch.jit.load(config.get('gallery_model_path'))
    query_model = torch.jit.load(config.get('query_model_path'))

    data = SubImageFolder(**config.get('dataset_params'))

    cmc_out, mean_ap_out = cmc_evaluate(
        gallery_model,
        query_model,
        data.val_loader,
        device,
        **config.get('eval_params')
    )

    if config.get('eval_params').get('per_class'):
        print('CMC Top-1 = {}, CMC Top-5 = {}'.format(*cmc_out[0]))
        print('Per class CMC: {}'.format(cmc_out[1]))
    else:
        print('CMC Top-1 = {}, CMC Top-5 = {}'.format(*cmc_out))

    if mean_ap_out is not None:
        print('mAP = {}'.format(mean_ap_out))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)
    main(read_config)
