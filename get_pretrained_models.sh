#!/usr/bin/env bash
#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

wget https://devpubs.s3.amazonaws.com/ml-research/models/fct/imagenet_250_simclr.pt -P checkpoints
wget https://devpubs.s3.amazonaws.com/ml-research/models/fct/imagenet_500_simclr.pt -P checkpoints
wget https://devpubs.s3.amazonaws.com/ml-research/models/fct/imagenet_1000_simclr.pt -P checkpoints
