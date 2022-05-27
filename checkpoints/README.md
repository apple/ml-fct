# Pretrained models
We provide pretrained side-information models trained with [SimCLR](https://arxiv.org/abs/2002.05709)
using the same training setup as in [Stochastic Contrastive Learning](https://arxiv.org/abs/2110.00552):

- [ResNet50-128-ImageNet250](https://docs-assets.developer.apple.com/ml-research/models/fct/imagenet_250_simclr.pt)
- [ResNet50-128-ImageNet500](https://docs-assets.developer.apple.com/ml-research/models/fct/imagenet_500_simclr.pt)
- [ResNet50-128-ImageNet1000](https://docs-assets.developer.apple.com/ml-research/models/fct/imagenet_1000_simclr.pt)

The above models have [ResNet50](https://arxiv.org/abs/1512.03385) architecture with feature dimension of 128. The models are trained on the first 250, 500, and 1000 classes of ImageNet, respectively.
