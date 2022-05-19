#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os
import pickle

from PIL import Image
from tqdm import tqdm

from torchvision.datasets import CIFAR100


def save_pngs(untar_dir: str, split: str) -> None:
    """Save loaded data as png files.

    :param untar_dir: Path to untared dataset.
    :param split: Split name (e.g., train, test)
    """
    split_map = {'train': 'training',
                 'test': 'validation'}
    split_dir = os.path.join(untar_dir, split_map.get(split))

    os.makedirs(split_dir, exist_ok=True)

    for i in range(100):
        class_dir = os.path.join(split_dir, str(i))
        os.makedirs(class_dir, exist_ok=True)

    with open(os.path.join(untar_dir, split), 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')

    data = data_dict.get('data')  # numpy array
    # Reshape and cast
    data = data.reshape(data.shape[0], 3, 32, 32)
    data = data.transpose(0, 2, 3, 1).astype('uint8')

    labels = data_dict.get('fine_labels')

    for i, (datum, label) in tqdm(enumerate(zip(data, labels)),
                                  total=len(labels)):
        image = Image.fromarray(datum)
        image = image.convert('RGB')
        file_path = os.path.join(split_dir, str(label), '{}.png'.format(i))
        image.save(file_path)


def get_cifar100() -> None:
    """Get and reformat cifar100 dataset.

    See https://www.cs.toronto.edu/~kriz/cifar.html for dataset description.
    """
    data_store_dir = 'data_store'

    if not os.path.exists(data_store_dir):
        os.makedirs(data_store_dir)

    dataset = CIFAR100(root=data_store_dir, download=True)

    # Load files and convert to PNG
    untar_dir = os.path.join(data_store_dir, dataset.base_folder)
    save_pngs(untar_dir, 'test')
    save_pngs(untar_dir, 'train')


if __name__ == '__main__':
    get_cifar100()
