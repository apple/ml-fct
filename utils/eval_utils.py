#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Union, Tuple, Dict, Optional
from collections import defaultdict

from sklearn.metrics import average_precision_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions.

    From https://github.com/YantaoShen/openBCT/blob/main/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cmc(
        distmat,
        query_ids=None,
        gallery_ids=None,
        topk=100,
        single_gallery_shot=False,
        first_match_break=False,
        per_class=False,
        verbose=False,
):
    """Compute Cumulative Matching Characteristics metric.

    From https://github.com/YantaoShen/openBCT/blob/main/evaluate/ranking.py
    """
    distmat = distmat.cpu().numpy()
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)

    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = gallery_ids[indices] == query_ids[:, np.newaxis]
    # Compute CMC for each query
    ret = np.zeros(topk)

    if per_class:
        ret_per_class = {cls: np.zeros(topk) for cls in set(gallery_ids)}
        num_valid_queries_per_class = {cls: 0 for cls in set(gallery_ids)}

    num_valid_queries = 0

    iterator = tqdm.tqdm(range(m)) if verbose else range(m)
    for i in iterator:
        if list(query_ids) == list(gallery_ids):
            # If query set is part of gallery set
            valid = np.arange(n)[indices[i]] != np.arange(m)[i]
        else:
            valid = None

        if not np.any(matches[i, valid]):
            continue

        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)

            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1

        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = valid & _unique_sample(ids_dict, len(valid))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]

            delta = 1.0 / (len(index) * repeat)

            for j, k in enumerate(index):
                if k - j >= topk:
                    break

                if first_match_break:
                    ret[k - j] += 1

                    if per_class:
                        ret_per_class[query_ids[i]][k - j] += 1

                    break

                ret[k - j] += delta

                if per_class:
                    ret_per_class[query_ids[i]][k - j] += delta

        num_valid_queries += 1

        if per_class:
            num_valid_queries_per_class[query_ids[i]] += 1

    if num_valid_queries == 0:
        raise RuntimeError("No valid query")

    if per_class:
        return ret.cumsum() / num_valid_queries, {
            cls: ret_class.cumsum() / num_valid_queries_per_class[cls]
            for cls, ret_class in ret_per_class.items()
        }
    else:
        return ret.cumsum() / num_valid_queries


def mean_ap(
        distmat,
        query_ids=None,
        gallery_ids=None,
):
    """Compute Mean Average Precision.

    From https://github.com/YantaoShen/openBCT/blob/main/evaluate/ranking.py
    """
    distmat = distmat.cpu().numpy()
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)

    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = gallery_ids[indices] == query_ids[:, np.newaxis]
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same img
        if list(query_ids) == list(gallery_ids):
            valid = np.arange(n)[indices[i]] != np.arange(m)[i]
        else:
            valid = None
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cosine_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Get pair-wise cosine distances.

    :param x: A torch feature tensor with shape (n, d).
    :param y: A torch feature tensor with shape (n, d).
    :return: Distance tensor between features x and y with shape (n, n).
    """
    x_norm = F.normalize(x, p=1, dim=-1)
    y_norm = F.normalize(y, p=1, dim=-1)

    return 1 - x_norm @ y_norm.T


def l2_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Get pair-wise l2 distances.

    :param x: A torch feature tensor with shape (n, d).
    :param y: A torch feature tensor with shape (n, d).
    :return: Distance tensor between features x and y with shape (n, n).
    """
    return torch.cdist(x, y, p=2)


def cmc_evaluate(
        gallery_model: Union[nn.Module, torch.jit.ScriptModule],
        query_model: Union[nn.Module, torch.jit.ScriptModule],
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        distance_metric: str,
        verbose: bool = False,
        per_class: bool = False,
        compute_map: bool = False,
        **kwargs
) -> Union[Tuple[Tuple[float, float], Optional[float]],
           Tuple[Tuple[Tuple[float, float], Dict], Optional[float]]]:
    """Run CMC and mAP evaluations.

    :param gallery_model: Model to compute gallery features.
    :param query_model: Model to compute query features.
    :param val_loader: Data loader to get gallery/query data.
    :param device: Device to use for computations.
    :param distance_metric: A callable that gets two feature tensors and return
        their distance tensor.
    :param verbose: Whether to be verbose.
    :param per_class: Whether to compute per class CMCs.
    :param compute_map: Whether to compute mean average precision.
    :return: Top-1 CMC, Top-5 CMC, optionally per class CMCs, optionally mAP.
    """
    distance_map = {'l2': l2_distance_matrix,
                    'cosine': cosine_distance_matrix}
    distance_metric = distance_map.get(distance_metric)

    gallery_model.eval()
    query_model.eval()

    gallery_model.to(device)
    query_model.to(device)

    gallery_features = []
    query_features = []
    labels = []

    iterator = tqdm.tqdm(val_loader) if verbose else val_loader

    with torch.no_grad():
        for data, label in iterator:
            data = data.to(device)
            gallery_feature = gallery_model(data)
            query_feature = query_model(data)

            gallery_features.append(gallery_feature.squeeze())
            query_features.append(query_feature.squeeze())

            labels.append(label)

    gallery_features = torch.cat(gallery_features)
    query_features = torch.cat(query_features)
    labels = torch.cat(labels)

    print("=> Computing Distance Matrix")
    distmat = distance_metric(query_features.cpu(), gallery_features.cpu())

    print("=> Starting CMC computation")
    cmc_scores, cmc_scores_per_class = cmc(
        distmat=distmat,
        query_ids=labels.cpu(),
        gallery_ids=labels.cpu(),
        topk=5,
        single_gallery_shot=False,
        first_match_break=True,
        verbose=verbose,
        per_class=True,
    )

    if compute_map:
        mean_ap_out = mean_ap(distmat=distmat, query_ids=labels.cpu(),
                              gallery_ids=labels.cpu())
    else:
        mean_ap_out = None

    if not per_class:
        cmc_out = (cmc_scores[0], cmc_scores[4])
    else:
        cmc_out = (cmc_scores[0], cmc_scores[4]), cmc_scores_per_class

    return cmc_out, mean_ap_out
