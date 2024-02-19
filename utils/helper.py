import torch
import argparse


def dict2namespace(config):
    namespace = argparse.Namespace()

    for k, v in config.items():
        if isinstance(v, dict):
            new_v = dict2namespace(v)
        else:
            new_v = v

        setattr(namespace, k, new_v)

    return namespace


def cycle(dl):
    while True:
        for data in dl:
            yield data


def get_accuracy(pred, gt):
    """
    pred: probabilities for classes
    gt: one-hot encoded ground truth labeling
    """

    pred = torch.argmax(pred, dim=-1)
    gt = torch.argmax(gt, dim=-1)

    return (pred == gt).int().cpu()
