import numpy as np


def channel_first(image):
    return np.transpose(image, (2, 0, 1))


def channel_last(image):
    return np.transpose(image, (1, 2, 0))


def get_acc(preds, target):
    return np.sum(preds == target)/len(preds)
