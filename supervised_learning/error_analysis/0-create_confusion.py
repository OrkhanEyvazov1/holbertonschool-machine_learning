#!/usr/bin/env python3
"""This module contains a function
to create a confusion matrix
one-hot encoded labels and logits.
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Create a confusion matrix frm one-hot encoded labels and logits.

    Args:
        labels (np.ndarray): True labels, shape (m, classes).
        logits (np.ndarray): Predicted labels, shape (m, classes).

    Returns:
        np.ndarray: Confusion matrix of shape (classes, classes).
    """
    if labels.ndim != 2 or logits.ndim != 2:
        raise ValueError("labels and logits must be 2D arrays")
    if labels.shape != logits.shape:
        raise ValueError("labels and logits must have the same shape")

    classes = labels.shape[1]
    true_idx = labels.argmax(axis=1)
    pred_idx = logits.argmax(axis=1)

    confusion = np.zeros((classes, classes), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        confusion[t, p] += 1
    return confusion
