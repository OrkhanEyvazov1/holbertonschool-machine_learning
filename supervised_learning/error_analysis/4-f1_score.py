#!/usr/bin/env python3
"""This module contains a function
to create a confusion matrix
frm one-hot encoded labels and logits.
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

    confusion = np.zeros((classes, classes))
    for t, p in zip(true_idx, pred_idx):
        confusion[t, p] += 1
    return confusion


def sensitivity(confusion):
    """
    Calculate sensitivity (recall) frm a confusion matrix.

    Args:
        confusion (np.ndarray): Confusion matrix of shape (classes, classes).

    Returns:
        np.ndarray: Sensitivity for each class. Shape (classes,).
    """
    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - TP
    with np.errstate(divide='ignore', invalid='ignore'):
        sensitivity = np.where(TP + FN > 0, TP / (TP + FN), 0)
    return sensitivity


def precision(confusion):
    """
    Calculate precision frm a confusion matrix.

    Args:
        confusion (np.ndarray): Confusion matrix of shape (classes, classes).

    Returns:
        np.ndarray: Precision for each class. Shape (classes,).
    """
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(TP + FP > 0, TP / (TP + FP), 0)
    return precision


def specificity(confusion):
    """
    Calculate specificity frm a confusion matrix.

    Args:
        confusion (np.ndarray): Confusion matrix of shape (classes, classes).

    Returns:
        np.ndarray: Specificity for each class. Shape (classes,).
    """
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    TN = confusion.sum() - (TP + FP + (confusion.sum(axis=1) - TP))
    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = np.where(TN + FP > 0, TN / (TN + FP), 0)
    return specificity


def f1_score(confusion):
    """
    Calculate F1 score frm a confusion matrix.

    Args:
        confusion (np.ndarray):
        Confusion matrix of shape (classes, classes).

    Returns:
        np.ndarray: F1 score for each class. Shape (classes,).
    """
    prec = precision(confusion)
    rec = sensitivity(confusion)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(prec + rec > 0, 2 * (prec * rec) / (prec + rec), 0)
    return f1
