#!/usr/bin/env python3
"""7-early_stopping.py"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early.

    Args:
        cost: current cost of the network
        opt_cost: optimal cost of the network
        threshold: maximum acceptable difference between cost and opt_cost
        patience: patience count for early stopping
        count: count of how many times cost has been greater than opt_cost + threshold
    Returns:
        True if you should stop gradient descent early, or False otherwise
    """
    if cost > opt_cost + threshold:
        count += 1
    else:
        count = 0
    return count >= patience
