#!/usr/bin/env python3
"""3-optimum.py
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(kmin, int) or kmin <= 0:
            return None, None
        if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
            return None, None
        if not isinstance(iterations, int) or iterations <= 0:
            return None, None

        if kmax is None:
            kmax = X.shape[0]

        if kmin >= kmax:
            return None, None

        results = []
        d_vars = []
        var_kmin = None

        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            results.append((C, clss))
            var = variance(X, C)
            if k == kmin:
                var_kmin = var
            d_vars.append(var_kmin - var)

        return results, d_vars
    except Exception:
        return None, None
