#!/usr/bin/env python3


'''doc heree'''

def summation_i_squared(n):
    """Calculate the sum of i squared from 1 to n."""
    if type(n) is not int or n < 0:
        return None

    result = (n * (n + 1) * (2 * n + 1)) // 6
    return result
