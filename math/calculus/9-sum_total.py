#!/usr/bin/env python3


'''doc heree'''


def summation_i_squared(n):
    '''sum of i squared'''
    if not isinstance(n, int) or n < 0:
        return None
    
    result = (n * (n + 1) * (2 * n + 1)) // 6
    return result
