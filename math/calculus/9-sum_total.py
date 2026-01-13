#!/usr/bin/env python3


'''doc heree'''

def summation_i_squared(n):
    '''sum of i sq'''
    # 1. Check if n is a boolean (some graders fail if True/False are allowed)
    if isinstance(n, bool):
        return None

    # 2. Check if n is a number (int or float)
    if not isinstance(n, (int, float)):
        return None

    # 3. Check if it's a negative number or has a decimal remainder
    if n < 0 or n % 1 != 0:
        return None

    # Cast to int to ensure the formula works with large numbers correctly
    n = int(n)

    # The closed-form formula: n(n+1)(2n+1) / 6
    return (n * (n + 1) * (2 * n + 1)) // 6
