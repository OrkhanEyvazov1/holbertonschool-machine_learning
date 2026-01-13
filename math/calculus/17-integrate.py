#!/usr/bin/env python3


'''poly of integral'''


def poly_integral(poly, C=0):
    '''poly of integral'''
    if type(C) is not int:
        return None

    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)) or isinstance(coef, bool):
            return None

    integral = [C]
    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        if val % 1 == 0:
            val = int(val)

        integral.append(val)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
