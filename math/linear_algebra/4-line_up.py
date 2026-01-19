#!/usr/bin/env python3


'''doc here '''


def add_arrays(arr1, arr2):
    ''' adds two arrays element-wise '''
    if len(arr1) != len(arr2):
        raise None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
