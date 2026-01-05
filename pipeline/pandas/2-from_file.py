#!/usr/bin/env python3


"""doc"""


import pandas as pd

'''doc'''

def from_file(filename, delimiter):
    '''
    Docstring for from_file

    :param filename: Description
    :param delimiter: Description
    '''
    return pd.read_csv(filename, delimiter=delimiter)
