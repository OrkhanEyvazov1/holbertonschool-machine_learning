#!/usr/bin/env python3

''' doccc '''

import pandas as pd

def from_numpy(array):
    """
    alphabetical and capitalized column names for a DataFrame created from a NumPy array.
    """
    num_columns = array.shape[1]
    columns = [chr(i) for i in range(65, 65 + num_columns)]

    return pd.DataFrame(array, columns=columns)
