#!/usr/bin/env python3


''' doc here'''


import pandas as pd


index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two dataframes with a hierarchical index.

    Args:
        df1: First pandas DataFrame (coinbase).
        df2: Second pandas DataFrame (bitstamp).

    Returns:
        The concatenated pandas DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)
    r = range(1417411980, 1417417980 + 60, 60)
    df1_filtered = df1.loc[df1.index.isin(r)]
    df2_filtered = df2.loc[df2.index.isin(r)]

    df = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp', 'coinbase'])
    
    df = df.swaplevel(0, 1).sort_index()

    return df
