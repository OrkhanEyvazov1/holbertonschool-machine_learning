#!/usr/bin/env python3


''' doc here '''


import pandas as pd


index = __import__('10-index').index

def concat(df1, df2):
    """
    Concatenates two dataframes with specific filtering and indexing.
    """
    # Index both dataframes on their Timestamp columns
    df1 = index(df1)
    df2 = index(df2)

    # Filter df2 for timestamps up to and including 1417411920
    df2_filtered = df2.loc[:1417411920]

    # Concatenate df2 (bitstamp) to the top of df1 (coinbase) with keys
    concat_df = pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])

    return concat_df
