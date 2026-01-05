#!/usr/bin/env python3


'''doc here'''


def slice(df):
    """
    Takes a pd.DataFrame, extracts
    specific columns, and selects every 60th row.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The sliced DataFrame.
    """
    # Extract specific columns
    df_subset = df[['High', 'Low', 'Close', 'Volume_BTC']]

    # Select every 60th row
    # The syntax is df[start:stop:step]
    sliced_df = df_subset.iloc[::60]

    return sliced_df
