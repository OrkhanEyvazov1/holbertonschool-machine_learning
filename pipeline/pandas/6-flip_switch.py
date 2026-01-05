#!/usr/bin/env python3


''' doc here '''


def flip_switch(df):
    """

    Args:
        df: Input pandas DataFrame, assumed to have
        a datetime index or similar.

    Returns:
        The transformed pandas DataFrame.
    """
    # Sort the index in descending order (reverse chronological)
    df_sorted = df.sort_index(ascending=False)

    # Transpose the dataframe (swap rows and columns)
    df_transposed = df_sorted.T

    return df_transposed
