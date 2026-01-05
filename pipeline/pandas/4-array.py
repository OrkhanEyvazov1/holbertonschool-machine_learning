#!/usr/bin/env python3


'''doc here'''


def array(df):
    """
    Selects the last 10 rows of the 'High'
    and 'Close' columns from the DataFrame
    and converts them into a numpy array.
    """
    # Select the last 10 rows of 'High' and 'Close' columns
    selected_data = df[['High', 'Close']].tail(10)

    # Convert to numpy array and return
    return selected_data.to_numpy()
