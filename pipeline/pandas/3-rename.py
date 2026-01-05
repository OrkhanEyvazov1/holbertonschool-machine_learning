#!/usr/bin/env python3


'''dccc doc'''


import pandas as pd


'''dc docdco'''


def rename(df):
    '''
    Docstring for rename
    :param df: Description
    '''
    # Rename Timestamp to Datetime
    df = df.rename(columns={'Timestamp': 'Datetime'})
    # Convert the timestamp values to datatime values
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    # Display only the Datetime and Close column
    df = df[['Datetime', 'Close']]
    return df
