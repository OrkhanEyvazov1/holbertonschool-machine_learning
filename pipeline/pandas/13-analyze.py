#!/usr/bin/env python3


'''doc here'''


def analyze(df):
    ''' doc here'''
    return df.drop(columns=['Timestamp']).describe()

