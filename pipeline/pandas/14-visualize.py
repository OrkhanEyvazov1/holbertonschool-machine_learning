#!/usr/bin/env python3


''' doc here '''


def process_and_plot(df):
    ''' doc here '''
    df = df.drop(columns=['Weighted_Price'])
    df = df.rename(columns={'Timestamp': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    df = df.set_index('Date')
    df['Close'] = df['Close'].ffill()

    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    df_2017 = df[df.index.year >= 2017]

    grouped = df_2017.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    plt.figure(figsize=(15, 7))
    plt.plot(grouped.index, grouped['Close'], label='Close Price')
    plt.title('Daily Crypto Data from 2017')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return grouped
