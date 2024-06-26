import pandas as pd
from pathlib import Path

#%%

tabular_folder = '/home/tyarosevich/Projects/decrypto/data/'
lookup_path = Path(tabular_folder + 'crypto_lookup_202304251523.csv')
crypto_prices_path = Path(tabular_folder + 'raw_crypto_prices_202304251523.csv')
stock_index_path = Path(tabular_folder + 'raw_stock_indexes_202304251524.csv')
tweets_path = Path(tabular_folder + 'raw_tweets_202304251524.csv')
stock_index_lookup_path = Path(tabular_folder + 'stock_index_lookup_202304251536.csv')
hourly_bitcoin_path = Path(tabular_folder + 'Bittrex_BTCUSD_1h.csv')
hourly_ether_path = Path(tabular_folder + 'Bittrex_ETHUSD_1h.csv')

df_ether_hourly = pd.read_csv(hourly_ether_path, header=1)
df_ether_hourly['date'] = pd.to_datetime(df_ether_hourly['date'])
df_bitcoin_hourly = pd.read_csv(hourly_bitcoin_path, header=1)
df_bitcoin_hourly['date'] = pd.to_datetime(df_bitcoin_hourly['date'])
df_crypto_lookup = pd.read_csv(lookup_path)
df_crypto_prices = pd.read_csv(crypto_prices_path)
df_crypto_prices['created_at'] = pd.to_datetime(df_crypto_prices['created_at'])
df_stock_prices = pd.read_csv(stock_index_path)
df_stock_prices['created_at'] = pd.to_datetime(df_stock_prices['created_at'])
df_tweets = pd.read_csv(tweets_path)
df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'])
df_stock_index_lookup = pd.read_csv(stock_index_lookup_path)
for df in [df_tweets, df_crypto_prices, df_stock_prices]:
    df.rename(columns={'created_at': 'date'}, inplace=True)

#%% Hourly historical Indexes
hourly_index_col_names = ['date', 'hour', 'open', 'high', 'low', 'close', 'volume']
nasdaq_hourly_path = Path(r'/home/tyarosevich/Projects/decrypto/data/nasdaq_hourly.csv')
s_and_p_hourly_path = Path(r'/home/tyarosevich/Projects/decrypto/data/s_and_p_hourly.csv')
dowjones_hourly_path = Path(r'/home/tyarosevich/Projects/decrypto/data/dowjones_hourly.csv')
df_nasdaq_hourly = pd.read_csv(nasdaq_hourly_path, low_memory=False, names=hourly_index_col_names, delimiter=';')
df_dowjones_hourly = pd.read_csv(dowjones_hourly_path, low_memory=False, names=hourly_index_col_names, delimiter=';')
df_s_and_p_hourly = pd.read_csv(s_and_p_hourly_path, low_memory=False, names=hourly_index_col_names, delimiter=';')

